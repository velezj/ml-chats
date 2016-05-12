#####
## This file is subject to the terms and conditions defined in
## file 'LICENSE', which is part of this source code package.
####

import logging
logger = logging.getLogger( __name__ )

import copy
import math

import numpy as np

from sklearn import gaussian_process 

import autocorrelation
import categorical_distribution



##=======================================================================

##
# Estimate the period of a sequence using the autocorrelation.
# Returns a distirbution over the period (cat_distribution)
def period_distribution( x,
                         prior = categorical_distribution.cat_distibution({}) ):

    # first, calculate autocorrelation
    ac = autocorrelation.autocorrelation_estimate( x )

    # Ok, now ignore first datapoint and take the top 5 percent of the
    # height
    n = len(x)
    height_thresh = 2 * np.std( ac[1:n/2] )
    height_thresh = 0.5 * max( ac[1:n/2] )

    # now find continuous ranges with value above the threhold
    ranges = [] # [start,end) intervals
    current_start = None
    for i, a in enumerate(ac[1:]):
        idx = i + 1
        if a >= height_thresh:

            # add to range if we have one
            if current_start is not None:
                pass
            elif current_start is None:
                current_start = idx

        else:

            # stop range if we were accruing one
            if current_start is not None:
                ranges.append( (current_start, idx ) )
                current_start = None

    # Ok, if we have no ranges then there is no period
    if len(ranges) < 2:
        return categorical_distribution.cat_distibution( { 0: 1.0 } )

    # ok, grab the midpoint of each range
    mids = map(lambda (a,b): a + (b-a) / 2 , ranges )

    # Ok, calculate probability for different period lengths
    counts = {}
    for a,b in zip(mids,mids[1:]):
        k = b - a
        if k not in counts:
            counts[ k ] = 0.0
        counts[ k ] += 1.0

    # Ok, add prior counts
    for k,c in prior.counts.iteritems():
        counts[ k ] += c

    # return the distribution
    return categorical_distribution.cat_distibution( counts )



##=======================================================================

##
# Returns an estimate of the period along with a value with how likely
# the signal is actually periodic with the given period
def estimate_period( x ):

    n = len(x)

    # Ok, build up a prior over the period given the signal length
    prior = categorical_distribution.cat_distibution( {} )

    # compute the period distribution
    period_dist = period_distribution( x, prior=prior )

    # ok, judge how well we think the signal is actually periodic
    period_interval, mass = period_dist.credible_interval( 0.5 )

    # too large an interval means no
    if period_interval[1] - period_interval[0] > 3:
        return None, 0.0

    # If the average period < 2 return none
    avg_period = period_interval[0] + ( period_interval[1] - period_interval[0] ) / 2.0
    if avg_period < 2:
        return None, 0.0

    # widen the period interval by one to either side if it is a point interval
    if period_interval[0] == period_interval[1]:
        period_interval = ( period_interval[0] - 1.0,
                            period_interval[1] + 1.0 )

    # Ok, we have a peak but did we find enough of hte peaks according to
    # the raw counts    
    num_expected_peaks = int( math.floor( n / avg_period ) )
    num_found_peaks = 0
    for k, c in period_dist.counts.iteritems():
        if k >= period_interval[0] and k <= period_interval[1]:
            num_found_peaks += c
    found_peak_thresh = 0.75
    if float(num_found_peaks) / num_expected_peaks < found_peak_thresh:
        return None, 0.0

    # Ok, we have found enough of hte peaks, so let's calculate the probability
    max_periods = set([])
    max_count = None
    for k, c in period_dist.counts.iteritems():
        if k >= period_interval[0] and k <= period_interval[1]:
            if max_count is None or c > max_count:
                max_count = c
                max_periods = set( [k] )
            elif c == max_count:
                max_periods.add( k )
    period_est = sorted(max_periods)[ len(max_periods) / 2 ]

    return ( period_est, period_dist.pmf( period_est ) )

##=======================================================================

##
# Given a sequence x and a period distribution, renormalizes the time
# so get a set of sequences for each period.  This will infer the
# best points in time for the start of each period.
# The return will be the start-of-period indices for the sequence
#
# @param x : A sequence
# @param period_dist : a cat_distibution with the period distribution
# @param noise_sigma : noise level in the sequence x values.
#                      If None this is estimated from the signal itself
def start_of_period_fit( x, period_dist, noise_sigma = None ):

    # Initialize the start-of-periods (sop) by finding first peak in
    # autocorrelation and using hte period distribution
    n = len(x)
    ac = autocorrelation.autocorrelation_estimate( x )
    height_thresh = 0.5 * max( ac[1:n/2] )
    init_start = 0
    for i,a in enumerate(ac):
        if a >= height_thresh:
            init_start = i
            break

    # build up the rest of the period starts from the initial and
    # the given period distribution
    sops = [ init_start ]
    mode = list(period_dist.mode())
    mode_period = mode[ len(mode)/2]
    if mode_period <= 0:
        raise RuntimeError( "Period Distribution has mode which include non-positive values! {0}".format( period_dist ) )
    while True:
        new_sop = sops[-1] + mode_period
        if new_sop >= n:
            break
        sops.append( new_sop )
    logger.info( "Initial SOPs: {0}".format( sops ) )

    # estimate hte noise sigma if wanted
    if noise_sigma is None:
        noise_sigma = estimate_periodic_signal_noise_sigma( x, period_dist )
    logger.info( "Noise Sigma: {0}".format( noise_sigma ) )

    # the score function for a particular set of sops
    # higher is better 
    def _score( sops ):

        # having a single sop is an edge case
        # treat it as if the last or first is a sop
        if len(sops) == 1:
            if sops[0] < n/2:
                sops = sops + [n]
            else:
                sops = [0] + sops

        # everything before first sop is discarded for now
        # ok, renormalize time based on sops
        y_data = []
        y_time = []
        steps = []
        for sop0, sop1 in zip( sops, sops[1:] ):
            y_slice = list( x[ sop0 : sop1 ] )
            y_data += y_slice
            y_time += list(np.linspace( 0.0, 1.0, len(y_slice) ))
            steps.append( len(y_slice ) )

        # ok, add in things before sops[0] and after sops[-1]
        # where we will do time step adjustment according to the
        # mean time steps in the slices above
        mean_step = int(np.mean( steps ))
        step_lookup = np.linspace( 0.0, 1.0, max( mean_step, sops[0]) )
        for i,y in enumerate( x[:sops[0]] ):
            time = 1.0 - step_lookup[ sops[0] - i - 1 ]
            y_data.insert( 0, y )
            y_time.insert( 0, time )
        step_lookup = np.linspace( 0.0, 1.0, max( mean_step, n - sops[-1]) )
        for i,y in enumerate( x[sops[-1] : ] ):
            time = step_lookup[ i ]
            y_data.append( y )
            y_time.append( time )

        # jitter time to make sure they are unique :-)
        y_time = map(lambda t: t + np.random.random() * 1.0e-5, y_time )

        # Ok, now that we have the renomalized time, treat data as
        # 2D data and fit a GP to it :-)
        nugget = map(lambda y: ( noise_sigma / y ) ** 2, y_data )
        gp = gaussian_process.GaussianProcess(nugget=nugget)
        gp.fit( np.array(y_time).reshape(-1,1), y_data )

        # Ok compute the likelihood of the fit
        # p( y | X, w ) = N( X_T * w , sigma^2 * I )
        # where X is training data and w is learned weights of GP
        # and sigma is the kernel sigma
        #return gp.reduced_likelihood_function_value_
        return gp.score( np.array(y_time).reshape(-1,1), y_data)

    # Ok, we will do a gradient descent algorithm to find hte best sops
    max_lik_sops, max_lik = _gradient_descent_sops(
        _score,
        sops,
        n,
        max_iters = 10 * len(sops),
        num_restarts = 2 * len(sops))
    
    return max_lik_sops, max_lik, _score
    
##=======================================================================

##
# Perform random 1-step changes to the SOPs with a given likelihood/score
# function (higher is better) to find hte maximum scored set of SOPs
#
# Returns the max SOPs and the max score found.
#
# Will restart with teh *same* initial SOP num_restarts times,
# and each restart will try at most max_iters single-step changes to the
# SOP.  Each generation/restart ends when we hit a local maxima
#
# @param lik : a function f( sops ) returning the likelihood or score for a
#              SOP. Higher is better
# @param init_sops : the initial SOP for all restarts.
# @param n : the maximum data size to cap SOPs at.
# @param max_iters : the maximum number of 1-step changes to try per restart
# @param num_restarts : the number of restarts to run
def _gradient_descent_sops( 
        lik,
        init_sops,
        n,
        max_iters = 100,
        num_restarts = 10 ):
    
    generation_sops = []
    generation_liks = []
    generation_stats = []

    # look over each restart
    for r in xrange(num_restarts):

        # start at initial sops always
        sops = init_sops
        max_lik = lik( sops )
        logger.info( "[{0}] SOPs GD: init lik = {1}".format( r, max_lik ) )

        # Ok, iterate to find max lik sop
        for i in xrange(max_iters):

            # pick a random sop to shift
            sop_to_explore_idx = np.random.choice(len(sops))
            logger.info( "[{0}] SOPs GD: explore index = {1}".format( r, sop_to_explore_idx ) )

            # ok, step to either side
            a_sops = copy.deepcopy(sops)
            b_sops = copy.deepcopy(sops)
            a_sops[ sop_to_explore_idx ] -= 1
            if a_sops[ sop_to_explore_idx ] < 0:
                a_sops[ sop_to_explore_idx ] = 0
            b_sops[ sop_to_explore_idx ] += 1
            if b_sops[ sop_to_explore_idx ] >= n:
                b_sops[ sop_to_explore_idx ] = n-1

            # Renormalize by removing redundant sops
            if sop_to_explore_idx > 0 and a_sops[ sop_to_explore_idx ] == a_sops[ sop_to_explore_idx -1 ]:
                del a_sops[ sop_to_explore_idx ]
            if sop_to_explore_idx < len(sops) - 1 and b_sops[ sop_to_explore_idx ] == b_sops[ sop_to_explore_idx + 1 ]:
                del b_sops[ sop_to_explore_idx ]

            # calculate new likelihoods
            a_lik = lik( a_sops )
            b_lik = lik( b_sops )

            # keep highest between current a and b likelihoods
            if a_lik >= max_lik and a_lik >= b_lik:
                max_lik = a_lik
                sops = a_sops
            elif b_lik >= max_lik and b_lik > a_lik:
                max_lik = b_lik
                sops = b_sops
            else:
                # we are done with this generation
                break

        # add the best found to generation
        generation_liks.append( max_lik )
        generation_sops.append( sops )
        generation_stats.append( { 'iter' : i } )
        logger.info( "[{0}] SOPs GD: generation max in {1} iterations = {2} {3}".format(
            r,
            i,
            max_lik,
            sops ) )
                

    # ok, no do a last ordered pass to tune the sops
    sops = generation_sops[0]
    max_lik = generation_liks[0]
    for s,l in zip( generation_sops, generation_liks ):
        if l >max_lik:
            max_lik = l
            sops = s
    gen_lik = max_lik
    for i in xrange(max_iters):
        for sop_idx in xrange(len(sops)):
            # ok, step to either side
            a_sops = copy.deepcopy(sops)
            b_sops = copy.deepcopy(sops)
            a_sops[ sop_idx ] -= 1
            if a_sops[ sop_idx ] < 0:
                a_sops[ sop_idx ] = 0
            b_sops[ sop_idx ] += 1
            if b_sops[ sop_idx ] >= n:
                b_sops[ sop_idx ] = n-1

            # Renormalize by removing redundant sops
            if sop_idx > 0 and a_sops[ sop_idx ] == a_sops[ sop_idx -1 ]:
                del a_sops[ sop_idx ]
            if sop_idx < len(sops) - 1 and b_sops[ sop_idx ] == b_sops[ sop_idx + 1 ]:
                del b_sops[ sop_idx ]

            # calculate new likelihoods
            a_lik = lik( a_sops )
            b_lik = lik( b_sops )

            # keep highest between current a and b likelihoods
            if a_lik >= max_lik and a_lik >= b_lik:
                max_lik = a_lik
                sops = a_sops
            elif b_lik >= max_lik and b_lik > a_lik:
                max_lik = b_lik
                sops = b_sops
        new_gen_lik = lik( sops )
        logger.info( "Generation {0} tuned likelihood = {1}".format(i,new_gen_lik))
        if gen_lik >= new_gen_lik:
            break
        else:
            gen_lik = new_gen_lik
    
    # ok, return best generation
    return sops, max_lik
            
##=======================================================================

##
# Estimate the noise sigma floor fo a periodic signal
# given the period distribution for the period
def estimate_periodic_signal_noise_sigma( x, period_dist ):

    n = len(x)
    
    # we will calcualte a sigma for each period
    sigmas = {}

    # iterate over known period in domain of distribution
    for period in period_dist.counts:

        # iterate over possible peior start points (start-of-period)
        sop_variances = []
        for sop in xrange(period):

            # calcualte noise sigma for this sop and period
            variances = []
            for i in xrange(period):
                data = []
                cur_index = i
                while cur_index < n:
                    data.append( x[cur_index] )
                    cur_index += period
                variances.append( np.var( data ) )

            # calculate the average variance
            mean_var = np.mean( variances )

            # store for this sop
            sop_variances.append( mean_var )

        # Ok, grab the *smallest* mean variance for any SOP as the
        # mean variance for that perior
        min_var = np.min( sop_variances )

        # store sigma for this period
        sigmas[ period ] = np.sqrt( min_var )

    # Ok, compute the expected sigma by using hte period distribution
    e_sigma = 0.0
    for period,s in sigmas.iteritems():
        p = period_dist.pmf( period )
        e_sigma += ( p * s )

    # return the expeted sigma
    return e_sigma

##=======================================================================

def plot_sops( x, sops ):

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot( x, 'b-' )
    plt.plot( sops, np.array(x)[ sops ], 'rx', ms=10 )
    

##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
