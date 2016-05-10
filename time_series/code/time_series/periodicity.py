import logging
logger = logging.getLogger( __name__ )

import math

import numpy as np

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
##=======================================================================
##=======================================================================
##=======================================================================
