import logging
logger = logging.getLogger()

import itertools
import numpy as np
import scipy.stats

##======================================================================

##
# A domain which has a size and can be split, also can be checked
# for insideness
class Domain(object):
    def is_inside(self,x):
        raise NotImplementedError()
    def domain_size(self):
        raise NotImplementedError()
    def subdivide_into_compact_nonoverlapping(self, T):
        raise NotImplementedError()

##======================================================================

##
# A bounded multidimensional domain
class BoundedDomain(object):
    def __init__( self, lower_upper_bounds ):
        self.lower_upper_bounds = lower_upper_bounds
    def domain_size(self):
        size = None
        for (l,u) in self.lower_upper_bounds:
            s = u - l
            if size is None:
                size = s
            else:
                size *= s
        return size
    def is_inside(self, x):
        for (l,u), y in zip(self.lower_upper_bounds, x):
            if y < l or y >= u:
                return False
        return True
    def subdivide_into_compact_nonoverlapping(self,T):
        V = len(self.lower_upper_bounds)
        step = float(1.0)/T
        subs = []
        ticks = [ float(i)/T for i in xrange(T) ]
        ticks_d = [ticks] * V
        for splits in itertools.product( * ticks_d ):
            uls = []
            for s, (l,u) in zip(splits,self.lower_upper_bounds):
                lower = l + ( u - l ) * s
                upper = l + ( u - l ) * ( s + step )
                uls.append( ( lower, upper ) )
            subs.append( BoundedDomain( uls ) )
        return subs
    

##======================================================================
##======================================================================

##
# Compte an approximate bayes error rate from finite samples.
#
# This returns both an estimate as well as the upper and lower bounds.
# These are true finite-sample bounds :-) yay.
#
# Implements the paper:
# Two-Sided Exponential Concentration Bounds for Bayes Error Rate
# and Shannon Entropy.
# By Jean Honorio and Tommi Jaakkola.
# Appearing in ICML 2016

def approximate_bayes_error_rate(
        samples_a,
        samples_b,
        domain,
        lipschitz_k,
        delta ):

    # some checks
    N = len(samples_a)
    if len(samples_b) != N:
        raise RuntimeError( "Need to be given the same number of samples from a and b classes!" )

    # ok, divive domain in non-overlapping subdomains
    T = int(np.ceil(np.power( N, 0.25 )))
    subdomains = domain.subdivide_into_compact_nonoverlapping( T )

    # ok, compute emprirical counts in each subdomain
    empirical_a = {}
    empirical_b = {}
    for a in samples_a:
        for dom in subdomains:
            if dom.is_inside( a ):
                if dom not in empirical_a:
                    empirical_a[ dom ] = 0.0
                empirical_a[ dom ] += (1.0 / N)
                break
    for b in samples_b:
        for dom in subdomains:
            if dom.is_inside( b ):
                if dom not in empirical_b:
                    empirical_b[ dom ] = 0.0
                empirical_b[ dom ] += (1.0 / N)
                break

    # compute empirical bayes error rate
    empirical_ber = 0.0
    for dom in subdomains:
        pa = empirical_a.get( dom, 0.0 )
        pb = empirical_b.get( dom, 0.0 )
        empirical_ber += min( pa, pb )

    # compute bounds
    e_ndelta = 1.0 / T * np.sqrt( 1.0/8.0 * np.log(N) + 1.0/2.0 * np.log(4.0/delta))
    e_nkd = ( lipschitz_k * np.power(domain.domain_size(),2) ) / ( 2.0 * T )
    lower_ber = empirical_ber - e_ndelta - e_nkd
    upper_ber = empirical_ber + e_ndelta

    # return results
    return empirical_ber, ( lower_ber, upper_ber )
    
##======================================================================
##======================================================================
##======================================================================
##======================================================================

##
# A test scenario (base class)
class Scenario( object ):
    
    def lipschitz_k(self):
        raise NotImplementedError()
    def domain(self):
        raise NotImplementedError()

    def sample_from_a( self, n ):
        raise NotImplementedError()
    def sample_from_b( self, n ):
        raise NotImplementedError()

    def a_b_samples( self, N ):
        a = self.sample_from_a( N )
        b = self.sample_from_b( N )
        return (a, b)
    
        
##======================================================================

##
# A very simple test scenario where we have a 0.0 bayes error rate
# using a 1-dimensional bounded value
class Bounded1dPerfectScenario( Scenario ):

    def __init__(self, lower, upper,
                 frac = 0.5,
                 lipschitz_k = 0.1):
        self.lower = lower
        self.upper = upper
        self.threshold = lower + ( upper - lower ) * frac
        self._domain = BoundedDomain( [ (lower,upper) ] )
        self._lipschitz_k = lipschitz_k

        # build up the wanted gaussian
        self.sigma = np.sqrt( 1.0 / ( lipschitz_k * np.sqrt( 2 * np.pi * np.e ) ) )
        self.rv = scipy.stats.norm( loc=0.0,
                                    scale = self.sigma )

    def lipschitz_k( self ):
        return self._lipschitz_k

    def domain(self):
        return self._domain
        
    def sample_from_a( self, n ):
        samples = []
        a_mean = self.lower + ( self.threshold - self.lower) / 2.0
        for i in xrange(n):

            # pick from gaussian centered on middle of segment until
            # we get one inside the segment
            x = a_mean + self.rv.rvs()
            while x < self.lower or x >= self.threshold:
                x = a_mean + self.rv.rvs()

            samples.append( [x] )
        return samples

    def sample_from_b( self, n ):
        samples = []
        b_mean = self.threshold + ( self.upper - self.threshold) / 2.0
        for i in xrange(n):

            # pick from gaussian centered on middle of segment until
            # we get one inside the segment
            x = b_mean + self.rv.rvs()
            while x < self.threshold or x >= self.upper:
                x = b_mean + self.rv.rvs()

            samples.append( [x] )
        return samples
    

##======================================================================

##
# A test scenario with two guassians 1dimensional
class Bounded1dTwoGuassiansScenerio( Scenario ):

    def __init__( self,
                  lower,
                  upper,
                  mean_a,
                  sigma_a,
                  mean_b,
                  sigma_b ):
        self.lower = lower
        self.upper = upper
        self.mean_a = mean_a
        self.sigma_a = sigma_a
        self.mean_b = mean_b
        self.sigma_b = sigma_b
        self._lipschitz_a = 1.0 / ( sigma_a**2 * np.sqrt( 2 * np.pi * np.e ) )
        self._lipschitz_b = 1.0 / ( sigma_b**2 * np.sqrt( 2 * np.pi * np.e ) )
        self._lipschitz_k = max( self._lipschitz_a,
                                 self._lipschitz_b )
        self._domain = BoundedDomain( [ (lower,upper) ] )
        self.rv_a = scipy.stats.norm( mean_a, sigma_a )
        self.rv_b = scipy.stats.norm( mean_b, sigma_b )
        

    def domain(self):
        return self._domain

    def lipschitz_k(self):
        return self._lipschitz_k

    def sample_from_a(self,n):
        samples = []
        for i in xrange( n ):

            x = self.rv_a.rvs()
            while x < self.lower or x >= self.upper:
                x = self.rv_a.rvs()

            samples.append( [x] )
        return samples

    def sample_from_b(self,n):
        samples = []
        for i in xrange( n ):

            x = self.rv_b.rvs()
            while x < self.lower or x >= self.upper:
                x = self.rv_b.rvs()

            samples.append( [x] )
        return samples
    


##======================================================================

##
# A test scenario with two multidimensional normal distributions
class BoundedTwoGuassiansScenerio( Scenario ):

    def __init__( self,
                  bounds,
                  mean_a,
                  cov_a,
                  mean_b,
                  cov_b,
                  lipschitz_k):
        self.bounds = bounds
        self.lower_bounds = np.array(map(lambda (l,u): l, bounds))
        self.upper_bounds = np.array(map(lambda (l,u): u, bounds))
        self.mean_a = mean_a
        self.cov_a = cov_a
        self.mean_b = mean_b
        self.cov_b = cov_b
        self._lipschitz_k = lipschitz_k
        self._domain = BoundedDomain( bounds )
        self.rv_a = scipy.stats.multivariate_normal( mean_a, cov_a )
        self.rv_b = scipy.stats.multivariate_normal( mean_b, cov_b )
        

    def domain(self):
        return self._domain

    def lipschitz_k(self):
        return self._lipschitz_k

    def sample_from_a(self,n):
        samples = []
        for i in xrange( n ):

            x = self.rv_a.rvs()
            while np.any(x < self.lower_bounds) or np.any(x >= self.upper_bounds):
                x = self.rv_a.rvs()

            samples.append( x )
        return samples

    def sample_from_b(self,n):
        samples = []
        for i in xrange( n ):

            x = self.rv_b.rvs()
            while np.any(x < self.lower_bounds) or np.any(x >= self.upper_bounds):
                x = self.rv_b.rvs()

            samples.append( [x] )
        return samples
    


##======================================================================
##======================================================================
##======================================================================

##
# Test a scenario
def test_scenario( S, N, delta ):

    # grab samples
    samples_a, samples_b = S.a_b_samples( N )

    # run estimate
    ber, ( lower_ber, upper_ber ) = approximate_bayes_error_rate(
        samples_a,
        samples_b,
        S.domain(),
        S.lipschitz_k(),
        delta )

    return ( ber, ( lower_ber, upper_ber ) )

##======================================================================
##======================================================================

##
# compute domain from samples
def compute_domain_from_samples( samples ):
    sa = np.array( samples )
    mins = np.min( sa, axis=0 )
    maxs = np.max( sa, axis=0 )
    return BoundedDomain( zip( mins, maxs ) )


##======================================================================

##
# rough estimate of lipschitz_k which just takes the maximum
# seen derivative :-)
def rough_lipschitz_k( samples, bandwidth=None, num_samples_per_dimension = 10, EPS=1.0e-5 ):

    # estimate a distribution from samples using
    # kernel density estimation (guassian kernel)
    sa = np.array(samples)
    try:
        kde = scipy.stats.gaussian_kde( sa.transpose() )
    except np.linalg.LinAlgError as le:
        sa += np.random.random( size=sa.shape ) * EPS
        kde = scipy.stats.gaussian_kde( sa.transpose() )

    # densly sample the domain to approximate derivatives
    domain = compute_domain_from_samples( samples )
    subds = domain.subdivide_into_compact_nonoverlapping( num_samples_per_dimension )
    derivative_x = np.array(map(lambda d: np.mean( d.lower_upper_bounds, axis=1 ), subds ))
    N = len(derivative_x)

    # ok, now compute largest derivative
    p_x = kde.pdf( derivative_x.transpose() )
    derivs = []
    for i in xrange( N ):
        for j in xrange( N ):
            p_diff = abs(p_x[j] - p_x[i])
            x_diff = np.linalg.norm( derivative_x[i] - derivative_x[j] )
            if x_diff != 0:
                d = p_diff / x_diff
                derivs.append( d )

    # find max derivative
    return max( derivs )


##======================================================================
##======================================================================

def approximate_bayes_error_rate_using_only_samples(
        samples_a,
        samples_b,
        delta ):

    # compute domain and rough lipschitz_k
    samples_a = np.array( samples_a )
    samples_b = np.array( samples_b )
    domain = compute_domain_from_samples( np.vstack( [ samples_a, samples_b ] ) )
    lipschitz_a = rough_lipschitz_k( samples_a )
    lipschitz_b = rough_lipschitz_k( samples_b )
    lipschitz_k = max( lipschitz_a, lipschitz_b )

    ber = approximate_bayes_error_rate(
        samples_a,
        samples_b,
        domain,
        lipschitz_k,
        delta )

    return ber, lipschitz_k, domain 

##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
##======================================================================
