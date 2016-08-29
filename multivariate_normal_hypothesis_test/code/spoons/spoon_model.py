import logging
logger = logging.getLogger( __name__ )


import numpy as np
import scipy
import scipy.stats
import itertools
import operator


##===================================================================

##
# A spoon model where we treat the potential observation of a spoon
# accross *any*/*all* configurations as a multivariate normal
#
class ConfigurationAgnosticSpoon( object ):

    ##
    # Initialie model with mean vector and covariance matrix
    #
    # Number of cells is implicit in the size of the mean/cov
    #
    # We also have an i.i.d noise term for each of the cells for
    # observation functions :-)
    def __init__( self, mu, cov, iid_noise_sigma = 0.1 ):
        self.mu = mu
        self.cov = cov
        self.noise_sigma = iid_noise_sigma
        try:
            self._rv = scipy.stats.multivariate_normal(
                mean = self.mu,
                cov = self.cov )
        except np.linalg.LinAlgError as le:
            # probably have a singular matrix in the covariance, add
            # some identity noise :-)
            self.cov += ( np.eye( len(self.cov) ) * np.random.random( size=len(self.cov) ) * iid_noise_sigma )
            self._rv = scipy.stats.multivariate_normal(
                mean = self.mu,
                cov = self.cov )            
        self._noise_rv = scipy.stats.norm(
            loc=0.0,
            scale=self.noise_sigma )


    ##
    # Return the number of cells for hte model
    def num_cells(self):
        return len(self.mu)

    ##
    # Sample a set of observations from a *single* sample configuration
    # of the spoon
    #
    # returns ( observations, configuration )
    # if n is None, returns ( single observation, configuration )
    def sample_observations_from_single_configuration( self, n=None ):

        num = n
        if n is None:
            num = 1
        
        # sample the configuration
        configuration = self._rv.rvs()

        # ok, sample observations from this configuration using
        # iid observation noise
        samples = []
        for i in xrange(num):
            s = configuration + self._noise_rv.rvs()
            samples.append( s )

        # return samples
        if n is None:
            return samples[0], configuration
        return np.array( samples ), configuration

##===================================================================

##
# A model of a spoon which has an explicit finite number of
# configurations, each of which is defined by a multivariate
# normal distribution.
#
class FiniteConfigurationSpoon( object ):

    ##
    # Create a new spoon model with a set of
    # multivariate normals.
    # Also include iid additive observation noise.
    #
    # The given mu_sigmas parameters is a List[ ( mu, cov) ]
    # with mean vector and covariance matrix for the multivariate
    # normal defining a particular configuration
    #
    # Optionally, each configuration can have a weight to be treated as
    # the probability of the spoon being in that configuration
    def __init__( self, mu_covs, iid_noise_sigma, configuration_weight = None ):
        n = len(mu_covs)
        self.mu_covs = mu_covs
        self.noise_sigma = iid_noise_sigma
        self._noise_rv = scipy.stats.norm(
            scale=self.noise_sigma)
        self.rvs = [
            scipy.stats.multivariate_normal(
                mean=mu,
                cov=cov)
            for (mu,cov) in self.mu_covs ]
        if configuration_weight is None:
            self.configuration_weight = np.array( [1.0 / n] * n )
        else:
            self.configuration_weight = np.array( configuration_weight )
            self.configuration_weight /= np.sum(self.configuration_weight)
        self._configuration_rv = scipy.stats.rv_discrete(
            values = ( range(n),
                       self.configuration_weight ) )
        self.configurations_data = None

    ##
    # The number of cells
    def num_cells(self):
        return len(self.mu_covs[0][0])


    ##
    # Sample a set of observations from a *single* sample configuration
    # of the spoon
    #
    # returns ( observations, configuration )
    # if n is None, returns ( single observation, configuration )
    def sample_observations_from_single_configuration( self, n=None ):

        # sample the configuration
        cid = self._configuration_rv.rvs()

        # generate samples from configuration
        samples, configuration = self._sample_observations_from_known_configuration( cid, n=n )
        
        return samples, configuration

    ##
    # Sample a set of obsrvations from given known configuration (as id).
    # This is an internal method, gnerally not used externally since
    # we generaly want to sample the configuration as well.
    #
    # Returns ( samples, configuration )
    def _sample_observations_from_known_configuration( self, cid, n=None ):
        num = n
        if n is None:
            num = 1

        # grab configuration and sample from it
        configuration_rv = self.rvs[ cid ]
        configuration = configuration_rv.rvs()

        # ok, sample observations from this configuration using
        # iid observation noise
        samples = []
        for i in xrange(num):
            s = configuration + self._noise_rv.rvs()
            samples.append( s )

        # return samples
        if n is None:
            return samples[0], configuration
        return np.array( samples ), configuration

    
##===================================================================
##===================================================================

##
# Build up the configuration agnostic spoon model from data
def fit_configuration_agnostic_spoon( data, noise_sigma=0.1 ):

    # we use the maximum likelihood estimator for the
    # single multivariate normal
    mu = np.mean( data, axis = 0 )
    cov = np.cov( data, rowvar=0 )

    # ensure that at least hte covariance diagonals at
    # noise_sigma or more
    for i in xrange(cov.shape[0]):
        cov[i,i] = max( cov[i,i], noise_sigma )

    return ConfigurationAgnosticSpoon(
        mu,
        cov,
        iid_noise_sigma = noise_sigma )


##===================================================================

##
# Fit a finite configuration spoon model from the data.
# Here, we do an online single-pass algorithm to detect
# previously unseen configuration as well as merge any
# configurations that are too similar.
#
# We have three thresholds:
#
#  The first is to detect when a new data point seems different
#  enough from any previsouly seen configuration. This is
#  the new_configuration_thres and represent the largest
#  probability below-which we decide a point is a new configuration.
#  (so having this at 0.05 means that it must be less than 5% likely
#   a point comes from any previous configuration before we try to
#   make a new configuration).
#
#  The second threshold has to do with how we merge configurations.
#  After adding data to configurations it may be that two configurations
#  start to look very similar.  We perform a hypothesis test to decide
#  where the data points assigned to two different configurations
#  actuall compe from multivariate normal distributions with
#  the *same* mean.  The NULL hypothesis is that they do, and if
#  we are unable to reject this NULL hypothesis, we will merge
#  the two configurations into a new single configuration with the
#  data points from the previous two assigned to it.
#  The p-value (probability) threhold used to reject the NULL hypothesis,
#  and hence to stop a merging of configurations, is the
#  equal_pvalue_thresh parameter.  Since we test equality over all
#  traked configurations, we apply a bonferoni correction to
#  the given p-value threhold in order to ensure a strong
#  guarantee.
#
#  The third threshold has to do with the minimum number of elements a
#  configuration must have, in terms of assigned data samples, before
#  we try to check if it needs to be merged with other configurations.
#  This ensures that our hypothesis test has at leasst some varied samples
#  for each configuration.  The parametr for this is min_samples_to_merge.
#  If set to 1, we will try to merge given only a single sample.
#  This will cause a lot of merging since it is very hard to reject
#  the NULL hypothesis that two means are the same when we only have a
#  single datapoint for one of them.
#  As a testing rule-f-thumb try setting this number to 1.5 * num_cells
#  for the model since the higher the dimensionality of the multivariate
#  normal, the more samples you need to pool to successfully perform a
#  hypothesis test :-)
#
def fit_finite_configuration_spoon( data,
                                    new_configuration_thres = 0.05,
                                    equal_pvalue_thresh = 0.05,
                                    min_samples_to_merge = 11,
                                    initial_configuration_samples = 10,
                                    noise_sigma = 0.1,
                                    verbose=True):

    # the configurations we are trackin
    configurations = []
    crvs = []
    configurations_data = []

    # how we fit a single configuration from data
    def make_configuration( inner_data ):
        spoon = fit_configuration_agnostic_spoon( inner_data,
                                                  noise_sigma = noise_sigma )
        return ( (spoon.mu,
                  spoon.cov),
                 spoon._rv )

    # ok, start the first configuration from the initial samples
    if len(data) < initial_configuration_samples:

        # we assume a single configuration here
        initial_configuration_samples = len(data)

    # fit initial configuration
    init_data = data[ : initial_configuration_samples ]
    init_configuration, init_rv = make_configuration( init_data )
    configurations.append( init_configuration )
    crvs.append( init_rv )
    configurations_data.append( init_data )
    if verbose:
        logger.info( "Fit initial configuration using #{0} data: {1}".format(
            initial_configuration_samples,
            init_configuration ) )

    # ok, process the rest of the data
    for x in data[ initial_configuration_samples: ]:

        # first, choose the configuration which has the max likelihood
        # for this data point
        liks = [ rv.pdf( x ) for rv in crvs ]
        ml_cid = np.argmax( liks )

        # compute the extrema probability
        # (the probability to seeing a data point further than the given)
        extrema_prob = _multivariate_normal_extrema_probability(
            configurations[ ml_cid ],
            x )

        # the set of configurations to test against
        active_set = set(xrange(len(configurations)))

        if verbose:
            logger.info( "tracking #{0}: max lik={1} for conf {2}, prob={3}".format(
                len(configurations),
                np.max(liks),
                ml_cid,
                extrema_prob ) )

        # if the extrema probability is too low, make a new configuration
        updated_configuration = None
        updated_rv = None
        updated_data = None
        updated_cid = None
        if extrema_prob <= new_configuration_thres:
            ncells = len(x)
            updated_configuration = ( x,
                                      np.eye( ncells ) * noise_sigma * 3 )
            updated_rv = scipy.stats.multivariate_normal(
                mean = updated_configuration[0],
                cov = updated_configuration[1] )
            updated_data = np.array( [ x ] )
            updated_cid = len(configurations)

            if verbose:
                logger.info( "  trying new configuration: {0}".format(
                    updated_configuration ))

        else:
            
            # ok, make new configuration assuming we added this point
            # to the ml configuration
            updated_data = np.vstack( [ configurations_data[ml_cid], x ] )
            updated_configuration, updated_rv = make_configuration( updated_data )
            active_set -= { ml_cid }
            updated_cid = ml_cid

            if verbose:
                logger.info( "  adding to ml configuration")

        if verbose:
            logger.info( "active set: {0}".format( active_set ) )

        # see if we have to join any of our configurations because they
        # are actually the same
        #
        # The NULL hypothesis is they *are* the same, but we can reject this
        # null hypothesis and hence not merge them :-)
        pvalues = []
        if len(updated_data) >= min_samples_to_merge:
            for cid in active_set:
                pvalue = hypothesis_test_equal_multivariate_normal(
                    configurations_data[ cid ],
                    updated_data,
                    noise_sigma,
                    verbose=verbose)
                pvalues.append( ( cid, pvalue ) )
            if verbose:
                logger.info( "pvalues: {0}".format( pvalues ) )

        # calculate the bonferoni correction for multiple hypothesis testing
        # since we are actually testing against all previously tracked
        # configurations :-)
        equal_pvalue_thresh_bonferoni = equal_pvalue_thresh / len(configurations)

        # find min pvalue and see if we can reject the null hypothesis and hence
        # do not have to merge with anything, otherwise we need to merge
        # with the highest p-value (since that is the multivariate
        # normal closest to ours)
        min_cid = None
        min_pvalue = -np.inf
        if len(pvalues) > 0:
            min_cid, min_pvalue = sorted( pvalues, key=lambda (c,p): p )[0]
        if min_cid is not None and min_pvalue > equal_pvalue_thresh_bonferoni:

            # we are merging, find maximum p-value
            max_cid, max_pvalue = sorted( pvalues, key=lambda (c,p): p )[-1]
            

            # merge
            new_data = np.vstack( [ configurations_data[ max_cid ],
                                    updated_data ] )
            new_configuration, new_rv = make_configuration( new_data )
            configurations[ max_cid ] = new_configuration
            configurations_data[ max_cid ] = new_data
            crvs[ max_cid ] = new_rv

            # remove previous if any
            if updated_cid < len(configurations):
                del configurations[ updated_cid]
                del configurations_data[ updated_cid]
                del crvs[ updated_cid]

            if verbose:
                logger.info( "for datapoint: Merging {0} with {1}: pvalue {2} > {3}".format(
                    updated_cid,
                    max_cid,
                    min_pvalue,
                    equal_pvalue_thresh_bonferoni ) )


        elif updated_cid >= len(configurations):

            # add new configuration
            configurations.append( updated_configuration )
            configurations_data.append( updated_data )
            crvs.append( updated_rv )

            if verbose:
                logger.info( "for datapoint: tracking new configuration" )

        else:

            configurations[ updated_cid ] = updated_configuration
            configurations_data[ updated_cid ] = updated_data
            crvs[ updated_cid ] = updated_rv

            if verbose:
                logger.info( "for datapoint: updated ml configuration" )

    # return the configurations
    fcs = FiniteConfigurationSpoon(
        configurations,
        noise_sigma,
        configuration_weight = map(float,map(len,configurations_data)))
    fcs.configurations_data = configurations_data
    return fcs

##===================================================================
##===================================================================

##
# Return the survival probability that the two samples came from
# a multivariate guassian with the same mean and same covariance.
#
# NULL hypothesis: data0 and data1 came from multivariate normal
#                  distributions with the same mean and
#                  assumed equal covariance
#
# Equal covariance is *assumed* and same mean is tested!
#
# The return value is a p-value to reject the HULL hypothesis
def hypothesis_test_equal_multivariate_normal(
        data0,
        data1,
        noise_sigma,
        EPS = 1.0e-5,
        assume_independence = True,
        verbose = True):

    # we will use Hotelling's T-Square Test for
    # equal multivariate normal means.
    # we assume:
    #  1) Equal Convariances
    #  2) Independent Samples
    #  3) Underlying multivariate normal distribution

    # number of data pints
    n0 = len(data0)
    n1 = len(data1)
    num_vars = 1
    if hasattr( data0[0], '__len__' ):
        num_vars = len( data0[0] )

    # compute sample means
    mean0 = np.mean( data0, axis=0 )
    mean1 = np.mean( data1, axis=0 )

    # compute sample covariances
    cov0 = np.cov( data0, rowvar=0 )
    cov1 = np.cov( data1, rowvar=0 )

    # since we *assume* equal covariances, we use all the
    # data to compute an estimated covariance :-)
    if n0 == 1 and n1 == 1:
        S = np.cov( np.vstack( [ data0, data1] ), rowvar=0 )
    else:
        S = ( ( n0 - 1) * cov0 + ( n1 - 1 ) * cov1 ) / ( n0 + n1 - 2 )

    # ensure at least noise_sigma covariance for diagonal elements
    for i in xrange(S.shape[0]):
        S[i,i] = max( S[i,i], noise_sigma )

    # kill covariance off-diagonal terms if we are assuming independence
    if assume_independence:
        for i in xrange(S.shape[0]):
            for j in xrange(S.shape[1]):
                if i != j:
                    S[i,j] = 0.0
    
    # the T statistic
    delta = (mean0 - mean1)
    try:
        Sinv = np.linalg.inv( S )
    except np.linalg.LinAlgError as le:
        # probabily S was wingular, add some noise
        S += ( np.eye( S.shape[0] ) * np.random.random(size=S.shape[0]) * EPS )
        Sinv = np.linalg.inv( S )
    T2 = np.dot( np.dot( delta, Sinv ),
                 delta )
    T2 *= float( n0 * n1 ) / ( n0 + n1 )
    T2 = np.abs( T2 )

    # form into F statistic
    if n0 + n1 - 2.0 > 0:
        F = float( n0 + n1 - num_vars - 1.0 ) / ( num_vars * ( n0 + n1 - 2.0 ) ) * T2
    else:
        F = 0.0

    # ok, return the CDF of the F distribution defined from above
    # for the given value
    if n0 + n1 - num_vars - 1 > 0:
        survival = scipy.stats.f( num_vars, n0 + n1 - num_vars - 1 ).sf( F )
    else:
        survival = 0.0

    # make sure survival is finite
    if not np.isfinite( survival ):
        survival = 0.0
    if verbose:
        logger.info( "  delta: {0}, T2:{1}, F:{2}, surv:{3} n={4},{5}".format(
            delta,
            T2,
            F,
            survival,
            n0,
            n1) )
    
    # return the survival rate for the given value (so how extreme the value is)
    # this is a p-value, so a return of < 0.05 means that we could reject
    # the NULL hypothesis that data came from same means since the chance
    # is less than 5% of seeing the given data from same means.
    return survival

##===================================================================

##
# Return the probability of a given multivariate normal distribution
# generating values *more* extreme than hte given x.
# By extreme we take it to mean values further from the mean in
# eucledean space.
#
# We use the chi^2 distribution as the confidence interval of
# the distance from the mean.
#
# We could also perform monte-carlo integration to compute the probability
# of generating values further than x, but this is slower :-)
def _multivariate_normal_extrema_probability(
        mean_cov,
        x,
        num_samples_per_dimension = 100,
        EPS = 1.0e-5):

    ##
    # use the chi^2 distribution :=)
    mu = np.array(mean_cov[0])
    cov = mean_cov[1]
    k = len(mu)
    try:
        cov_inv = np.linalg.inv( cov )
    except np.linalg.LinAlgError as le:
        cov_inv = np.linalg.inv( cov + np.eye( k ) * np.random.random(EPS) )
    delta = np.array(x) - mu
    d = np.dot( delta,
                np.dot( cov_inv,
                        delta ) )
    survival = scipy.stats.chi2( k ).sf( d )
    return survival


    # num_samples = num_samples_per_dimension * len(x)
    # mu = np.array(mean_cov[0])
    # cov = np.array(mean_cov[1])
    # rv = scipy.stats.multivariate_normal( mean=mu,
    #                                       cov=cov )

    # # ok, do monte-carlo integration by taking samples and just
    # # counting how many lie further from mean that x
    # x_dist = np.linalg.norm( mu - x )
    # samples = rv.rvs( size=num_samples )
    # num_further = 0
    # for s in samples:
    #     s_dist = np.linalg.norm( s - mu )
    #     if s_dist > x_dist:
    #         num_further += 1

    # # ok, return probability of being further than x
    # p = float(num_further) / num_samples
    # return p

##===================================================================
##===================================================================
##===================================================================
##===================================================================

##
# Test a fitting procedure.
#
# We are given a *true* model.
# We generate data from this true model then perform the given
# fitting procedure.  We then test the resulting model using
# a non-parametric distribution test.
def test_fit_algorithm(
        true_model,
        fit_algorithm,
        num_configurations_samples = 20,
        num_samples_per_configuration = 10,
        num_configurations_samples_for_distribution_test = 10,
        num_samples_for_distribution_test = 30 ):

    # ok, first generate data from the true model
    data = []
    for ci in xrange(num_configurations_samples):
        cdata = true_model.sample_observations_from_single_configuration(
            n=num_samples_per_configuration)
        data.extend( list(cdata[0]) )
    data = np.array( data )
    
    # Ok, apply hte fit algorithm
    model = fit_algorithm( data )

    # generate samples to test fit versus true models
    true_data = []
    model_data = []
    for ci in xrange(num_configurations_samples_for_distribution_test):
        cdata = true_model.sample_observations_from_single_configuration(
            n=num_samples_for_distribution_test )
        true_data.extend( list(cdata[0]) )
        cdata = model.sample_observations_from_single_configuration(
            n=num_samples_for_distribution_test)
        model_data.extend( list(cdata[0]) )
    true_data = np.array(true_data)
    model_data = np.array(model_data)

    # test the samples using Kolmogorov-Smirnov test
    #logger.info( "starting KS test" )
    stat = None
    #stat = _multidimenaional_ks_test( true_data, model_data )

    return model, stat
        
##===================================================================
##===================================================================
##===================================================================

##
# The multi-dimensional Kolmogorov-Snirnov test.
#
# We take the maximum difference in the empirical CDFs
# taking into account *all* possible CDF directions :-)
# so this takes longer the higher dimensional our samples
# are.
def _multidimenaional_ks_test( a_samples, b_samples ):
    dim = len( a_samples[0] )
    max_diff = 0.0
    for x in a_samples + b_samples:
        for ops in itertools.product( [ operator.le, operator.ge ],
                                      repeat = dim ):
            a_cdf = _ks_cdf( a_samples, x, ops )
            b_cdf = _ks_cdf( b_samples, x, ops )
            diff = abs( a_cdf - b_cdf )
            if diff > max_diff:
                max_diff = diff
    return max_diff

##
# Computes the empirical CDF from samples for a particular x
# We need to give the orintation of the CDF by giving the
# comparison operators wanted for each dimension of the samples
def _ks_cdf( samples, x, ops ):
    num_found = 0
    n = len(samples)
    for s in samples:
        if _ks_all_ops( s, x, ops ):
            num_found += 1
    return float( num_found ) / float( n )

##
# Apply the given comparator per dimension to the given values,
# returns if all resulting in true
def _ks_all_ops( a, b, ops ):
    return np.all(map(lambda (a0,b0,ops0): ops0(a0,b0),
                      zip( a, b, ops ) ) )

##===================================================================
##===================================================================
##===================================================================
##===================================================================

##
# A simple 4-cell independent 2-configuration spoon model
NOISE_SIGMA = 0.1
SPOON_4CELL_2CONFIGURATION_INDEPENDENT = FiniteConfigurationSpoon(
    [ ( np.array( [1.0, 1.0, 1.0, 1.0]),
        np.eye(4) * 0.01 ),
      ( np.array( [0.0, 1.0, 0.0, 1.0]),
        np.eye(4) * 0.01 ),
    ],
    NOISE_SIGMA )


##
# A simple 10-cell independent 2-configuration spoon model
SPOON_10CELL_2CONFIGURATION_INDEPENDENT = FiniteConfigurationSpoon(
    [ ( np.array( [1.0] * 10 ),
        np.eye(10) * 0.01 ),
      ( np.array( [0.0, 1.0] * 5),
        np.eye(10) * 0.01 ),
    ],
    NOISE_SIGMA )


##
# A 10-cell independent 4-configuration spoon model
SPOON_10CELL_4CONFIGURATION_INDEPENDENT = FiniteConfigurationSpoon(
    [ ( np.array( [1.0] * 10 ),
        np.eye(10) * 0.01 ),
      ( np.array( [0.0, 1.0] * 5),
        np.eye(10) * 0.01 ),
      ( np.array( [0.0] * 5 + [1.0] * 5 ),
        np.eye(10) * 0.01 ),
      ( np.array( [1.0] * 5 + [0.0] * 5 ),
        np.eye(10) * 0.01 ),    
    ],
    NOISE_SIGMA )


##
# Some data for this model
SAMPLE_DATA_4CELL_2CONFIGURATION = []
SAMPLE_DATA_4CELL_2CONFIGURATION.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 0, n=10 )[0]) )
SAMPLE_DATA_4CELL_2CONFIGURATION.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 1, n=10 )[0]) )

##
# some larger sample data
SAMPLE_DATA_4CELL_2CONFIGURATION_MEDIUM = []
SAMPLE_DATA_4CELL_2CONFIGURATION_MEDIUM.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 0, n=15 )[0]) )
SAMPLE_DATA_4CELL_2CONFIGURATION_MEDIUM.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 1, n=15 )[0]) )


##
# some intrleaved data
SAMPLE_DATA_4CELL_2CONFIGURATION_INTERLEAVE = []
SAMPLE_DATA_4CELL_2CONFIGURATION_INTERLEAVE.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 0, n=10 )[0]) )
SAMPLE_DATA_4CELL_2CONFIGURATION_INTERLEAVE.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 1, n=10 )[0]) )
SAMPLE_DATA_4CELL_2CONFIGURATION_INTERLEAVE.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 0, n=10 )[0]) )
SAMPLE_DATA_4CELL_2CONFIGURATION_INTERLEAVE.extend(
    list(SPOON_4CELL_2CONFIGURATION_INDEPENDENT._sample_observations_from_known_configuration( 1, n=10 )[0]) )


    
##===================================================================
##===================================================================
##===================================================================
##===================================================================

def test_increasing_cell_performance(
        min_cells = 10,
        max_cells = 10000,
        cell_step_factor = 10.0,
        cell_min_sample_factor = 1.5,
        noise_sigma = 0.1,
        configuration_sigma = 0.01 ):

    results = []

    # iterate over num cells
    num_cells = min_cells
    while num_cells <= max_cells:

        # generate *true* 4-configuration model
        true_model_configurations = []
        true_model_configurations.append(
            ( np.array( [1.0] * num_cells ),
              np.eye( num_cells ) * configuration_sigma ) )
        interleave = [0.0, 1.0] * int((num_cells+1)/2)
        interleave = interleave[:num_cells]
        true_model_configurations.append(
            ( np.array( interleave ),
              np.eye( num_cells ) * configuration_sigma ) )
        na = num_cells / 2
        nb = num_cells - na
        true_model_configurations.append(
            ( np.array( [0.0] * na + [1.0] * nb ),
              np.eye( num_cells ) * configuration_sigma ) )
        true_model_configurations.append(
            ( np.array( [1.0] * na + [0.0] * nb ),
              np.eye( num_cells ) * configuration_sigma ) )
        true_model = FiniteConfigurationSpoon(
            true_model_configurations,
            noise_sigma )


        # ok, test the fit
        fit_model, fit_stat = test_fit_algorithm(
            true_model,
            lambda data: fit_finite_configuration_spoon(
                data,
                min_samples_to_merge = int(num_cells * cell_min_sample_factor),
                noise_sigma = noise_sigma,
                verbose = False),
            num_samples_per_configuration = int(num_cells * cell_min_sample_factor * 3) )

        # add to result
        results.append( fit_model )

        logger.info( "[{0}]: found {1} configurations".format(
            num_cells,
            len(fit_model.mu_covs) ) )

        # increase num cells
        num_cells = int( num_cells * cell_step_factor )

    return results
        
##===================================================================
##===================================================================
##===================================================================
##===================================================================
##===================================================================
