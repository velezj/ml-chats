import logging
logger = logging.getLogger( __name__ )

import numpy as np

from categorical_distribution import cat_distibution

##=======================================================================

##
# Estimate the autocorrlation of a sequence.
def autocorrelation_estimate( x ):
    x = np.array( x )
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate( x, x, mode='full' )[-n:]
    ac_est = r / ( variance * ( np.arange( n, 0, -1 ) ) )
    return ac_est

##=======================================================================

##
# Estimate the period of a sequence using the autocorrelation.
# Returns a distirbution over the period (discrete)
def period_estimate( x, prior = cat_distibution({}) ):

    # first, calculate autocorrelation
    ac = autocorrelation_estimate( x )

    # Ok, now ignore first datapoint and take the top 5 percent of the
    # height
    n = len(x)
    height_thresh = 2 * np.std( ac[1:n/2] )

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
        return cat_distibution( { 0: 1.0 } )

    # ok, grab the midpoint of each range
    mids = map(lambda (a,b): a + (b-a) / 2.0 , ranges )

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
    return cat_distibution( counts )


##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
