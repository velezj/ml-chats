import logging
logger = logging.getLogger( __name__ )

import numpy as np

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
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
##=======================================================================
