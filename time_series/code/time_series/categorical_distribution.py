import logging
logger = logging.getLogger( __name__ )

import numpy as np


##========================================================================

##
# A categorical distribution represented as counts
class cat_distibution( object ):
    def __init__( self, counts ):
        self.counts = counts

    def mode(self):
        if len(self.counts) < 1:
            return None
        best_k = self.counts.keys()[0]
        best_c = self.counts[ best_k ]
        for k,c in self.counts.iteritems():
            if c > best_c:
                best_k = k
                best_c = c
        return best_k

    def credible_interval( self, p ):
        mode_k = self.mode()
        if mode_k is None:
            return None

        # ok, search around the mode
        ordered_k = sorted( self.counts.keys() )
        mode_idx = ordered_k.index( mode_k )
        max_i = max( mode_idx - 1, len(ordered_k) - mode_idx )
        total = float(np.sum( self.counts.values() ))
        mass = self.counts[ mode_k ] / total
        interval = set({})
        interval.add( mode_k )
        for i in xrange( max_i ):
            high_i = mode_idx + ( i + 1 )
            low_i = mode_idx - ( i + 1 )
            if mass >= p:
                break
            pot_mass = 0.0
            low_mass = None
            high_mass = None
            low_k = mode_k
            high_k = mode_k
            if low_i >= 0:
                low_mass = self.counts[ ordered_k[ low_i ] ] / total
                pot_mass += low_mass
                low_k = ordered_k[ low_i ]
            if high_i < len(ordered_k):
                high_mass = self.counts[ ordered_k[ high_i ] ] / total
                pot_mass += high_mass
                high_k = ordered_k[ high_i ]
            if mass + pot_mass <= p:
                interval.add( low_k )
                interval.add( high_k )
                mass += pot_mass
            elif low_mass is None:
                if high_mass is None:
                    interval.add( high_k )
                    mass += high_mass
                else:
                    raise RuntimeError( "No mass for anything!!" )
            elif high_mass is None:
                if low_mass is not None:
                    interval.add( low_k )
                    mass += low_mass
                else:
                    raise RuntimeError( "No mass to anything!!" )
            else:
                # hmm , adding both might be too mush, lets add
                # the one which reaches closer to mass first
                if low_mass + mass >= p and high_mass + mass >= p:
                    if low_mass < high_mass:
                        interval.add( low_k )
                        mass += low_mass
                    else:
                        interval.add( high_k )
                        mass += high_mass
                elif low_mass + mass >= p:
                    interval.add( low_k )
                    mass += low_mass
                elif high_mass + mass >= p:
                    interval.add( high_k )
                    mass += high_mass
                else:
                    raise RuntimeError( "Mass not conserved!" )

        # ok no compute the interval as low/high
        order = sorted(interval)
        return ( ( order[0], order[-1] ),
                 mass )
                    

##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
