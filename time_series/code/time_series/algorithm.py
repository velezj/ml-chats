import logging
logger = logging.getLogger( __name__ )


##=============================================================================

##
# An algorithm specification which is just a function that takes in
# a historical timeseries (not a dataset)
# and a new target and returns a single number
class AlertAlgorithm( object ):
    def __init__( self, identifier ):
        self.identifier = identifier

    def __call__( self, target, history ):
        pass

##=============================================================================
