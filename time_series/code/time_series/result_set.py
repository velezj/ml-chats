#####
## This file is subject to the terms and conditions defined in
## file 'LICENSE.txt', which is part of this source code package.
####


import logging
logger = logging.getLogger( __name__ )

import copy

##=============================================================================

##
# A result set contains the results for a set of runs of an
# algorithm with a dataset generator.
# We keep the actual timeseries used as well as the generator to
# be able to recreate things.
#
# ResutlSets are usually created using the collect_results() of a RunSpec
class ResultSet( object ):
    def __init__(self,
                 ds_generator,
                 algorithm,
                 num_datapoints,
                 num_runs,
                 min_datapoints,
                 results,
                 datasets,
                 full_algorithm_results ):
        self.dataset_generator = ds_generator
        self.algorithm = algorithm
        self.num_datapoints = num_datapoints
        self.num_runs = num_runs
        self.min_datapoints = min_datapoints
        self.results = results
        self.datasets = datasets
        self.full_algorithm_results = full_algorithm_results

##=============================================================================

##
# A running spec.
#
# This bins a dataset generator and an algorithm and allows us to
# create a ResultSet using collect_results()
class RunSpec( object ):
    def __init__(self, ds_generator, algorithm ):
        self.ds_generator = ds_generator
        self.algorithm = algorithm
        self.identifier = "Spec( {0} with {1} )".format(
            self.ds_generator.identifier,
            self.algorithm.identifier ) 

    def collect_results( self,
                         num_datapoints,
                         num_runs,
                         min_datapoints = 3 ):

        res = []
        dsets = []
        full_results = []
        for i in xrange( num_runs ):

            local_res = []
            local_full = []

            # generate a dataset
            ds = self.ds_generator( num_datapoints )
            dsets.append( ds )

            # iterate over the different suffixes of the data
            for data_idx_i in xrange( num_datapoints - min_datapoints - 1 ):
                ts_idx = data_idx_i + min_datapoints

                target = ds.time_series[ ts_idx + 1 ]
                history = copy.deepcopy( ds.time_series[ : ts_idx + 1 ] )
                groudtruth_bad = ( ts_idx + 1 ) in ds.bad_indices

                alg_res, full = self.algorithm( target, history )
                local_res.append( ( ts_idx + 1, groudtruth_bad, alg_res ) )
                local_full.append( full )
            res.append( local_res )
            full_results.append( local_full )

        # create a resutl structure
        return ResultSet(
            self.ds_generator,
            self.algorithm,
            num_datapoints,
            num_runs,
            min_datapoints,
            res,
            dsets,
            full_results)
    

##=============================================================================

##
# Given a ResutlSet and a threshold for significance, computes
# the classifier statistics for the resutls in the ResultSet.
# This includes precision, recall, as well as true positive, true negative,
# false positive anf false negative counts.
def compute_classifier_stats(
        result_set,
        alg_result_threshold = 2.0 ):

    # Ok, compute true positive, true negative, false positive, false negative
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for results in result_set.results:
        for ( idx, truth_bad, alg_res ) in results:
            alg_bad = ( alg_res >= alg_result_threshold )
            if truth_bad and alg_bad:
                true_positive += 1
            elif truth_bad and not alg_bad:
                false_negative += 1
            elif not truth_bad and alg_bad:
                false_positive += 1
            elif not truth_bad and not alg_bad:
                true_negative += 1
            else:
                raise RuntimeError( "hgug? somhow got not booleans" )

    # make into floats for computations
    true_positive = float( true_positive )
    true_negative = float( true_negative )
    false_positive = float( false_positive )
    false_negative = float( false_negative )

    # compute common metrics
    if true_positive + false_positive > 0:
        precision = true_positive / ( true_positive + false_positive )
    else:
        precision = -1
    if true_positive + false_negative > 0:
        recall = true_positive / ( true_positive + false_negative )
    else:
        recall = -1
    if true_positive + false_positive + true_negative + false_negative > 0:
        accuracy = ( true_positive + true_negative ) / ( true_positive + false_positive + true_negative + false_negative )
    else:
        accuracy = -1
    if false_positive + true_negative > 0:
        fall_out = false_positive / ( false_positive + true_negative )
    else:
        fall_out = -1

    return {
        'tp' : true_positive,
        'tn' : true_negative,
        'fp' : false_positive,
        'fn' : false_negative,
        'precision' : precision,
        'recall': recall,
        'accuracy' : accuracy,
        'fall_out' : fall_out,
    }

##=============================================================================
