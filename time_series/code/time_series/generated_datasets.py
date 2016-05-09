import logging
logger = logging.getLogger( __name__ )

import random
import math
import datetime
import pickle
import copy


##=============================================================================

##
# A testing dataset (usually generated)
class GeneratedDataset(object):
    def __init__( self, taxonomy, time_series, bad_indices ):
        self.taxonomy = taxonomy
        self.time_series = time_series
        self.bad_indices = set(bad_indices)

##=============================================================================

##
# Generates a dataset with a given number of samples
class DatasetGenerator(object):
    def __init__( self, identifier ):
        self.identifier = identifier
    def __call__(self, num_datapoints):
        pass

##=============================================================================

##
# Transforms a generated dataset into another generated dataset.
# This *can modify* the input dataset and return it, hence do *not*
# assume that a copy of the datasets is made before the transform happnes
# :-)
class DatasetTransformer(object):
    def __init__(self, identifier):
        self.identifier = identifier

    ##
    # Returns a tranformed dataset. This may share structure with the
    # input (so in fact this method may *modify* the input dataset and
    # return it ratehr than returning a fresh copy
    #
    # do *not* override this, override _transform instead
    def __call__(self, base_datasets):
        res_datasets = self._transform( base_datasets )
        for ds in res_datasets:
            ds.taxonomy.append( self.identifier )
        return res_datasets

    ##
    # Returns a tranformed dataset. This may share structure with the
    # input (so in fact this method may *modify* the input dataset and
    # return it ratehr than returning a fresh copy
    def _transform( self, base_datasets ):
        pass

##=============================================================================

##
# A time-series data generator
class TimeSeriesDataGenerator( object ):
    def __init__( self, identifier ):
        self.identifier = identifier

    ##
    # To be overridden by subclasses :-)
    def __call__( self, num_datapoints ):
        pass

##=============================================================================

##
# A Time Series Transformer
class TimeSeriesDataTransformer( object ):
    def __init__( self, identifier ):
        self.identifier = identifier

    def __call__(self, time_series):
        pass
        

##=============================================================================

##
# A Function timeseries generator which take a time generator and
# a function an return f( t ) for all time points generator by
# the time generator
class FunctionalTimeSeries( TimeSeriesDataGenerator ):
    def __init__( self, function_id, function, time_gen ):
        self.function = function
        self.time_gen = time_gen
        self.function_id = function_id
        ident = "{0}({1})".format( self.function_id, self.time_gen.identifier )
        TimeSeriesDataGenerator.__init__( self, ident )

    def __call__( self, num_datapoints ):
        return map(self.function, self.time_gen(num_datapoints))

##=============================================================================

##
# A simple linear time generator
class LinearTime( TimeSeriesDataGenerator ):
    def __init__(self, alpha = 1.0, offset = 0):
        self.offset = offset
        self.alpha = alpha
        ident = '{0}*t+{1}'.format(
            self.alpha,
            self.offset)
        TimeSeriesDataGenerator.__init__(self, ident )
    def __call__(self,num_datapoints):
        return map(lambda t: self.alpha * t + self.offset,
                   range(num_datapoints) )

##=============================================================================

##
# Exponential time generator
class ExponentialTime( TimeSeriesDataGenerator ):
    def __init__(self, alpha, beta, offset= 0 ):
        self.alpha = alpha
        self.beta = beta
        self.offset = offset
        ident = "{0}*Exp({1} t)+{2}".format(
            self.alpha,
            self.beta,
            self.offset )
        TimeSeriesDataGenerator.__init__( self, ident )

    def __call__(self, num_datapoints):
        return map(lambda t: self.alpha * math.exp( self.beta * t ) + self.offset, xrange(num_datapoints))

##=============================================================================

##
# Accelerating time
class StepwiseAcceleratingTime( TimeSeriesDataGenerator ):
    def __init__(self, init_pos, init_vel, accel ):
        self.init_vel = init_vel
        self.init_pos = init_pos
        self.accel = accel
        ident = "StepAccel( {0} * t + {1}, {2} )".format(
            self.init_vel,
            self.init_pos,
            self.accel )
        TimeSeriesDataGenerator.__init__( self, ident )

    def __call__(self, num_datapoints):
        ts = []
        pos = self.init_pos
        vel = self.init_vel
        for i in xrange(num_datapoints):
            ts.append( pos )
            pos += vel
            vel += self.accel
        return ts

##=============================================================================

##=============================================================================

##
# A sin time serier generator
class SinTimeSeries( FunctionalTimeSeries ):
    def __init__( self, amplitude, phase, period, time=LinearTime(), offset=0.0 ):
        self.amplitude = amplitude
        self.phase = phase
        self.period = period
        self.time = time
        self.offset = offset
        def _func( t ):
            return self.amplitude * math.sin( 2 * math.pi / (period - 1) * t + self.phase ) + self.offset
        ident = "Sin[{0},{1},{2},{3}]".format(
            self.amplitude,
            self.phase,
            self.period,
            self.offset)
        FunctionalTimeSeries.__init__( self, ident, _func, self.time )

##=============================================================================

##
# A abs(sin) time serier generator
class AbsSinTimeSeries( FunctionalTimeSeries ):
    def __init__( self, amplitude, phase, period, time=LinearTime(), offset=0.0 ):
        self.amplitude = amplitude
        self.phase = phase
        self.period = period
        self.time = time
        self.offset = offset
        def _func( t ):
            return self.amplitude * abs( math.sin( 2 * math.pi / (period - 1) * t + self.phase ) ) + self.offset
        ident = "AbsSin[{0},{1},{2},{3}]".format(
            self.amplitude,
            self.phase,
            self.period,
            self.offset)
        FunctionalTimeSeries.__init__( self, ident, _func, self.time )

##=============================================================================

##
# A constant time series
class ConstantTimeSeries( FunctionalTimeSeries ):
    def __init__( self, value ):
        self.value = value
        def _func( t ):
            return self.value
        ident = "Constant[{0}]".format( self.value )
        FunctionalTimeSeries.__init__( self, ident, _func, LinearTime() )

##=============================================================================

##
# a Sawtooth time series with a given period (in time)
class SawtoothTimeSeries( FunctionalTimeSeries ):
    def __init__( self, period, high, low, time=LinearTime() ):
        self.period = period
        self.high = high
        self.low = low
        period_norm = period - 1
        if period <= 1:
            period_norm = period
        self.unit_step = (high - low) / period_norm
        def _func( t ):
            r = t - int(t / self.period) * self.period
            return self.high - r * self.unit_step
        ident = "Saw[{0}->{1}, {2}]".format( self.high, self.low, self.period )
        FunctionalTimeSeries.__init__( self, ident, _func, time )

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================

##
# A Transformer that does things elemntiwise by applying a pattern/function
class ElementwiseTimeSeriesDataTransformer( TimeSeriesDataTransformer ):
    def __init__( self, transform_id, function ):
        self.function = function
        TimeSeriesDataTransformer.__init__(self, transform_id )
    def __call__(self, time_series ):
        return map(self.function, enumerate(time_series))
    
                
##=============================================================================

##
# A noise generator (just a function with a name :-) )
class NoiseGenerator( object ):
    def __init__( self, identifier ):
        self.identifier = identifier
    def __call__(self):
        pass

##=============================================================================

##
# A Gaussian noise generator
class GaussianNoiseGenerator( NoiseGenerator ):
    def __init__( self, mean, sigma, min = None, max = None ):
        self.mean = mean
        self.sigma = sigma
        self.min = min
        self.max = max
        trunc_string = ""
        if self.min is not None or self.max is not None:
            trunc_string = "[{0},{1}]".format(
                self.min if self.min is not None else '-inf',
                self.max if self.max is not None else 'inf' )
        ident = "Norm({0},{1}{2})".format(
            self.mean,
            self.sigma,
            trunc_string)
        NoiseGenerator.__init__( self, ident )
    def __call__(self):
        a = random.normalvariate( self.mean, self.sigma )
        while self._not_in_range( a ):
            a = random.normalvariate( self.mean, self.sigma )
        return a
    def _not_in_range( self, x ):
        if self.min is not None and x < self.min:
            return True
        if self.max is not None and x > self.max:
            return True
        return False

##=============================================================================

##
# A Uniform noise generator
class UniformNoiseGenerator( NoiseGenerator ):
    def __init__( self, low, high ):
        self.low = low
        self.high = high
        ident = "Uniform({0},{1})".format(
            self.low,
            self.high)
        NoiseGenerator.__init__( ident )
    def __call__(self):
        return random.uniform( self.low, self.high )

##=============================================================================

##
# Additive noise elementwise tranformer
class AdditiveNoiseDataTransformer( ElementwiseTimeSeriesDataTransformer ):
    def __init__( self, noise_generator ):
        def _transform( idx_x ):
            idx, x = idx_x
            return x + noise_generator()
        ElementwiseTimeSeriesDataTransformer.__init__(
            self,
            noise_generator.identifier,
            _transform )

##=============================================================================

##
# Adds a additive linear trend
class AdditiveLinearTrendDataTransformer( ElementwiseTimeSeriesDataTransformer ):
    def __init__( self, alpha, offset = 0.0 ):
        def _transform( idx_x ):
            idx, x = idx_x
            val = alpha * idx + offset
            return x + val
        ident = "LinearTrend[{0}x + {1}]".format(
            alpha,
            offset )
        ElementwiseTimeSeriesDataTransformer.__init__(
            self,
            ident,
            _transform )

##=============================================================================

##
# A Data Error Indicer which takes a clean time series and
# induces errors on it
class ErrorInducer( DatasetTransformer ):
    def __init__( self, inducer_id ):
        self.inducer_id = inducer_id
        self.identifier = "Induced[{0}]".format( self.inducer_id )

##=============================================================================

##
# An ordered sequence of error inducers
class ErrorInducerPipeline( ErrorInducer ):
    def __init__( self, inducers ):
        self.inducers = inducers
        ident = "Induced{0}".format(
            map(lambda ei: ei.inducer_id, self.inducers))
        self.inducer_id = ident
        self.identifier = ident
    def _transform(self, base_datasets ):
        result = base_datasets
        for inducer in self.inducers:
            result = inducer( result )
        return result

##=============================================================================

##
# A swap-error inducer.
# Here, a randomly chosen pair of the dataset is swapped
# The resulting dataset has the bad indices set
class SwapErrorInducer( ErrorInducer ):
    def __init__( self, first_index = None, second_index = None ):
        self.first_index = first_index
        self.second_index = second_index
        ident = "Swap({0},{1})".format(
            self.first_index if first_index is not None else 'RAND',
            self.second_index if second_index is not None else 'RAND' )
        ErrorInducer.__init__(self, ident )

    ##
    # This *will* modify the given datasets and return the modified versions.
    # So return datasets *shared structure* with the input datasets :-)
    def _transform(self, base_datasets ):
        return map(self._swap,base_datasets)

    def _swap(self,dataset):

        # grab the length
        ds_len = len(dataset.time_series)
        
        # choose first,second index if random, float fraction, or integer index
        if self.first_index is None:
            first_index = random.randint( 0, ds_len - 1 )
        elif isinstance( self.fisrt_index, float ):
            first_index = int(self.first_index * (ds_len - 1))
        elif self.first_index < ds_len and self.first_index >= 0:
            first_index = self.first_index
        else:
            raise RuntimeError( "Bad first index for Swap error inducer: first index = {0}, dataset length = {1}".format( self.first_index, ds_len ) )

        if self.second_index is None:
            second_index = random.randint( 0, ds_len - 1 )
        elif isinstance( self.fisrt_index, float ):
            second_index = int(self.second_index * (ds_len - 1))
        elif self.second_index < ds_len and self.second_index >= 0:
            second_index = self.second_index
        else:
            raise RuntimeError( "Bad second index for Swap error inducer: second index = {0}, dataset length = {1}".format( self.second_index, ds_len ) )

        # Ok, actually swap the dataset elements in timeseries and set
        # bad indices. This MODIFIES the given dataset :-)
        temp = dataset.time_series[ first_index ]
        dataset.time_series[ first_index ] = dataset.time_series[second_index]
        dataset.time_series[ second_index ] = temp
        dataset.bad_indices.add( first_index )
        dataset.bad_indices.add( second_index )
        return dataset
        

##=============================================================================

##
# A particular index is set to a given fixed value
class FixedValueErrorInducer( ErrorInducer ):
    def __init__( self, value, start_index = None, end_index = None ):
        self.start_index = start_index
        self.end_index = end_index
        self.value = value
        ident = "Set({0} @ {1}, {2} )".format(
            self.value,
            self.start_index if start_index is not None else 'RAND',
            self.end_index if end_index is not None else 'RAND' )
        ErrorInducer.__init__(self, ident )    

    def _transform(self, base_datasets ):
        return map( self._set, base_datasets )

    def _set( self, dataset ):

        # grab the dataset length
        ds_len = len(dataset.time_series)

        # compute index if needed
        if self.start_index is None:
            start_index = random.randint( 0, ds_len - 1 )
        elif isinstance( self.start_index, float ):
            start_index = int(self.start_index * ( ds_len - 1 ))
        elif self.start_index < ds_len and self.start_index >= 0:
            start_index = self.start_index
        else:
            raise RuntimeError( "Bad start  index for FixedValue error inducer: index = {0}, ds length = {1}".format( self.start_index, ds_len ) )
        if self.end_index is None and self.start_index is None:
            end_index = start_index
        elif self.end_index is None:
            end_index = random.randint( start_index, ds_len - 1 )
        elif isinstance( self.end_index, float ):
            end_index = int(self.end_index * ( ds_len - 1 ))
        elif self.end_index < ds_len and self.end_index >= start_index:
            end_index = self.end_index
        else:
            raise RuntimeError( "Bad end  index for FixedValue error inducer: index = {0}, start = {1}, ds length = {2}".format( self.end_index, start_index, ds_len ) )
        
        
        # set the value. This *modifies* the given dataset and then returns it
        for index in xrange( start_index, end_index + 1 ):
            dataset.time_series[ index ] = self.value
            dataset.bad_indices.add( index )
        return dataset

##=============================================================================

##
# An error inducer for *additive* noise
class AdditiveNoiseErrorInducer( ErrorInducer ):

    ##
    # Start and end indices are inclusive :-)
    def __init__( self, noise_generator, start_index = 0.0, end_index = 1.0 ):
        self.noise_generator = noise_generator
        self.start_index = start_index
        self.end_index = end_index
        ident = "AdditiveNoiseErrors({0} @ {1} to {2} )".format(
            self.noise_generator.identifier,
            self.start_index if self.start_index is not None else 'RAND',
            self.end_index if self.end_index is not None else 'RAND' )
        ErrorInducer.__init__(self, ident )

    def _transform( self, base_datasets ):
        return map( self._add_noise, base_datasets )

    def _add_noise( self, dataset ):

        ds_len = len(dataset.time_series)

        # compute start, end indices if needed
        if self.start_index is None:
            start_index = random.randint( 0, ds_len - 1 )
        elif isinstance( self.start_index, float ):
            start_index = int(self.start_index * ( ds_len - 1 ))
        elif self.start_index < ds_len and self.start_index >= 0:
            start_index = self.start_index
        else:
            raise RuntimeError( "Bad start index for Additive Noise error inducer: index = {0}, ds len = {1}".format( self.start_index, ds_len ) )
        if self.end_index is None:
            end_index = random.randint( 0, ds_len - 1 )
        elif isinstance( self.end_index, float ):
            end_index = int(self.end_index * ( ds_len - 1 ))
        elif self.end_index < ds_len and self.end_index >= 0:
            end_index = self.end_index
        else:
            raise RuntimeError( "Bad end index for Additive Noise error inducer: index = {0}, ds len = {1}".format( self.end_index, ds_len ) )

        # ok, iterate over the wanted indices and add some noise
        for idx in xrange( start_index, end_index + 1 ):
            dataset.time_series[ idx ] += self.noise_generator()
            dataset.bad_indices.add( idx )

        return dataset

##=============================================================================

class DatasetGeneratorPipeline( DatasetGenerator ):

    def __init__( self, pipeline_id, base_generator, transformers ):
        self.pipeline_id = pipeline_id
        self.base_generator = base_generator
        self.transformers = transformers
        ident = "{0}:= {1}( {2} )".format(
            self.pipeline_id,
            map(lambda trans: trans.identifier,
                self.transformers),
            self.base_generator.identifier )
        DatasetGenerator.__init__( self, ident )

    def __call__(self, num_datapoints ):

        # generate original dataset
        dset = [ self.base_generator( num_datapoints ) ]

        # apply the transforms
        for trans in self.transformers:
            dset = trans( dset )

        # return
        return dset[0]

##=============================================================================

##
# Creates a new *pristine* dataset (no bad indices) from a
# time series generator and time series tranforms applied in order
class TimeSeriesPipeline( DatasetGenerator ):
    def __init__( self, pipeline_id, time_series_generator, transformers ):
        self.pipeline_id = pipeline_id
        self.transformers = transformers
        self.time_series_generator = time_series_generator
        ident = "Dset{{ {0} = {1}( {2} ) }}".format(
            self.pipeline_id,
            map(lambda trans: trans.identifier, self.transformers ),
            self.time_series_generator.identifier)
        DatasetGenerator.__init__(self, ident )

    def __call__( self, num_datapoints ):

        # generate the time series
        ts = self.time_series_generator( num_datapoints )

        # transform the time series
        for trans in self.transformers:
            ts = trans( ts )

        # create a new dataset with initial taxonomy and no bad indices
        taxonomy = [ self.identifier ]
        bad_indices = set([])
        return GeneratedDataset( taxonomy,
                                 ts,
                                 bad_indices )
        
        

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================

##
# Plots a given dataset
def plot_dataset( dataset ):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot( dataset.time_series, 'b-' )
    plt.plot( dataset.time_series, 'bo' )
    for idx in dataset.bad_indices:
        plt.plot( idx, dataset.time_series[idx], 'rx' )
    plt.show()

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================

##
# A non-periodic dataset generator with a given percent noise sigma
def create_nonperiodic_guassian_noise_generator(
        identifier,
        mean,
        noise_sigma ):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        ConstantTimeSeries( mean ),
        [ AdditiveNoiseDataTransformer(
            GaussianNoiseGenerator( 0.0,
                                    noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A non-periodic dataset generator with a given percent noise sigma
# and with a trend
def create_nonperiodic_trending_guassian_noise_generator(
        identifier,
        mean,
        noise_sigma,
        trend):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        ConstantTimeSeries( mean ),
        [ AdditiveLinearTrendDataTransformer( trend ),
          AdditiveNoiseDataTransformer(
              GaussianNoiseGenerator( 0.0,
                                      noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A periodic dataset generatio with given noice sigma
def create_periodic_guassian_noise_generator(
        identifier,
        mean,
        amplitude,
        period,
        noise_sigma ):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        SinTimeSeries( amplitude, 0.0, period, time=LinearTime(), offset=mean ),
        [ AdditiveNoiseDataTransformer(
            GaussianNoiseGenerator( 0.0,
                                    noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A periodic dataset generatio with given noice sigma
def create_periodic_trending_guassian_noise_generator(
        identifier,
        mean,
        amplitude,
        period,
        noise_sigma,
        trend ):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        SinTimeSeries( amplitude, 0.0, period, time=LinearTime(), offset=mean ),
        [ AdditiveLinearTrendDataTransformer( trend ),
          AdditiveNoiseDataTransformer(
              GaussianNoiseGenerator( 0.0,
                                      noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A sawtooth dataset generatio with given noice sigma
def create_sawtooth_guassian_noise_generator(
        identifier,
        high,
        low,
        period,
        noise_sigma ):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        SawtoothTimeSeries( period, high, low, time=LinearTime() ),
        [ AdditiveNoiseDataTransformer(
            GaussianNoiseGenerator( 0.0,
                                    noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A sawtooth dataset generatio with given noice sigma and trend
def create_sawtooth_trending_guassian_noise_generator(
        identifier,
        high,
        low,
        period,
        noise_sigma,
        trend):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        SawtoothTimeSeries( period, high, low, time=LinearTime() ),
        [ AdditiveLinearTrendDataTransformer( trend ),
          AdditiveNoiseDataTransformer(
            GaussianNoiseGenerator( 0.0,
                                    noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A periodic dataset generatio with given noice sigma
def create_abssin_guassian_noise_generator(
        identifier,
        mean,
        amplitude,
        period,
        noise_sigma ):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        AbsSinTimeSeries( amplitude, 0.0, period, time=LinearTime(), offset=mean ),
        [ AdditiveNoiseDataTransformer(
            GaussianNoiseGenerator( 0.0,
                                    noise_sigma ) ) ] )
    return ts_pipe

##=============================================================================

##
# A periodic trending dataset generatio with given noice sigma
def create_abssin_trending_guassian_noise_generator(
        identifier,
        mean,
        amplitude,
        period,
        noise_sigma,
        trend):

    ts_pipe = TimeSeriesPipeline(
        identifier,
        AbsSinTimeSeries( amplitude, 0.0, period, time=LinearTime(), offset=mean ),
        [ AdditiveLinearTrendDataTransformer( trend ),
          AdditiveNoiseDataTransformer(
              GaussianNoiseGenerator( 0.0,
                                      noise_sigma ) ) ] )
    return ts_pipe


##=============================================================================
##=============================================================================


##=============================================================================
##=============================================================================

##
# Some dataset generators

SIGNAL_FLOOR = 10000.0
DSS = []
for drop_percent in [ 0.0, 0.1, 0.5, 0.8 ]:
    for noise_sigma in [ 0.01 * SIGNAL_FLOOR,
                         0.10 * SIGNAL_FLOOR,
                         0.50 * SIGNAL_FLOOR,
                         0.80 * SIGNAL_FLOOR ]:

        # non-periodic non-trend percent drop
        DSS.append(
            DatasetGeneratorPipeline(
                "Dropper {0}".format(drop_percent),
                create_nonperiodic_guassian_noise_generator(
                    "non-periodic, non-trend, {1}, {0} noise".format(
                        noise_sigma,
                        SIGNAL_FLOOR),
                    SIGNAL_FLOOR,
                    noise_sigma ),
                [ FixedValueErrorInducer(drop_percent * SIGNAL_FLOOR,1.0,1.0) ]))

        # trending non-periodic percent drop
        for trend in [ 1.0, -1.0, 10.0, -10.0, 0.2 / 30.0 * SIGNAL_FLOOR, -0.2 / 30.0 * SIGNAL_FLOOR ]:
            DSS.append(
                DatasetGeneratorPipeline(
                    "Dropper {0}".format(drop_percent),
                    create_nonperiodic_trending_guassian_noise_generator(
                        "non-periodic, trend {1}, {2}, {0} noise".format(
                            noise_sigma, trend, SIGNAL_FLOOR),
                        SIGNAL_FLOOR,
                        noise_sigma,
                        trend),
                    [ FixedValueErrorInducer(drop_percent * SIGNAL_FLOOR,1.0,1.0) ]))
            
        
        for period in [ 7, 10, 20, 30 ]:
            for amp in [ 0.01 * SIGNAL_FLOOR,
                         0.10 * SIGNAL_FLOOR,
                         0.50 * SIGNAL_FLOOR,
                         0.80 * SIGNAL_FLOOR ]:

                # periodic non-trend percent drop
                DSS.append(
                    DatasetGeneratorPipeline(
                        "Dropper {0}".format(drop_percent),
                        create_periodic_guassian_noise_generator(
                            "periodic {0} {1}, non-trend, {3}, {2} noise".format(
                                period, amp, noise_sigma, SIGNAL_FLOOR),
                            SIGNAL_FLOOR,
                            amp,
                            period,
                            noise_sigma ),
                        [ FixedValueErrorInducer(drop_percent * SIGNAL_FLOOR,1.0,1.0) ]))

                # sawtooth non-trend percent drop
                DSS.append(
                    DatasetGeneratorPipeline(
                        "Dropper {0}".format(drop_percent),
                        create_sawtooth_guassian_noise_generator(
                            "sawtooth {0} {1}, non-trend, {3}, {2} noise".format(
                                period, amp, noise_sigma, SIGNAL_FLOOR),
                            amp + SIGNAL_FLOOR,
                            SIGNAL_FLOOR,
                            period,
                            noise_sigma ),
                        [ FixedValueErrorInducer(drop_percent * SIGNAL_FLOOR,1.0,1.0) ]))


                # periodic trending percent drop
                for trend in [ 1.0, -1.0, 10.0, -10.0 ]:
                    DSS.append(
                        DatasetGeneratorPipeline(
                            "Dropper {0}".format(drop_percent),
                            create_periodic_trending_guassian_noise_generator(
                                "periodic {0} {1}, trend {3}, {4}, {2} noise".format(
                                    period, amp, noise_sigma, trend, SIGNAL_FLOOR),
                                SIGNAL_FLOOR,
                                amp,
                                period,
                                noise_sigma,
                                trend),
                            [ FixedValueErrorInducer(drop_percent * SIGNAL_FLOOR,1.0,1.0) ]))

                    # sawtooth trend percent drop
                    DSS.append(
                        DatasetGeneratorPipeline(
                            "Dropper {0}".format(drop_percent),
                            create_sawtooth_trending_guassian_noise_generator(
                                "sawtooth {0} {1}, trend {3}, {4}, {2} noise".format(
                                    period, amp, noise_sigma, trend, SIGNAL_FLOOR),
                                amp + SIGNAL_FLOOR,
                                SIGNAL_FLOOR,
                                period,
                                noise_sigma,
                                trend),
                            [ FixedValueErrorInducer(drop_percent * SIGNAL_FLOOR,1.0,1.0) ]))


                    
##
# A second set of dataset generators

GOOD_DSS = [
    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_nonperiodic_guassian_noise_generator(
            "non-periodic, non-trend, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR),
            SIGNAL_FLOOR,
            0.1 * SIGNAL_FLOOR ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_nonperiodic_trending_guassian_noise_generator(
            "non-periodic, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                10.0),
            SIGNAL_FLOOR,
            0.1 * SIGNAL_FLOOR,
            10.0 ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_nonperiodic_trending_guassian_noise_generator(
            "non-periodic, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                -10.0),
            SIGNAL_FLOOR,
            0.1 * SIGNAL_FLOOR,
            -10.0 ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_sawtooth_trending_guassian_noise_generator(
            "saw {3}, {4}, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                10.0,
                11,
                3000),
            3000 + SIGNAL_FLOOR,
            SIGNAL_FLOOR,
            11,
            0.1 * SIGNAL_FLOOR,
            10.0 ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_sawtooth_trending_guassian_noise_generator(
            "saw {3}, {4}, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                -10.0,
                11,
                3000),
            3000 + SIGNAL_FLOOR,
            SIGNAL_FLOOR,
            11,
            0.1 * SIGNAL_FLOOR,
            -10.0 ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_abssin_trending_guassian_noise_generator(
            "abs(sin) {3}, {4}, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                10.0,
                11,
                3000),
            SIGNAL_FLOOR,
            3000.0,
            11,
            0.1 * SIGNAL_FLOOR,
            10.0 ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),


    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_abssin_trending_guassian_noise_generator(
            "abs(sin) {3}, {4}, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                -10.0,
                11,
                3000),
            SIGNAL_FLOOR,
            3000.0,
            11,
            0.1 * SIGNAL_FLOOR,
            -10.0 ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_sawtooth_trending_guassian_noise_generator(
            "saw {3}, {4}, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                0.2 / 30.0 * SIGNAL_FLOOR,
                11,
                3000),
            3000 + SIGNAL_FLOOR,
            SIGNAL_FLOOR,
            11,
            0.1 * SIGNAL_FLOOR,
            0.2 / 30.0 * SIGNAL_FLOOR ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

    DatasetGeneratorPipeline(
        "Dropper {0}".format(0.5),
        create_sawtooth_trending_guassian_noise_generator(
            "saw {3}, {4}, trend {2}, {1}, {0} noise".format(
                0.1,
                SIGNAL_FLOOR,
                -0.2 / 30.0 * SIGNAL_FLOOR,
                11,
                3000),
            3000 + SIGNAL_FLOOR,
            SIGNAL_FLOOR,
            11,
            0.1 * SIGNAL_FLOOR,
            -0.2 / 30.0 * SIGNAL_FLOOR ),
        [ FixedValueErrorInducer(0.5 * SIGNAL_FLOOR,1.0,1.0) ]),

]

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
