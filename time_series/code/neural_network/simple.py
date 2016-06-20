import logging
logger = logging.getLogger( __name__ )

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

##=========================================================================

##
# A simple, fixed, 1 hidden layer network with a
# single input and single output good for a
# scalar function :-).
#
# Wil *do* include an offset/bias term
class SimpleScalarF_1Layer( object ):

    ##
    # Build an new neurla network with a single hidden layer
    # with the given number of hidden nodes
    def __init__( self,
                  learning_rate = 0.001,
                  hidden_layer_size = 10,
                  init_weight_max_magnitude = 1e-1,
                  output_sigma = 2.0):
        self.hidden_layer_size = hidden_layer_size
        self.output_sigma = output_sigma
        self.learning_rate = learning_rate

        # the hidden layer weights
        self.V = np.random.random( size=(hidden_layer_size, 2 ) ) * init_weight_max_magnitude

        # the output layer weights
        self.w = np.random.random( size=self.hidden_layer_size ) * init_weight_max_magnitude

        # the number of epochs seen
        self.epoch = 0

    ##
    # Trains the network using the given data.
    # We will use min-batched stochastic gradient descent
    def train( self, data, max_epochs = np.inf, min_epochs = 0 ):

        # compute the pre-training log likelihood
        pre_ll = self.ll( data )

        # keep feeding in training data
        counter = 0
        while counter < max_epochs:

            # count new epoch
            self.epoch += 1
            counter += 1
            
            # iterate over each element randomly
            for i in xrange(len(data)):
                idx = np.random.choice(len(data))
                (input,output) = data[idx]
                
                # convert from data to input x (including clamped last)
                x = np.array( [ input, 1.0] )
                y = np.array( [ output ] )

                # update weights for input
                self.update_weights_using_gradient_learning( x, y )

                
            # grab new log likelihood
            new_ll = self.ll( data )

            # stop when we are no longer better
            if counter >= min_epochs and new_ll <= pre_ll:
                break

        return counter

    ##
    # Produce output for inputs
    def regress(self, xs):
        ys = []
        for x in xs:
            x = np.array([x,1.0])
            z = np.zeros( self.hidden_layer_size )
            for h_i in xrange(self.hidden_layer_size):
                z_i = np.dot( self.V[h_i,:], x )
                z_i = scipy.special.expit( z_i )
                z[ h_i ] = z_i
            mu = np.dot( self.w, z )
            y = mu
            ys.append( y )
        return ys

    ##
    # The log likelihood for a data set
    def ll(self, data):
        res = 0.0
        for (e,y) in data:
            x = np.array( [e, 1.0] )
            y = np.array( [y] )
            res += self._single_ll( x, y )
        return res

    ##
    # The log likelihood for a y given an x
    def _single_ll(self, x, y ):

        # compute z
        z = np.zeros( self.hidden_layer_size )
        for h_i in xrange(self.hidden_layer_size):
            z_i = np.dot( self.V[h_i,:], x )
            z_i = scipy.special.expit( z_i )
            z[ h_i ] = z_i

        # Ok, merge for gaussian mean
        mu = np.dot( self.w, z )

        # compute pdf
        return scipy.stats.norm.logpdf( y, loc=mu, scale=self.output_sigma)


    ##
    # Update weights using gradient and learning rate
    def update_weights_using_gradient_learning(self, x, y ):

        # Foward pass, compute inputs and such for nertwork
        a = np.zeros( shape=(self.hidden_layer_size,1) )
        z = np.zeros( shape=(self.hidden_layer_size,1) )
        b = np.zeros( shape=(1,1) )
        for h_i in xrange(self.hidden_layer_size):
            a_i = np.dot( self.V[ h_i, :] , x )
            a[ h_i ] = a_i
            z_i = scipy.special.expit( a_i )
            z[ h_i ] = z_i
        b = np.dot( self.w, z )

        # ok, gradient updating using learning rate :-)

        # output layer error
        d2 = b - y

        # backpropagate to get hidden layer error
        d1 = np.zeros( shape=(self.hidden_layer_size,1) )
        for h_i in xrange( self.hidden_layer_size ):
            sig_a = scipy.special.expit( a[h_i] )
            deriv_a = sig_a * (1.0 - sig_a)
            d1[h_i] = d2 * self.w[h_i] * deriv_a

        # Ok, we now have the gradient, so move in the opposite
        # direction multiplied by hte learning rate
        for h_i in xrange(self.hidden_layer_size):
            self.w[h_i] += self.learning_rate * (- d2 * z[h_i])
            self.V[h_i,:] += self.learning_rate * (- d1[h_i] * x)

##=========================================================================

def plot_fits( nn,
               data,
               train_epochs = 100,
               num_iterations = 5 ):

    # grab only the xs
    xs = map(lambda o: o[0], data )
    ys = map(lambda o: o[1], data )

    # ok, see the regression after a number of training epochs
    regs = []
    lls = []
    regs.append( nn.regress( xs ) )
    lls.append( nn.ll( data ) )
    for i in xrange(num_iterations):

        nn.train( data,
                  min_epochs = train_epochs,
                  max_epochs = train_epochs )

        regs.append( nn.regress( xs ) )
        lls.append( nn.ll( data ) )

    # plot the regressions
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,(reg,ll) in enumerate(zip(regs,lls)):
        x = xs
        y = reg
        ax.plot( x, y, label="{0} ll={1}".format( i * train_epochs,
                                                   ll) )
    # plot hte actual data
    ax.plot( xs, ys, 'k--', lw=2, label='data')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
##=========================================================================

def generate_hill_data(num_points=100):
    x = np.linspace( -10, 10, num_points )
    y = np.power( x, 2.0 )
    y -= np.mean(y)
    return zip( x, y )

##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
##=========================================================================
