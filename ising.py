#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy
from boltzmann import SymmetricHopfieldNetwork
from mcmc import MetropolisAlgorithm, WolffClusterAlgorithm
from util import acf
import sys

import matplotlib.pyplot as pyplot	
import matplotlib.animation as animation
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

class Ising2d(SymmetricHopfieldNetwork):
    """Simple Ising model in 2d."""
    
    SPINONEHALF = 1
    SPINTHREEHALVES = 2
    
    def __init__(self, rows, cols, J = -1., spins = 1, pbc = True):
        super( Ising2d, self ).__init__( spins * rows * cols )

        #if spins != 1: raise NotImplementedError("Only spin-1/2 is supported")

        self._spins = spins
        self._rows = rows
        self._cols = cols
        self._pbc = pbc
        
        self._setup(J, rows, cols, spins)
        
    def _setup(self, J, rows, cols, spins):
        size = rows * cols * spins
        
        # For spin-1/2 Ising model, biases should be:
        #   b = - (# nearest neighbors) J / 2
        self.biases[:] = numpy.ones( spins * rows * cols ) * ( - 2. * J )
        if spins == Ising2d.SPINTHREEHALVES:
            self.biases[0::2] = numpy.ones( rows * cols ) * ( - 6. * J )
            self.biases[1::2] = numpy.ones( rows * cols ) * ( - 12. * J )
        
        ds = 1
        dc = spins
        dr = spins * cols
        
        for i in range( size ):

            # Convert to "3d" array indecies
            row = (i/dr)%rows
            col = (i/dc)%cols
            spin = (i/ds)%spins
            
            # This is mostly to remind us how to convert back to a 1d array
            assert i == row*dr + col*dc + spin*ds
            
            if self._pbc:
                # Use PBC
                for ispin in range(spins):
                    if spin == ispin and spin == 0: weight = 1. * J
                    elif spin == ispin and spin == 1: weight = 4. * J
                    else: weight = 2. * J
                    j = row*dr + ((col + 1) % cols)*dc + ((spin + ispin)%spins) * ds
                    self.weights[i,j] = weight
                    j = row*dr + ((col - 1) % cols)*dc + ((spin + ispin)%spins) * ds
                    self.weights[i,j] = weight
                    j = ((row + 1) % rows)*dr + col*dc + ((spin + ispin)%spins) * ds
                    self.weights[i,j] = weight
                    j = ((row - 1) % rows)*dr + col*dc + ((spin + ispin)%spins) * ds
                    self.weights[i,j] = weight
            else:
                # No PBC
                for ispin in range(spins):
                    if spin == ispin and spin == 0: weight = 1. * J
                    elif spin == ispin and spin == 1: weight = 4. * J
                    else: weight = 2. * J
                    if col < cols - 1:
                        j = row*dr + (col + 1)*dc + ((spin + ispin)%spins) * ds
                        self.weights[i,j] = weight
                    if col > 0:
                        j = row*dr + (col - 1)*dc + ((spin + ispin)%spins) * ds
                        self.weights[i,j] = weight
                    if row < rows - 1:
                        j = (row + 1)*dr + col*dc + ((spin + ispin)%spins) * ds
                        self.weights[i,j] = weight
                    if row > rows:
                        j = (row - 1)*dr + col*dc + ((spin + ispin)%spins) * ds
                        self.weights[i,j] = weight
    
    @property
    def spin(self):
        return self._spins
    
    @property
    def rows(self):
        return self._rows
        
    @property
    def cols(self):
        return self._cols
    
    @property
    def onGrid(self):
        return self.units.reshape(self._rows, self._cols, self._spins).sum(axis=2)

class PlotTracker:    
    def __init__(self, source, property_name, axis):
        self._source = source
        self._property_name = property_name
        self._axis = axis
        self._line = Line2D([],[])
        self._boff = Line2D([0,0],[0,1])
        self._axis.add_line(self._line)
        self._axis.set_ylabel(property_name)
        self._axis.set_xlabel('iteration')
    
    def update(self, it, burnoff = 0):
        x, y = self._line.get_data()
        x.append(it)
        y.append(eval('self._source.%s'%self._property_name))
        self._line.set_data(x,y)
        self._axis.set_xlim([0,it])
        self._axis.set_ylim(min(y[burnoff:]),max(y[burnoff:]))

class Ising2dPlot(animation.TimedAnimation):
    """GUI to examine Ising2d statistics using Monte Carlo."""
    
    _plots = []
    
    def __init__(self, network):
        self._network = network
        self._algorithm = MetropolisAlgorithm(network)
        
        fig = pyplot.figure()
        image_axis = fig.add_subplot(131, aspect='equal')
        self._plots.append(PlotTracker(network, 'energy', fig.add_subplot(132)))
        self._plots.append(PlotTracker(network, 'polarization', fig.add_subplot(133)))
        
        self.image = AxesImage(image_axis, cmap='Greys',  interpolation='nearest')
        image_axis.set_xlim(0,network.cols - 1)
        image_axis.set_ylim(0,network.rows - 1)
        image_axis.add_image(self.image)
        
        burnAxis = pyplot.axes([0.1, 0.05, 0.8, 0.05])
        self.burnSlider = Slider(burnAxis, 'Burn-off', 0, 1, valinit=0, valfmt=u'%d')
        self.burnoff = 0
        self.burnSlider.on_changed(lambda x: self._update_burnoff(x))
        
        animation.TimedAnimation.__init__(self, fig, blit = True)

    def _update_burnoff(self, x):
        print(x,self._it)
        self.burnoff = int(x * self._it)

    def _draw_frame(self, i):
        self._algorithm.update()

        # Update image
        self.image.set_array(self._network.onGrid)
        self._it = i

        # Update plots
        for plot in self._plots:
            plot.update(i, self.burnoff)

    def new_frame_seq(self):
        i = 0
        while True:
            yield i
            i += 1

    def _init_draw(self):
        self.image.set_array(self._network.onGrid)

def warmup(ising):
    pyplot.close()
    ani = Ising2dPlot(ising)
    pyplot.show()    

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def train(ising):
    rbm1 = BernoulliRBM(n_components=ising.size/4, verbose = False, n_iter = 200, batch_size = 100)
    rbm2 = BernoulliRBM(n_components=ising.size/4/4, verbose = False, n_iter = 200, batch_size = 100)
    rbm3 = BernoulliRBM(n_components=ising.size/4/4/4, verbose = False, n_iter = 200, batch_size = 100)
    dnn = Pipeline(steps=[('rbm1', rbm1),('rbm2', rbm2),('rbm3', rbm3)])
    algo = MetropolisAlgorithm(ising)
    batch = 100
    for i in range(400):
        print('Iteration %d:' % i)
        data = numpy.zeros([batch, ising.size], dtype = bool)
        for j in range(batch):
            algo.update()
            data[j,:] = ising.units.copy()
        dnn.fit(data)
    return dnn

#if __name__ == '__main__':
#    ising2d = Ising2d(40, 40, J = -1./0.408, pbc = True, spins = 1)
#    warmup(ising2d)