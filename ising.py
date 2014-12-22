#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy
from boltzmann import SymmetricHopfieldNetwork
from mcmc import MetropolisAlgorithm
from util import acf

import matplotlib.pyplot as pyplot	
import matplotlib.animation as animation
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

SPINONEHALF = 1
SPINONE = 2
SPINTHREEHALVES = 3

class Ising2d(SymmetricHopfieldNetwork):
    """
    Simple Ising model in 2d.
    """
    def __init__( self, rows, cols, J = -1., spin = SPINONEHALF, pbc = True):
        super( Ising2d, self ).__init__( spin * rows * cols )

        if spin != SPINONEHALF: raise NotImplementedError("Only spin-1/2 is supported")
        if not pbc: raise NotImplementedError("Only PBC supported")

        self._spin = spin
        self._rows = rows
        self._cols = cols

        # For Ising model, biases should be: b = - (# nearest neighbors) J / 2
        size = self.weights.size
        self.biases = numpy.ones( size ) * ( - 2. * J )
        
        for i in range( size ):
            # TODO: implemented PBC and spin > 1/2
            col = i % cols
            row = i / cols
            self.weights[ i, cols * row + (col + 1) % cols ] =  J
            self.weights[ i, cols * row + (col - 1) % cols ] =  J
            self.weights[ i, ( i + cols ) % size ] = J
            self.weights[ i, ( i - cols ) % size ] = J
    
    @property
    def spin(self):
        return self._spin
    
    @property
    def rows(self):
        return self._rows
        
    @property
    def cols(self):
        return self._cols
    
    @property
    def onGrid(self):
        return self.units.reshape(self.rows, self.cols, self.spin).sum(axis=2)

class Ising2dPlot(animation.TimedAnimation):
    """
    GUI to examine Ising2d statistics using Monte Carlo.
    """
    def __init__(self, network, temperature = 0.5):
        self._network = network
        self._algorithm = MetropolisAlgorithm(network, temperature)
        
        fig = pyplot.figure()
        self.axis1 = fig.add_subplot(221, aspect='equal')
        self.axis2 = fig.add_subplot(222)
        self.axis3 = fig.add_subplot(223)
        self.axis4 = fig.add_subplot(224)
        
        self.image = AxesImage(self.axis1, cmap='Greys',  interpolation='nearest')
        self.axis1.set_xlim(0,network.cols - 1)
        self.axis1.set_ylim(0,network.rows - 1)
        self.axis1.add_image(self.image)
        
        self.energy = Line2D([], [], color = 'gray')
        self.mean_energy = Line2D([], [], color = 'blue')
        self.std_energy = Line2D([], [], color = 'red')
        self.axis2.add_line(self.energy)
        self.axis2.add_line(self.mean_energy)
        self.axis2.add_line(self.std_energy)
        self.axis2.set_xlabel('iteration')
        self.axis2.set_ylabel('Energy')
        
        self.autocorrelation = Line2D([], [], color = 'black')
        self.axis3.add_line(self.autocorrelation)
        self.axis3.set_ylabel('Autocorrelation')
        self.axis3.set_xlabel('Lag')
        
        animation.TimedAnimation.__init__(self, fig, blit = True)

    def _draw_frame(self, i):
        self._algorithm.update()
        
        burnin = i/2
        
        # Update image
        self.image.set_array(self._network.onGrid)
        
        # Update energy
        x,ene = self.energy.get_data()
        x.append(i)
        ene.append(self._network.energy / self._network.size)
        self.energy.set_data(x,ene)
        self.axis2.set_xlim(0,i)
        self.axis2.set_ylim(min(ene[burnin:]),max(ene[burnin:]))
        
        # Update autocorrelation
        ac = numpy.array(acf(ene[burnin:]))
        x = range(len(ac))
        self.autocorrelation.set_data(x,ac)
        self.axis3.set_xlim(0,len(ac))
        self.axis3.set_ylim(min(ac),max(ac))
        
        # Smallest ACF = 0 lag
        ac_lag = numpy.argmin(ac > 0.00)
        
        # Histogram of energies using lag
        self.axis4.clear()
        self.axis4.hist(ene[burnin::ac_lag])
        
        # Mean and std of histogram
        m = numpy.mean(ene[burnin::ac_lag])
        s = numpy.std(ene[burnin::ac_lag])
        self.axis4.set_xlim(m - 3 * s, m + 3 * s)
        x,mean_ene = self.mean_energy.get_data()
        mean_ene.append(m)
        x.append(i)
        self.mean_energy.set_data(x,mean_ene)
        x,std_ene = self.std_energy.get_data()
        std_ene.append(s)
        x.append(i)
        self.std_energy.set_data(x,std_ene)

    def new_frame_seq(self):
        i = 0
        while True:
            yield i
            i += 1

    def _init_draw(self):
        self.image.set_array(self._network.onGrid)

if __name__ == '__main__':
    ising2d = Ising2d(50, 50, J = -1.)
    ani = Ising2dPlot(ising2d, temperature = .5)
    pyplot.show()