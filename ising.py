#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy
from boltzmann import SymmetricHopfieldNetwork, RestrictedBoltzmannMachine
from mcmc import MetropolisAlgorithm
from util import acf, sigmoid
import sys
import itertools

import matplotlib.pyplot as pyplot	
import matplotlib.animation as animation
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

class Ising2d(SymmetricHopfieldNetwork):
    """
    Simple Ising model in 2d with spin-1/2 and spin-3/2.
    
    The Hamiltonian is
    \begin{align}
    H &= \sum_{\langle i,j\rangle} J_{ij} s_i s_j
    \end{align}
    where $s_i$ are z-component spin operators and $\langle\cdot\rangle$ denotes
    nearest-neighbor pairs.
    
    For spin-1/2, we can convert the spins (1/2, -1/2) to binary units using the
    transform $s_i = \sigma_i - 1/2$. Plugging this into the Hamiltonain, and 
    focusing on terms with $\sigma_i$, we get
    \begin{align}
    H_i &= -\frac{NJ}{2}\sigma_i + J\sigma_i\sum_{j}\sigma_j
    \end{align}
    where $N$ is the number of nearest neighbors to $i$. For the square lattice,
    $N=4$, so the biases on the units is $-2J$ and the weights are $J$.
    
    For spin-3/2 models, the mapping requires two unconnected binary units to
    participate. The conversion is $s_i = \sigma_{i1} + 2\sigma_{i2} - 3/2$. 
    Using the same process as above, the weights can be found.
    """
    
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
        """ This is not quite right! """
        if self._spins == 2:
            return numpy.dot(ising2d.units.reshape(self._rows, self._cols, self._spins),numpy.array([1,2]))
        else:
            return ising2d.units.reshape(self._rows, self._cols)


class BilayerIsing2d(SymmetricHopfieldNetwork):
    """
    2-layer Ising model (quasi-2d) with spin-1/2
    """
    def __init__(self, rows, cols, J = -1., alpha = 1., spins = 1, pbc = True):
        super( BilayerIsing2d, self ).__init__( spins * rows * cols )


        self._spins = spins
        self._rows = rows
        self._cols = cols
        self._pbc = pbc
        
        self._setup(J, rows, cols, spins)

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

class ConvolutionalNetwork2D(RestrictedBoltzmannMachine):
    def __init__(self, rows = 16, cols = 16):
        print(rows*cols)
        assert (rows*cols)%4 == 0
        super(ConvolutionalNetwork2D, self).__init__(rows * cols, rows * cols // 4)
        for i,j in itertools.product(range(rows // 2), range(cols // 2)):
            self.weights[(i+1) * rows + j    ,i * rows + j] = numpy.random.normal(scale = 0.01)
            self.weights[(i+1) * rows + j + 1,i * rows + j] = numpy.random.normal(scale = 0.01)
            self.weights[ i    * rows + j    ,i * rows + j] = numpy.random.normal(scale = 0.01)
            self.weights[ i    * rows + j + 1,i * rows + j] = numpy.random.normal(scale = 0.01)

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

#def train(ising, size = 100000):
#    rbm1 = BernoulliRBM(n_components=ising.size/4, verbose = True, n_iter = 50)
#    rbm2 = BernoulliRBM(n_components=ising.size/4/4, verbose = True, n_iter = 50)
#    rbm3 = BernoulliRBM(n_components=ising.size/4/4/4, verbose = True, n_iter = 50)
#    dnn = Pipeline(steps=[('rbm1', rbm1),('rbm2', rbm2),('rbm3', rbm3)])
#    algo = MetropolisAlgorithm(ising)
#
#    data = numpy.zeros([size, ising.size], dtype = bool)
#    for j in range(size):
#        algo.update()
#        algo.update()
#        data[j,:] = ising.units.copy()
#    dnn.fit(data)
#    return dnn

def train(x1, rbm, learning_rate = 0.1):
    rbm.units[rbm.visible] = x1

    mask = rbm.hidden
    Qh1 = sigmoid(rbm.gaps[mask])
    h1 = Qh1 > numpy.random.random(len(Qh1))
    rbm.units[mask] = h1.copy()
    
    mask = rbm.visible
    Px2 = sigmoid(rbm.gaps[mask])
    x2 = Px2 > numpy.random.random(len(Px2))
    rbm.units[mask] = x2.copy()
    
    mask = rbm.hidden
    Qh2 = sigmoid(rbm.gaps[mask])
    h2 = Qh2 > numpy.random.random(len(Qh2))
    rbm.units[mask] = h2.copy()

    adj = (numpy.outer(h1,x1) - numpy.outer(Qh2,x2))
    for i,wi in enumerate(rbm.visible):
        for j,wj in enumerate(rbm.hidden):
            if rbm.weights[wi,wj] == None: continue
            rbm.weights[wi,wj] = rbm.weights[wi,wj] + learning_rate * adj[j,i]
    rbm.biases[rbm.visible] += learning_rate * (x1 - x2)
    rbm.biases[rbm.hidden]  += learning_rate * (h1 - Qh2)
    print(numpy.linalg.norm(x2-x1,1))
    print(numpy.linalg.norm(h1-Qh2,1))
    print(numpy.linalg.norm(adj,1))

if __name__ == '__main__':
    ising2d = Ising2d(40, 40, J = -1./2.408, pbc = True, spins = 2)
    algo = MetropolisAlgorithm(ising2d)
    cnn1 = ConvolutionalNetwork2D(rows = 20, cols = 20)
    cnn2 = ConvolutionalNetwork2D(rows = 10, cols = 10)
    warmup(ising2d)