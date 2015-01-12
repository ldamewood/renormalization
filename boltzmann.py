#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy

from os import linesep
from graph import EdgeWeightedGraph, Bipartite

class SymmetricHopfieldNetwork(object):
    """
    The SymmetricHopfieldNetwork is a edge weighted graph with binary units
    on the vertecies. The energy of the network is calculated by adding the
    edge weight between two activated units and then adding a bias term to 
    activated units.
    """
    
    weights = None # EdgeWeightedGraph
    units = None
    biases = None
        
    def __init__(self, V):
        self.weights = EdgeWeightedGraph(V)
        self.units = numpy.random.random(V) < 0.5
        self.biases = numpy.zeros(V)
    
    @property
    def size(self):
        return len(self.units)
    
    @property
    def polarization(self):
        """Average number of activated units."""
        return numpy.mean(self.units)
    
    @property
    def gaps(self):
        """Energy gaps for each unit to go from True --> False."""
        return self.biases + self.weights.matrix.dot(self.units)
    
    @property
    def energy(self):
        """Total energy of the network."""
        return numpy.dot(self.units, self.biases + 0.5 * self.weights.matrix.dot(self.units))

    def __str__(self):
        ret = 'Units: ' + self.units.__str__() + linesep
        ret += 'Biases: ' + self.biases.__str__() + linesep
        ret += 'Weights: ' + self.weights.__str__() + linesep
        ret += 'Energy: ' + self.energy.__str__() + linesep
        return ret

class SigmoidBeliefNetwork:
    pass

class BoltzmannMachine(SymmetricHopfieldNetwork):
    """
    The BoltzmannMachine is a Hopfield network with visible and hidden units.
    """
    _visible = 0
    _hidden = 0
    
    def __init__(self, visible, hidden):
        super(BoltzmannMachine, self).__init__(visible + hidden)
        self._visible = visible
        self._hidden = hidden
    
    @property
    def visible(self):
        """Visible units."""
        return range(self._visible)
    
    @property
    def hidden(self):
        """Hidden units."""
        return range(self._visible, self.size)

class RestrictedBoltzmannMachine(Bipartite):
    """
    Bipartite class that determines if the partition splits the visible and
    hidden layers of a Boltzmann machine
    """
    
    _isRBM = True
    _model = None
    
    __RANDOM_RBM_SCALE = 0.01
    
    def __init__(self, boltzmannMachine):
        super(RestrictedBoltzmannMachine, self).__init__(boltzmannMachine.weights)
        self._isRBM = True
        self._checkRBM(boltzmannMachine)

    def _checkRBM(self, model):
        # Check if model is RBM
        color = None
        for v in model.visible:
            if color == None:
                color = self._color[v]
            elif self._color[v] != color: self._isRBM = False
        color = None
        for h in model.hidden:
            if color == None:
                color = self._color[h]
            elif self._color[h] != color: self._isRBM = False
            
    @property
    def isRBM(self):
        return self._isRBM
    
    @property
    def visible_mask(self):
        # Get all units the same color as the first unit
        return self.mask(self._color[0])

    @property
    def hidden_mask(self):
        # Get all units the same color as the last unit
        return self.mask(self._color[-1])
    
    @classmethod
    def randomRBM(cls, visible, hidden):
        """Create a random restricted Boltzmann machine."""
        this = BoltzmannMachine(visible, hidden)
        this.units[:] = numpy.random.random(this.size) < 0.5
        this.biases[:] = numpy.random.normal(scale = cls.__RANDOM_RBM_SCALE, size = this.size)
        for v in this.visible:
            for h in this.hidden:
                this.weights[v,h] = numpy.random.normal(scale = cls.__RANDOM_RBM_SCALE)
        return this

def _testRandomHopfieldNecklace():
    network = SymmetricHopfieldNetwork(6)
    network.units[:] = numpy.random.random(6) < 0.5
    network.biases[:] = numpy.random.normal(size = 6)
    network.weights[0,1] = numpy.random.normal()
    network.weights[1,2] = numpy.random.normal()
    network.weights[2,3] = numpy.random.normal()
    network.weights[3,4] = numpy.random.normal()
    network.weights[4,5] = numpy.random.normal()
    network.weights[5,0] = numpy.random.normal()
    print(network)
    # Check that the energy for switching a unit correctly gives the energyGap
    # for that unit.
    for i in numpy.random.randint(0, 6, 100):
        E = network.energy
        network.units[i] = not network.units[i]
        assert abs(abs(E - network.energy) - abs(network.gaps[i])) < 1e-12

def _testRandomRBM():
    boltzmann = RestrictedBoltzmannMachine.randomRBM(4,5)
    rbm = RestrictedBoltzmannMachine(boltzmann)
    assert rbm.isRBM
    print(boltzmann)
    print(rbm)

def _spin32RBM():
    """
    Example of a RBM that represents a the renormalization of 3 spin-1/2 objects
    into a spin-3/2 object.
    """
    pass
#    import itertools
#
#    # Polynomial fit for the weights of the RBM.
#    x = numpy.zeros([11,32])
#    for i,v in enumerate(itertools.product(range(2),range(2),range(2),range(2),range(2))):
#        print(v)
#        x[:5,i] = v
#        x[5,i] = v[0]*v[3]
#        x[6,i] = v[0]*v[4]
#        x[7,i] = v[1]*v[3]
#        x[8,i] = v[1]*v[4]
#        x[9,i] = v[2]*v[3]
#        x[10,i] = v[2]*v[4]
#    # These values of y give large probabilities for correct matches.
#    y = numpy.array([21.92723856,-23.02585093,-23.02585093,-23.02585093,
#                    -23.02585093,21.92723856,-23.02585093,-23.02585093,
#                    -23.02585093,21.92723856,-23.02585093,-23.02585093,
#                    -23.02585093,21.92723856,-23.02585093,-23.02585093,
#                    -23.02585093,-23.02585093,21.92723856,-23.02585093,
#                    -23.02585093,-23.02585093,21.92723856,-23.02585093,
#                    -23.02585093,-23.02585093,21.92723856,-23.02585093,
#                    -23.02585093,-23.02585093,-23.02585093,21.92723856])
#    solver1 = NormalSVD(reglambda = 100.)
#    #solver1 = NormalSVD()
#    ans = solver1.getWeights(x,y)
#    print(ans)
#    
#    #energies = [ -3./2, -1./2, -1./2, -1./2, 1./2, 1./2, 1./2, 3./2 ]
#    #probs = numpy.exp(-numpy.array(energies))
#    
#    # Manually setup a RBM that gives the correct renormalization.
#    bm = BoltzmannMachine(3,2)
#    bm.biases = [-8.70315643,  -8.70315643,  -8.70315643, -19.0956329 ,    -7.85848425]
#    bm.weights[0,3] = 43.26091464
#    bm.weights[0,4] = -24.16506737
#    bm.weights[1,3] = -1.68992731
#    bm.weights[1,4] = 20.78577458
#    bm.weights[2,3] = -1.68992731
#    bm.weights[2,4] = 20.78577458
#    rbm = RestrictedBoltzmannMachine(bm)
#    solver = SigmoidUpdate(bm)
#    for v in itertools.product(range(2),range(2)):
#        # Loop over the hidden layer states and print the total spin using the
#        # visible layer
#        bm.units[rbm.hidden] = v
#        solver._updateOnMask(rbm.visible)
#        print('Hidden layer representation: (%d %d)' % (v[0], v[1]))
#        print('Total spin: %f' % (1. * sum(bm.units[rbm.visible]) - 3./2))

    # TODO: train the RBM from a random state

if __name__ == '__main__':
    _testRandomHopfieldNecklace()
    _testRandomRBM()
    #_spin32RBM()