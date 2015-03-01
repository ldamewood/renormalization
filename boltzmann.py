#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy

from os import linesep
from graph import UndirectedGraph, Bipartite

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
        self.weights = UndirectedGraph(V)
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

class RestrictedBoltzmannMachine(BoltzmannMachine):
    __RANDOM_RBM_SCALE = 0.01
    
    def __init__(self, visible, hidden):
        super(RestrictedBoltzmannMachine, self).__init__(visible, hidden)
        self._isRBM = None
    
    @property
    def isRBM(self):
        if self._isRBM == None:
            bipartite = Bipartite(self.weights)
            self._isRBM = True
            # Check if model is RBM
            color = None
            for v in self.visible:
                if color == None:
                    color = bipartite._color[v]
                elif bipartite._color[v] != color: self._isRBM = False
            color = None
            for h in self.hidden:
                if color == None:
                    color = bipartite._color[h]
                elif bipartite._color[h] != color: self._isRBM = False
        return self._isRBM
    
    
    @classmethod
    def randomRBM(cls, visible, hidden):
        """Create a random restricted Boltzmann machine."""
        this = RestrictedBoltzmannMachine(visible, hidden)
        this.units[:] = numpy.random.random(this.size) < 0.5
        this.biases[:] = numpy.random.normal(scale = cls.__RANDOM_RBM_SCALE, size = this.size)
        for v in this.visible:
            for h in this.hidden:
                this.weights[v,h] = numpy.random.normal(scale = cls.__RANDOM_RBM_SCALE)
        assert this.isRBM
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
    rbm = RestrictedBoltzmannMachine.randomRBM(4,5)
    assert rbm.isRBM
    print(rbm)

if __name__ == '__main__':
    _testRandomHopfieldNecklace()
    _testRandomRBM()