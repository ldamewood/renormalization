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
        """
        Initialize the network of size V with random binary units and zero bias.
        """
        self.weights = EdgeWeightedGraph(V)
        self.units = numpy.random.random(V) < 0.5
        self.biases = numpy.zeros(V)
    
    @property
    def size(self):
        return len(self.units)
    
    @property
    def polarization(self):
        """
        Average number of activated units.
        """
        return numpy.mean(self.units)
    
    @property
    def gaps(self):
        """
        Energy gaps for each unit to go from True --> False
        """
        return self.biases + self.weights.matrix.dot(self.units)
    
    @property
    def energy(self):
        """
        Total energy of the network.
        """
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
        """
        Generator for visible units.
        """
        for v in range(self._visible): yield v
    
    @property
    def hidden(self):
        """
        Generator for hidden units.
        """
        for h in range(self._hidden): yield self._visible + h

class RestrictedBoltzmannMachine(Bipartite):
    """
    Bipartite class that determines if the partition splits the visible and
    hidden layers of a Boltzmann machine
    """
    
    _isRBM = True
    
    def __init__(self, boltzmannMachine):
        super(RestrictedBoltzmannMachine, self).__init__(boltzmannMachine.weights)
        self._isRBM = True
        color = None
        for v in boltzmannMachine.visible:
            if color == None:
                color = self._color[v]
            elif self._color[v] != color: self._isRBM = False
        color = None
        for h in boltzmannMachine.hidden:
            if color == None:
                color = self._color[h]
            elif self._color[h] != color: self._isRBM = False
    
    @property
    def isRBM(self):
        return self._isRBM
    
    @staticmethod
    def randomRBM(visible, hidden):
        this = BoltzmannMachine(visible, hidden)
        this.units[:] = numpy.random.random(this.size) < 0.5
        this.biases[:] = numpy.random.random(this.size)
        for v in this.visible:
            for h in this.hidden:
                this.weights[v,h] = numpy.random.random()
        return this

def _testRandomHopfieldNecklace():
    network = SymmetricHopfieldNetwork(6)
    network.units[:] = numpy.random.random(6) < 0.5
    network.biases[:] = numpy.random.random(6)
    network.weights[0,1] = numpy.random.random()
    network.weights[1,2] = numpy.random.random()
    network.weights[2,3] = numpy.random.random()
    network.weights[3,4] = numpy.random.random()
    network.weights[4,5] = numpy.random.random()
    network.weights[5,0] = numpy.random.random()
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

if __name__ == '__main__':  
    _testRandomHopfieldNecklace()
    _testRandomRBM()
