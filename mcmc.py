#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy
from util import sigmoid

from abc import ABCMeta, abstractmethod
from graph import Bipartite
from collections import deque

class MCMethod(object):
    """Interface for Monte Carlo-like methods."""
    __metaclass__ = ABCMeta
    _network = None
    _evnMask = None
    _oddMask = None
    #_bipartite = None
    
    def __init__(self, network):
        self._network = network
        bipartite = Bipartite(network.weights)
        if not bipartite.isBipartite:
            raise NotImplementedError("Network must be bipartite")
        self._evnMask = bipartite.mask(True)
        self._oddMask = bipartite.mask(False)
    
    def update(self):
        self._updateOnMask(self._evnMask)
        self._updateOnMask(self._oddMask)
    
    @abstractmethod
    def _updateOnMask(self, mask):
        raise NotImplementedError("Please Implement this method")
        
class BinaryThreshold(MCMethod):
    """Finds the local minima.""" 
    def _updateOnMask(self, mask):
        self._network.units[mask] = self._network.gaps[mask] < 0

class SigmoidUpdate(MCMethod):
    """Used in RBMs."""    
    def _updateOnMask(self, mask):
        self._network.units[mask] = sigmoid(self._network.gaps[mask]) > numpy.random.random(len(mask))

class MetropolisAlgorithm(MCMethod):
    """Metropolis-Hastings algorithm."""    
    def _updateOnMask(self, mask):
        # Energy change due to flipping selected units.
        dE = self._network.gaps[mask] * ( 1. - 2. * self._network.units[mask] )
        # Update rule for Metrolopis-Hastings algorithm
        select = numpy.minimum(1, numpy.exp(-dE)) > numpy.random.random(len(dE))
        # XOR will flip the units where select == True
        self._network.units[mask] = numpy.logical_xor(select, self._network.units[mask])

class BinaryGibbsStep(MCMethod):
    def _updateOnMask(self, mask):
        sig = 1./(1 + numpy.exp(-self._network.gaps[mask]))
        select = sig > 0.5
        self._network.units[mask] = numpy.logical_xor(select, self._network.units[mask])

class WolffClusterAlgorithm(MCMethod):
    def __init__(self, network):
        super(WolffClusterAlgorithm, self).__init__(network)
        #raise NotImplementedError()
        
        # BFS is not easily parallel
        # Use union-find algorithms somehow?
        # Maybe this:
        # 1) union EVERY neighbor spin together with probability p in parallel
        # 2) select random site, create mask using find algorithm in parallel
        # 3) flip sites in parallel
        # Worst case when clusters are small (high-T)
    
    def update(self):
        # This is a terrible way to find J
        J = abs(self._network.weights.matrix.min())
        p = 1 - numpy.exp(2 * J)
        boundary = deque()
        marked = self._network.size * [False]

        site = numpy.random.randint(0, self._network.size - 1)        
        boundary.append(site)
        
        while len(boundary) > 0:
            site = boundary.popleft()
            marked[site] = True
            
            for neighbor in self._network.weights.adj(site):
                if self._network.units[neighbor] == self._network.units[site] and \
                    not marked[neighbor] and numpy.random.random() < p:
                    boundary.append(neighbor)
          
        mask = numpy.where(marked)
        self._network.units[mask] = numpy.logical_not(self._network.units[mask])
    
    def _updateOnMask(self, mask):
        raise NotImplementedError()