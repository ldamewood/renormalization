#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy
from util import sigmoid

from abc import ABCMeta, abstractmethod
from graph import Bipartite

class MCMethod(object):
    """
    Interface for Monte Carlo-like methods. Graphs must be bipartite (they usually are)
    """
    __metaclass__ = ABCMeta
    _network = None
    _evnMask = None
    _oddMask = None
    #_bipartite = None
    
    def __init__(self, network):
        self._network = network
        bipartite = Bipartite(network.weights)
        if not bipartite.isBipartite:
            raise NotImplementedError("Graph must be bipartite")
        self._evnMask = bipartite.mask(True)
        self._oddMask = bipartite.mask(False)
    
    def update(self):
        self._updateOnMask(self._evnMask)
        self._updateOnMask(self._oddMask)
    
    @abstractmethod
    def _updateOnMask(self, mask):
        raise NotImplementedError("Please Implement this method")
        
class BinaryThreshold(MCMethod):
    """
    Finds the local minima.
    """ 
    def _updateOnMask(self, mask):
        self._network.units[mask] = self._network.gaps[mask] < 0

class SigmoidUpdate(MCMethod):
    """
    Used in RBMs.
    """
    def _updateOnMask(self, mask):
        self._network.units[mask] = sigmoid(self._network.gaps[mask]) > numpy.random.random(len(mask))

class MetropolisAlgorithm(MCMethod):
    """
    Metropolis-Hastings algorithm.
    """
    def __init__(self, network, temperature = 1.):
        super(MetropolisAlgorithm, self).__init__(network)
        self.temperature = temperature
    
    def _updateOnMask(self, mask):
        dE = self._network.gaps[mask] * ( 1 - 2 * self._network.units[mask] )
        r = numpy.random.random(len(dE))
        f = numpy.minimum(1, numpy.exp(-dE/self.temperature)) > r
        self._network.units[mask] = numpy.logical_xor(f, self._network.units[mask])

class GibbsSampler(MCMethod):
    def __init__(self):
        raise NotImplementedError()
    
    def _updateOnMask(self, mask):
        raise NotImplementedError()