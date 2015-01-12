#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy

def rand_index(l):
    """
    Return an index of the list with the probability given in the list.
    
    Example: prob_index([0.5,0.25,0.25]) should return 0 50% of the time, 1 25% 
    of the time and 2 25% of the time.
    """
    r = numpy.random.uniform(0., sum(l))
    s = l[0]
    for i,p in enumerate(l):
        if r < s: return i
        s += p
    
    # Should only reach this point due to floating-point errors.
    return len(l) - 1

def acf(x):
    """
    Autocorrelation function. (Not verified)
    
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    x = numpy.array(x)
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = numpy.correlate(x, x, mode = 'full')[-n:]
    assert numpy.allclose(r, numpy.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(numpy.arange(n, 0, -1)))
    return result

def partial_sums(it):
    """
    Partial sum generator: p[i] = sum(it[:i])
    """
    p = 0
    for i in it:
        p += i
        yield p

def sigmoid(x):
    """
    Sigmoid function.
    """
    return 1. / ( 1. + numpy.exp( -1. * x ) )

if __name__ == '__main__':
    # Units tests (especially for acf function)
    pass