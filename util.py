#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy

def prob_index(l):
    """
    Return an index of the list with the probability given in the list.
    
    Example: prob_index([0.5,0.25,0.5]) should return 0 50% of the time, 1 25% 
    of the timem and 2 25% of the time.
    """
    r = numpy.random.uniform(0., sum(l))
    print(r)
    s = l[0]
    for i,p in enumerate(l):
        if r < s: return i
        s += p
    
    # Should only reach this point due to floating-point errors.
    return -1 

def acf(series):
    """
    Autocorrelation function. (Not verified)
    """
    n = len(series)
    data = numpy.asarray(series)
    mean = numpy.mean(data)
    c0 = numpy.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)

    x = numpy.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs

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