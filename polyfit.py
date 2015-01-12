#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy
import numpy.random
import matplotlib.pyplot as pyplot

class LinearRegression:
    """
    Abstract class for linear regression with regularization.
    """
    __metaclass__ = ABCMeta
    
    reglambda = None
    
    @abstractmethod
    def getWeights(self, x, y):
        """
        Abstract method to calculate the weights of the linear system.
        """
        pass

class NormalSVD(LinearRegression):
    """
    Linear regression using the normal equation by SVD with regularization.
    """
    def __init__(self, reglambda = 0.):
        self.reglambda = reglambda

    def getWeights(self, x, y):
        U,Sig,V = numpy.linalg.svd(x.T)
        # Weights: w = A' . A . y = V' . D . U . y
        # D = s / ( s^2 + l^2 )
        D = numpy.zeros(x.shape)
        D[:len(Sig),:len(Sig)] += numpy.diag(Sig / (Sig**2 + self.reglambda**2))
        weights = numpy.dot(U.T,y)
        weights = numpy.dot(D, weights)
        weights = numpy.dot(V.T, weights)
        return weights

class BatchGradientDescent(LinearRegression):
    """
    Linear regression using batch gradient descent with regularization and an
    adaptive learning rate.
    """
    def __init__(self, reglambda = 0., gdmax = 100000, rate = 0.01, conv = 1e-10):
        self.reglambda = reglambda
        self.gdmax = gdmax
        self.rate = rate
        self.conv = conv
    
    def getWeights(self, x, y):
        # Initialize random weights:
        weights = numpy.random.normal(size = x.shape[0])
        costs = numpy.zeros(self.gdmax)
        for i in range(self.gdmax):
            costs[i] = sum((numpy.dot(weights, x) - y)**2) + self.reglambda * sum(weights**2)
            costs[i] /= 2 * len(ydata)
            dcost = numpy.dot(numpy.dot(weights, x) - y,x.T) + self.reglambda * weights
            weights -= self.rate * dcost / len(ydata)
            if i > 1:
                if abs(costs[i] - costs[i-1])/costs[i] < self.conv:
                    break;
                if costs[i] - costs[i-1] > 1e-10: self.rate *= 0.5
                if costs[i] - costs[i-1] < 0: self.rate *= 1.1
        return weights

class OnlineGradientDescent(LinearRegression):
    """
    Linear regression using online (stocastic) gradient descent with
    regularization and an adaptive learning rate.
    """
    def __init__(self, reglambda = 0., gdmax = 1000, rate = 0.0001, conv = 1e-10):
        self.reglambda = reglambda
        self.gdmax = gdmax
        self.rate = rate
        self.conv = conv
        
    def getWeights(self, x, y):
        # Initialize random weights:
        weights = numpy.random.normal(size = x.shape[0])
        for i in range(self.gdmax):
            costs = numpy.zeros(x.shape[1])
            for j in range(x.shape[1]):
                costs[j] = (numpy.dot(weights, x[:,j]) - y[j])**2 + self.reglambda * sum(weights**2)
                costs[j] /= 2 * x.shape[1]
                dcost = numpy.dot(numpy.dot(weights, x[:,j]) - y[j],x[:,j]) + self.reglambda * weights
                weights -= self.rate * dcost / x.shape[1]
                if j > 1:
                    if costs[j] - costs[j-1] > 1e-10: self.rate *= 0.01
                    if costs[j] - costs[j-1] < 0: self.rate *= 1.1
                if i%100 == 0 : print(self.rate)
        return weights

class EmpericalBayes(LinearRegression):
    """
    Emperical Bayes update of the regularization parameter.
    """
    def __init__(self, linearMethod, reglambda = 0., regmax = 100, regconv = 1e-2):
        if reglambda < 0:
            raise ValueError("Regularization parameter must be > 0")
            
        if regmax < 1:
            raise ValueError("Must perform at least first regularization step")
        
        if regconv < 0: regconv = -regconv
        
        self.linearMethod = linearMethod
        self.reglambda = reglambda
        self.regmax = regmax
        self.regconv = regconv
        
    def getWeights(self, x, y):
        reglambdaold = self.reglambda
        for i in range(self.regmax):

            # Calculate weights
            self.linearMethod.reglambda = reglambdaold
            weights = self.linearMethod.getWeights(x, y)
            
            # Update the regularization factor
            sigmaD = numpy.std(numpy.dot(weights, x) - y)
            sigmaW = numpy.std(weights)
            print(sigmaD, sigmaW)
            
            # Convergence criteria
            if abs(reglambdaold - self.reglambda) < self.regconv: break
            reglambdaold = self.reglambda
        return weights

def _xpoly(x, deg = 2):
    """
    Helper function to flatten x and return array of
    [ x**deg, x**(deg - 1), ..., x**0 ]
    """
    x = numpy.array([x**i for i in range(1,deg+1)[::-1]])
    x = x.reshape([numpy.prod(x.shape[:-1]),x.shape[-1]])
    x = numpy.append(x, [numpy.ones(x.shape[1:])],axis=0)
    return x

def polyfit(xdata, ydata, deg = 2, linearMethod = NormalSVD()):
    """
    Fit a polynomial using LinearRegression model.
    """
    if deg < 1:
        raise ValueError("Polynomial degree must be > 1")
    
    return linearMethod.getWeights(_xpoly(xdata, deg), ydata)

if __name__ == '__main__':
    deg = 2
    mpoly = numpy.poly1d([0.22770148,  0.46653194,  0.52113885])
    xdata = numpy.array([  6.19775703,   8.95909588,   7.41216435,  -5.70706652,  10.22902444])
    ydata = numpy.array([ 10.26147673,  31.841685  ,  15.74792391,  -1.65491076,  30.3639627 ])

    solver1 = EmpericalBayes(NormalSVD())
    epoly = numpy.poly1d(numpy.polyfit(xdata,ydata,3))
    npoly = numpy.poly1d(solver1.getWeights(_xpoly(xdata, deg = 3), ydata))
    #print(solver2.getWeights(_xpoly(xdata), ydata))
    #print(solver3.getWeights(_xpoly(xdata), ydata))
    
    xr = numpy.linspace(xdata.min(),xdata.max(),100)
    pyplot.plot(xdata, ydata, '.')
    pyplot.plot(xr, epoly(xr))
    pyplot.plot(xr, mpoly(xr))
    pyplot.plot(xr, npoly(xr))
    pyplot.legend(['Data', 'numpy fit', 'model', 'Normal eq.'])
    pyplot.show()