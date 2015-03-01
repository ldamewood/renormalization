from __future__ import print_function, division
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy
import scipy.sparse

class Regression:
    """
    Abstract interface provides train and predict methods and weights property.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, x, y):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass

class LinearRegression(Regression):
    """
    Abstract class provides linear regression
    """
    def predict(self, x):
        x = numpy.vstack((numpy.ones(len(x)),numpy.array(x).T))
        return self._weights.dot(x)

class LogisticRegression(Regression):
    """
    Abstract class provides logistic(logit) regression
    """
    def predict(self, x):
        x = numpy.vstack((numpy.ones(len(x)),numpy.array(x).T))
        z = self._weights.dot(x)
        return 1./(1.+numpy.exp(-z))

class LinearSVD(LinearRegression):
    """
    Simple full batch linear regression with L2 regularization.
    """
    _weights = None

    def __init__(self, L2 = 0.):
        self.L2 = L2

    def train(self, x, y):
        L2 = self.L2
        x = numpy.array(x)
        y = numpy.array(y)
        U,Sig,V = numpy.linalg.svd(x.T)
	D = numpy.zeros(x.shape)
        D[:len(Sig),:len(Sig)] += numpy.diag(Sig / (Sig**2 + L2**2))
        weights = numpy.dot(U.T,y)
        weights = numpy.dot(D, weights)
        weights = numpy.dot(V.T, weights)
        self._weights = weights

class LogisticGradientDescent(LogisticRegression):
    
    def __init__(self, batch = None, sparsity = 0):
        self._sparsity = sparsity
        self._weights = None
        self._batchsize = batch
    
    def train(self, x, y):

        # number of data inputs
        m = x.shape[0]
        # number of features
        n = x.shape[1]
        
        if self._weights is None:
            if self._sparsity > 0:
                self._weights = scipy.sparse.lil_matrix((1,self._sparsity))
            else:
                self._weights = numpy.random.random(n+1)
            
        bs = self._batchsize
        if bs is None: bs = m
        i = 0
        while i < m:
            xbatch = x[i:min(i+bs,m)]
            ybatch = y[i:min(i+bs,m)]
            # Probability
            p = self.predict(xbatch)
            # Gradient
            g = numpy.dot(p - ybatch, xbatch) / bs # Add regularization
            i += bs        
    
model = LogisticGradientDescent(batch = 5)
x = numpy.random.random([15,2])
y = numpy.random.random(15)
model.train(x, y)
print(model.predict([[4,2]]))