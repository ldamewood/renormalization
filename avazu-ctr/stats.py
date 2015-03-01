from __future__ import print_function

from math import sqrt

try:
    from numpy import mean, var
except ImportError:
    def mean(lst):
        return 1. * sum(lst) / len(lst)
    def var(lst, ddof = 0):
        m = mean(lst)
        return sum([(x - m)**2. for x in lst])/(len(lst) - ddof)

try:
    from scipy.stats import skew, kurtosis, describe
except ImportError:
    def skew(lst):
        m = mean(lst)
        return sum([(x - m)**3. for x in lst])/len(lst)/sqrt(var(lst))**3.
    def kurtosis(lst):
        m = mean(lst)
        return sum([(x - m)**4. for x in lst])/len(lst)/var(lst, ddof=0)**2. - 3.
    def describe(lst):
        return len(lst),(min(lst),max(lst)),mean(lst),var(lst, ddof=1),skew(lst),kurtosis(lst)

class RunningStatistics(object):
    """
    Welford-Knuth Algorithm for calculating running statistics.
    http://www.johndcook.com/blog/skewness_kurtosis/
    
    An object that keeps track of the sample mean, variance, skewness and
    kurtosis of a stream of numbers (online/stocastic)
    """
    _n = 0
    _stats = None
    _min = float('inf')
    _max = -float('inf')
    
    def __init__(self, initialList = None):
        self._stats = 5*[0.]
        if initialList != None and hasattr(initialList, '__iter__'):
            n = len(initialList)
            self._n = n
            self._min = min(initialList)
            self._max = max(initialList)
            self._stats[0] = mean(initialList)
            self._stats[1] = var(initialList) * n
            self._stats[2] = self._stats[1]**1.5 * skew(initialList) / sqrt(n)
            self._stats[3] = (kurtosis(initialList) + 3.) * self._stats[1]**2 / n
            # Related to the Laplace distribution
            self._stats[4] = mean(initialList)

    def _batch_push(self, lst):
        other = RunningStatistics(lst)
        self._add(other)
    
    def _online_push(self, x):
        n1 = self._n
        self._n += 1
        n = self._n
        
        delta = x - self._stats[0]
        delta_n = delta / self._n
        delta_n2 = delta_n ** 2
        term1 = delta * delta_n * n1
        self._stats[0] += delta_n
        self._stats[3] += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * self._stats[1] - 4 * delta_n * self._stats[2]
        self._stats[2] += term1 * delta_n * (n - 2) - 3 * delta_n * self._stats[1]
        self._stats[1] += term1
        self._min = min(self._min, x)
        self._max = max(self._max, x)
    
    def _add(self, other):
        """
        Chan's algorithm for combining datasets. This method is numerically
        unstable for variance, skew and kurtosis when the datasets are large and
        about the same size.
        """
        if other.size == 0:
            return

        if self.size == 0:
            self._n = other._n
            self._min = other.min
            self._max = other.max
            self._stats = other._stats
            return

        n = self.size + other.size
        # Expressions using delta are numerically unstable.
        delta = other.mean - self.mean
        delta2 = delta**2
        delta3 = delta*delta2
        delta4 = delta2**2
        
        newStats = 5*[0.]
        newStats[0] = (self.size * self._stats[0] + other.size * other._stats[0]) / n

        newStats[1] = self._stats[1] + other._stats[1] + delta2 * self.size * other.size / n
        
        newStats[2] = self._stats[2] + other._stats[2]
        newStats[2] += delta3 * self.size * other.size * (self.size - other.size) / n**2
        newStats[2] += 3. * delta * (self.size * other._stats[1] - other.size * self._stats[1]) / n
        
        newStats[3] = self._stats[3] + other._stats[3]
        newStats[3] += delta4 * self.size * other.size * (self.size ** 2 - self.size * other.size + other.size ** 2) / n**3
        newStats[3] += 6. * delta2 * (self.size**2*other._stats[1] + other.size**2*self._stats[1]) / n**2
        newStats[3] += 4. * delta * (self.size * other._stats[2] - other.size * self._stats[2]) / n
        
        self._n = n
        self._min = min(self.min, other.min)
        self._max = max(self.max, other.max)
        self._stats = newStats

    def push(self, x):
        if hasattr(x, '_add'):
            self._add(x)
        elif hasattr(x, '__iter__'):
            self._batch_push(x)
        else:
            self._online_push(x)
    
    def clear(self):
        self._n = 0
        self._mean = 0
        self._var = 0
    
    @property
    def mean(self):
        return self._stats[0]
    
    def var(self, ddof = 0):
        return self._stats[1] / (self._n - ddof)

    @property
    def variance(self):
        return self.var(1)    
    
    @property
    def std(self):
        return sqrt(self.var())

    @property
    def skewness(self):
        return sqrt(1.*self._n) * self._stats[2] / self._stats[1] ** 1.5

    @property
    def kurtosis(self):
        return 1.*self._n*self._stats[3] / (self._stats[1]**2) - 3.

    @property
    def min(self):
        return self._min
    
    @property
    def max(self):
        return self._max
    
    @property
    def describe(self):
        return self.size,(self.min, self.max), self.mean, self.variance, self.skewness, self.kurtosis
    
    @property
    def size(self):
        return self._n

if __name__ == '__main__':
    """
    Unit tests.
    """
    from numpy import hstack
    from numpy.random import random
    
    def _compareDescriptions(A, B):
        err = 1.e-10
        assert abs(A[0] - B[0]) < err
        assert abs(A[1][0] - B[1][0]) < err
        assert abs(A[1][1] - B[1][1]) < err
        assert abs(A[2] - B[2]) < err
        assert abs(A[3] - B[3]) < err
        assert abs(A[4] - B[4]) < err
        assert abs(A[5] - B[5]) < err
    
    def _testOnline():
        A = random(10000)
        run = RunningStatistics()
        for x in A: run.push(x)
        _compareDescriptions(run.describe, describe(A))
    
    def _compareOnlineAndBatch():
        A = random(10)
        run1 = RunningStatistics()
        for x in A: run1.push(x)
        run2 = RunningStatistics(A)
        _compareDescriptions(run1.describe, run2.describe)
        run3 = RunningStatistics()
        run3.push(A)
        _compareDescriptions(run3.describe, describe(A))
    
    def _testBatch():
        A = random(100)
        run = RunningStatistics()
        run.push(A)
        _compareDescriptions(run.describe, describe(A))
    
    def _testCombine():
        A = random(10000)
        B = 10 * random(1000)
        C = hstack([A,B])
        run3 = RunningStatistics(A)
        run3.push(B)
        _compareDescriptions(run3.describe, describe(C))
    
    _testOnline()
    _compareOnlineAndBatch()
    _testBatch()
    _testCombine()