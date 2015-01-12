from mcmc import SigmoidUpdate
from boltzmann import RestrictedBoltzmannMachine

import numpy
from util import sigmoid

class ContrastiveDivergence(object):
    _model = None
    _rbm = None
    
    def __init__(self, model):
        self._model = model
        self._rbm = RestrictedBoltzmannMachine(self._model)
        if not self._rbm.isRBM:
            raise NotImplementedError()
    
    def learn(self, data, num = 1, rate = 0.01):
        """
        
        """
        if num <= 0: raise ValueError('Need at least one reconstruction')
        nVisible = len(self._model.visible)
        nHidden = len(self._model.hidden)
        self._model.units[self._model.visible] = data
        prob_hidden = sigmoid(self._model.gaps[self._model.hidden])
        self._model.units[self._model.hidden] = prob_hidden > numpy.random.random(nHidden)
        vh_data = self._model.units.copy()
        for i in range(num):
            prob_visible = sigmoid(self._model.gaps[self._model.visible])
            self._model.units[self._model.visible] = prob_visible > numpy.random.random(nVisible)
            prob_hidden = sigmoid(self._model.gaps[self._model.hidden])
            self._model.units[self._model.hidden] = prob_hidden > numpy.random.random(nHidden)
        vh_model = self._model.units.copy()
        for v in self._model.visible:
            for h in self._model.hidden:
                dw = vh_data[v] * vh_data[h] - vh_model[v] * vh_model[h]
                self._model.weights[v,h] = self._model.weights[v,h] + rate * dw
        for v in self._model.visible:
            da = vh_data[v] - vh_data[v]
            self._model.biases[v] = self._model.biases[v] + rate * da
        for h in self._model.hidden:
            da = vh_data[h] - vh_data[h]
            self._model.biases[h] = self._model.biases[h] + rate * da

if __name__ == '__main__':
    #ndata = 1000
    #nvis = 10
    #nhid = 2
    #data = numpy.random.random([ndata,nvis]) < 0.5 # Fake model data
    #bm = RestrictedBoltzmannMachine.randomRBM(nvis,nhid) # Model to train
    #rbm = RestrictedBoltzmannMachine(bm)
    #updater = SigmoidUpdate(bm)
    #cd = ContrastiveDivergence(bm)
    #rate = 0.01
    #for i in range(ndata):
    #    cd.learn(data[i,:], num = 10)
    import numpy as np
    from sklearn.neural_network import BernoulliRBM
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    model = BernoulliRBM(n_components=2, verbose = 10, n_iter = 100)
    f = model.fit(X)
    
    #bm = RestrictedBoltzmannMachine.randomRBM(3,2) # Model to train
    #cd = ContrastiveDivergence(bm)
    #cd.learn(X, num = 100, rate = 0.1)