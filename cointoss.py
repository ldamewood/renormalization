#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy
import numpy.random
import matplotlib.pyplot as pyplot
from matplotlib.widgets import Slider, Button
import operator

class CoinToss:
    # Number of points in prob distribution
    _ndist = 101
    
    # prior probability
    prior = numpy.array([])
    
    # posterior probability
    posterior = numpy.array([])
    
    # conditional probability
    conditional = numpy.array([])

    def __init__(self, tosses, prior = None):
        self.tosses = tosses
        if prior == None:
            prior = numpy.ones(self._ndist) / self._ndist
        self.prior = prior

    def doTosses(self):
        conditional = numpy.linspace(0,1,self._ndist)
        
        # Initial prior probability distribution (uniform)
        posterior = numpy.zeros([self.tosses + 1,self._ndist])
        posterior[0,:] = self.prior / sum(self.prior)
        
        # Collect data
        tosses = numpy.random.randint(0,2,self.tosses)
        
        for i in range(self.tosses):
            # Bayes theorem: p(W|D) = p(W) * p(D|W) / p(D)
            if tosses[i] == 0: posterior[i+1,:] = posterior[i,:] * (1-conditional)
            else:              posterior[i+1,:] = posterior[i,:] * conditional
            posterior[i+1,:] /= posterior[i+1,:].sum()
    
        self.conditional = conditional
        self.posterior = posterior
        
class GUI:
    
    def __init__(self, coinToss):
        self._coinToss = coinToss
        maxTosses = coinToss.tosses
        
        axcolor = 'lightgoldenrodyellow'
        
        self.figure = pyplot.figure()
        self.mainAxis = pyplot.axes([0.05, 0.2, 0.9, 0.75])
        
        # Slider for number of tosses
        tossAxis = pyplot.axes([0.1, 0.05, 0.8, 0.05])
        self.tossSlider = Slider(tossAxis, 'Tosses', 0., 1.*maxTosses, valinit=0, valfmt=u'%d')
        self.tossSlider.on_changed(lambda x: self.draw(x))

        # Reset button        
        resetAxis = pyplot.axes([0.8, 0.85, 0.1, 0.05])
        self.resetButton = Button(resetAxis, 'Re-toss', color=axcolor, hovercolor='0.975')
        self.resetButton.on_clicked(self.retoss)
        
        # Key press events
        self.figure.canvas.mpl_connect('key_press_event', lambda x: self.press(x))
        
        self._coinToss.doTosses()

    def retoss(self, event):
        self._coinToss.doTosses()
        self.draw(self.tossSlider.val)

    def press(self, event):
        if event.key == u'left' and self.tossSlider.val > self.tossSlider.valmin:
            self.tossSlider.set_val(self.tossSlider.val - 1)
        if event.key == u'right' and self.tossSlider.val < self.tossSlider.valmax:
            self.tossSlider.set_val(self.tossSlider.val + 1)
        if event.key == u'r':
            self.retoss(event)

    def draw(self, x = 0):
        c = self._coinToss
        m = max(enumerate(c.posterior[int(x),:]), key=operator.itemgetter(1))
        self.mainAxis.clear()
        self.mainAxis.plot(c.conditional,c.posterior[int(x),:], lw=2, color='red')
        self.mainAxis.vlines(c.conditional[m[0]],0,c.posterior.max())
        self.figure.canvas.draw()

if __name__ == '__main__':
    coinToss = CoinToss(100)
    gui = GUI(coinToss)
    gui.draw()
    pyplot.show()