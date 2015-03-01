#!/usr/bin/env python
from __future__ import print_function

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from stats import RunningStatistics
import numpy

import ctr

L2 = 0.1
interactions = True
max_est = 25
sgd = SGDClassifier(loss = 'log', penalty = 'L2', alpha = L2, shuffle = True, n_iter = 1)
hasher = FeatureHasher(n_features = 2 ** 20)
learner = AdaBoostClassifier(sgd, n_estimators = max_est)

batchsize = 10000

diff = [0. for _ in range(max_est)]
for epoch in range(100):
    rs = RunningStatistics()
    for i,(x,y) in enumerate(ctr.data(ctr.train, batchsize = batchsize)):
        learner.fit(x, y)
        p = learner.predict_proba(x)
        rs.push(y - p[:,1])
        print(log_loss(y, p))
        print('Batch #%d fitted on sgd[%d]: %d clicks out of %d' % (i,0,sum(y),batchsize))
        break
    for i, estimator in enumerate(learner.estimators_):
        diff[i] = estimator.alpha - numpy.std(estimator.coef_) / rs.std
        estimator.set_params(alpha = numpy.std(estimator.coef_) / rs.std)
    if numpy.max(diff) < 1e-4 and epoch > 0: break
    break