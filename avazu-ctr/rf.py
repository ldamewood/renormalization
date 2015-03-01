#!/usr/bin/env python
from __future__ import print_function

from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

import ctr

learner = RandomForestClassifier(verbose = False, n_jobs = -1)

for ID,x,y in ctr.data(ctr.train, batchsize = 1):
    pass