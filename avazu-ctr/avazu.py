import ctr
from ftrl_proximal import ftrl_proximal
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import random
from math import log
import numpy
from stats import RunningStatistics

alpha = 100.
beta = 1.
L1 = 0.
L2 = .1
D = 2**20
epoch = 100
kfold = 2
order = range(kfold)

learners = [SGDClassifier(alpha = alpha, loss = 'log', shuffle = True, warm_start = True, class_weight = 'auto') for _ in range(kfold)]
hasher = FeatureHasher(n_features = D, dtype = int)

# start training
for e in xrange(epoch):
    random.shuffle(order)
    loss = kfold*[0.]
    stats = [RunningStatistics() for _ in range(kfold)]
    
    count = 0
 
    for t, (ID, f, y) in enumerate(ctr.data(ctr.train, batchsize = 10000)):  # data is a generator
        x = hasher.transform(f) 
        # step 1, get prediction from learners
        if t > kfold:
            p = [learner.predict_proba(x) for learner in learners]
        else:
            p = [0.5 * numpy.ones([x.shape[0],2]) for learner in learners]
        for i in range(len(learners)):
            if t%kfold == order[i] and t > kfold:
                loss[order[i]] += log_loss(y, p[order[i]])
            else:
                learners[order[i]].fit(x,y)
                stats[order[i]].push(y - p[order[i]][:,1])
    for i,learner in enumerate(learners):
        if stats[i].std < 1e-10: continue
        learner.set_params(alpha = numpy.std(learner.coef_)**2 / stats[i].std**2)
        print('New alpha: %f' % learner.alpha)
    print('Loss:')
    print(loss)