from __future__ import division, print_function

import sys
from csv import DictReader
from math import exp, log, sqrt
import random
from datetime import date
from stats import RunningStatistics
import multiprocessing
from os.path import dirname, join

# from sklearn.utils.murmurhash import murmurhash3_bytes_s32
import mmh3

def murmur3(x):
    # return hash(x)
    return mmh3.hash(x)

def quickSelect(seq, k):
    # this part is the same as quick sort
    len_seq = len(seq)
    if len_seq < 2: return seq

    ipivot = len_seq // 2
    pivot = seq[ipivot]

    smallerList = [x for i,x in enumerate(seq) if x <= pivot and  i != ipivot]

    # here starts the different part
    m = len(smallerList)
    if k == m:
        return pivot
    elif k < m:
        return quickSelect(smallerList, k)
    else:
        largerList = [x for i,x in enumerate(seq) if x > pivot and  i != ipivot]
        return quickSelect(largerList, k-m-1)

def median(seq):
    # Find the median, but ignore the bias term so the list has odd number of
    # elements.
    k = len(seq) // 2
    return quickSelect(seq[1:], k)

# A, paths
train = join(dirname(__file__),'train')
test = join(dirname(__file__),'test')
submission = 'submission.csv'  # path of to be outputted submission file
ncpus = 2

# B, model
alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 10             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
bags = 4
epoch = 1000       # learn training data for N passes

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(murmur3(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path))):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # extract date
        year = int('20' + row['hour'][:2])
        month = int(row['hour'][2:4])
        day = int(row['hour'][4:6])
        dayofweek = str(date(year, month, day).weekday())
        row['hour'] = row['hour'][6:]
        row['dayofweek'] = dayofweek

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(murmur3(key + '_' + value)) % D
            x.append(index)

        yield t, date, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

# initialize ourselves a learner
learners = [ftrl_proximal(alpha, beta, L1, L2, D, interaction) for _ in range(bags)]
order = range(bags)
pool = multiprocessing.Pool()

# start training
for e in xrange(epoch):
    losses = [0. for _ in range(bags)]
    count = [0 for _ in range(bags)]
    stats = [RunningStatistics() for _ in range(bags)]

    for t, date, ID, x, y in data(train, D):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)
        random.shuffle(order)
        if t%10000 == 0 and t > 0:
            break
            print('Iteration %d' % t)
            for i in range(bags): 
                print('epoch %d : iteration %d : logloss[%d] = %f' % (e,t,i,losses[i]/count[i]))
        p = pool.map(lambda y:y.predict(x), learners)
        #p = [learner.predict(x) for learner in learners]
        for i,learner in zip(order,learners):
            if t%bags == i:
                losses[i] += logloss(p[i],y)
                count[i] += 1
            else:
                learner.update(x,p[i],y)
                stats[i].push(p[i]-y)
    
    for i,loss in enumerate(losses): 
        print('epoch %d : logloss[%d] = %f' % (e,i,loss))
    sys.stdout.flush()
    
    dL1all = []
    dL2all = []
    for stat,learner in zip(stats,learners):
        # Assume the median is zero
        mean = sum(learner.z.values()) / D
        med = median(learner.z.values())
        bhat = sum([abs(val - med) for val in learner.z.values()]) / D
        std = sqrt(sum([(val - mean)**2 for val in learner.z.values()]) / D)
        dL1 = abs(learner.L1 - bhat / stat.std ** 2)
        dL2 = abs(learner.L2 - std**2 / stat.std ** 2)
        learner.L1 = bhat / stat.std ** 2
        learner.L2 = std**2 / stat.std ** 2
        print('dL1 = %f, dL2 = %f' % (dL1/learner.L1, dL2/learner.L2))
        print('L1 = %f, L2 = %f' % (learner.L1,learner.L2))
        dL1all.append(dL1/learner.L1)
        dL2all.append(dL2/learner.L2)
    if max(dL1all) < 1.e-2 and max(dL2all) < 1.e-2: break

for learner in learners:
    print('L1 = %f, L2 = %f' % (learner.L1,learner.L2))

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

with open(submission, 'w') as outfile:
    outfile.write('id,click\n')
    for t, date, ID, x, y in data(test, D):
        p = sum([learner.predict(x) for learner in learners])/bags
        outfile.write('%s,%s\n' % (ID, str(p)))	
