#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from csv import DictReader
from os.path import join, expanduser
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer

train_set = join(expanduser('~'), 'Downloads', 'train_2013.csv')
test_set = join(expanduser('~'), 'Downloads', 'test_2014.csv')
outfile = join(expanduser('~'), 'Downloads', 'test_results_2014.csv')

#def weights(y):
#    ifreq = itemfreq(y)
#    w = {x : 1.*f/len(y) for x,f in ifreq}
#    return w#[w[x] / len(ifreq) for x in y]
#
#def dict_hash(d):
#    acc = 0
#    for key, value in sorted(d.items()):
#        if key in ['hash']: continue
#        acc <<= 1
#        acc |= 0 if math.isnan(value) else 1
#    return acc

def data(filepath):
    SET = -1
    nan = float('nan')
    nan_values = ['-99900.0','-99901.0','-99903.0','999.0','nan']
    for row in DictReader(open(filepath)):
        ntime = len(row['TimeToEnd'].split())
        d = {}
        for key, value in row.iteritems():
            if len(value.split()) > 1:
                d[key] = value.split()
            else:
                d[key] = ntime * [value]
        prevTTE = 0.
        d['Set'] = []
        for TTE in d['TimeToEnd']:
            if float(TTE) > prevTTE: SET += 1
            d['Set'].append(SET)
            prevTTE = float(TTE)
        for time in range(ntime):
            r = {}
            for key, value in sorted(d.items()):
                if value[time] in nan_values:
                    r[key] = nan
                else:
                    r[key] = float(value[time])
            #r['hash'] = dict_hash(r)
            yield r

def minibatch(data, bs = 1000):
    b = []
    for i,d in enumerate(data):
        b.append(d)
        if i % bs == 0 and i > 0 or bs == 1:
            yield b
            b = []
    yield b

def score(y, y_pred):
    return sum([(y[:,n] - (n > y_pred))**2 for n in range(70)]) / 70. / y_pred.shape[0]

def train(dataset):
    import datetime

    df = pd.DataFrame()

    a = datetime.datetime.now()
    bs = 100000
    for i,d in enumerate(minibatch(data(dataset), bs = bs)):
        df = pd.concat([df, pd.DataFrame.from_dict(d)], ignore_index = True)
        print(i * bs, datetime.datetime.now() - a)
    print(datetime.datetime.now() - a)
    
    #df.Expected = pd.cut(df.Expected, np.linspace(0,69,70), include_lowest=True)
    df[['Expected']] = df[['Expected']].astype(int)
    df.ix[df.Expected > 69,'Expected'] = 70
    
    features = [c for c in df.columns if c not in ['Id', 'Set', 'Expected', 'TimeToEnd']] 
    X = np.array(df.ix[:,features])
    y = np.array(df.Expected)

    trainer = make_pipeline(Imputer(), SGDClassifier(loss = 'log', n_jobs = -1, shuffle = True, class_weight = 'auto'))
    scorer = make_scorer(score, greater_is_better = False, needs_proba = True)
    print(cross_val_score(trainer, X, y, cv = 3, scoring = scorer))
    #trainer.fit(X,y)
    #return trainer

def predict(trainer, dataset):
    
    import datetime

    df = pd.DataFrame()

    a = datetime.datetime.now()
    bs = 100000
    for i,d in enumerate(minibatch(data(dataset), bs = bs)):
        df = pd.concat([df, pd.DataFrame.from_dict(d)], ignore_index = True)
        print(i * bs, datetime.datetime.now() - a)
    print(datetime.datetime.now() - a)
    features = [c for c in df.columns if c not in ['Id', 'Set', 'Expected', 'TimeToEnd']]
    
    X = np.array(df.ix[:,features])
    
    y_predict = trainer.predict_proba(X)
    
    res = np.zeros([X.shape[0],71])
    for i,j in enumerate(trainer.steps[-1][-1].classes_):
        res[:,j] = y_predict[:,i]
    
    cs = np.cumsum(res[:,:70], axis = 1)
    df_predict = pd.DataFrame(cs)
    df_predict.columns = ['Predicted%d' % i for i in range(70)]
    df_predict['Id'] = df.Id.astype('int')
    result = df_predict.groupby('Id').mean()
    result.to_csv(outfile)

if __name__ == '__main__':
    
    trainer = train(train_set)
    #predict(trainer, test_set)
    