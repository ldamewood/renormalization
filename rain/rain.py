#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from csv import DictReader
from os.path import join, expanduser
import numpy as np
import pandas as pd
import itertools
import math
import datetime

from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

train_set = join(expanduser('~'), 'Downloads', 'train_2013.csv')
test_set = join(expanduser('~'), 'Downloads', 'test_2014.csv')
outfile = join(expanduser('~'), 'Downloads', 'test_results_2014.csv')
poly2features = True

def data(filepath, poly2 = True):
    SET = -1
    nan = float('nan')
    nan_values = ['-99900.0','-99901.0','-99903.0','999.0','nan']
    nopoly2 = ['Expected', 'Id', 'Set']
    for row in DictReader(open(filepath)):
        ntime = len(row['TimeToEnd'].split())
        d = {}

        # Split the row by the provided spaces into a time series
        for key, value in row.iteritems():
            if len(value.split()) > 1:
                d[key] = value.split()
            else:
                d[key] = ntime * [value]

        # Assign the split rows into Sets if the row contains multiple time series
        prevTTE = 0.
        d['Set'] = []
        for TTE in d['TimeToEnd']:
            if float(TTE) > prevTTE: SET += 1
            d['Set'].append(SET)
            prevTTE = float(TTE)

        # Calculate extra features: http://cimms.ou.edu/~schuur/disdrom/kyoto.pdf
        d['RR4'] = []
        d['RR5'] = []
        for time in range(ntime):
            if d['Kdp'][time] in nan_values:
                d['RR4'].append(nan)
                d['RR5'].append(nan)
                continue
            else:
                kdp = float(d['Kdp'][time])
                d['RR4'].append(abs(kdp)**0.802 * math.copysign(1,kdp))
                if d['Zdr'][time] in nan_values:
                    d['RR5'].append(nan)
                    continue
                else:
                    zdr = float(d['Zdr'][time])
                    if zdr >= 0.5:
                        rr5 = abs(kdp)**0.910 * math.copysign(1,kdp) * zdr ** (-0.421)
                    else:
                        rr5 = abs(kdp)**0.878 * math.copysign(1,kdp) * 10 ** (-0.131 * zdr)
                    d['RR5'].append(rr5)

        # Loop over the times and yield time steps
        oldr = {}
        allr = []
        allexpected = []
        allId = []
        allSet = []
        for time in range(ntime):
            r = {}

            for key, value in d.items():

                if key in ['Set']: r['Set'] = float(value[time])

                if key in ['Expected','Id','Set']: continue
                
                # Standardize the NAN values
                if value[time] in nan_values:
                    r[key] = nan
                    r['%s_deriv' % key] = nan
                else:
                    r[key] = float(value[time])
                    
                    # Do not calculate derivatives of some features
                    if key in ['Expected','Set']: continue
                    
                    # "Derivatives"
                    if 'Set' in oldr and oldr['Set'] == d['Set'][time] and float(d['TimeToEnd'][time]) != oldr['TimeToEnd']:
                        r['%s_deriv' % key] = (r[key] - oldr[key])/(float(d['TimeToEnd'][time]) - oldr['TimeToEnd'])
                    else:
                        r['%s_deriv' % key] = 0.
                     
            # Interaction features
            if poly2:
                for (k1,v1),(k2,v2) in itertools.combinations_with_replacement(r.items(), 2):
                    if k1 not in nopoly2 and k2 not in nopoly2:
                        r['%s_%s' % (k1,k2)] = v1*v2

            # Cap the expected values
            expected = int(float(d['Expected'][time]))
            Id = int(float(d['Id'][time]))
            Set = int(float(d['Set'][time]))
            if expected > 69: expected = 70
            allr.append(r)
            allexpected.append(expected)
            allId.append(Id)
            allSet.append(Set)
            #yield r, expected, Id, Set
            oldr = r.copy()
        yield allr, allexpected, allId, allSet

def minibatch(data, bs = 1000):
    X = []
    y = []
    features = []
    ids = []
    sets = []
    for i,(data,expected,Id,Set) in enumerate(data):
        [X.append(d.values()) for d in data]
        [y.append(d) for d in expected]
        [ids.append(d) for d in Id]
        [sets.append(d) for d in Set]
        features = data[0].keys()
        if i % bs == 0 and i > 0 or bs == 1:
            yield X,y,ids,sets,features
            X = []
            y = []
            ids = []
            sets = []
    yield X,y,ids,sets,features

def group_and_mean(array, groups):
    assert sum([groups[i] != j for i,j in enumerate(sorted(groups))]) == 0
    df = pd.DataFrame(array)
    df['Groups'] = groups
    return np.array(df.groupby('Groups').mean())

def scoring(trainer, X_1, X_2, y_1, y_2, ids_1, x = range(70)):
    trainer.fit(X_2, y_2)
    result = crp(y_2, trainer.predict_proba(X_1))
    result_grouped = group_and_mean(result, ids_1)
    y_grouped = group_and_mean(y_1, ids_1)
    scr = np.array([(result_grouped[:,n] - (n >= y_grouped))**2 for n in x]).T / len(x) / y_grouped.shape[0]
    return scr.sum()

def score_crp(y_pred, y_real, ids):
    yp = group_and_mean(y_pred, ids).cumsum(axis=1)
    ya = group_and_mean(y_real, ids).flatten()
    x = range(70)
    return np.array([(yp[:,n] - (n >= ya))**2 for n in x]).T / len(x) / len(ya)

#def load_data(dataset):
#    import datetime
#
#    df = pd.DataFrame()
#
#    a = datetime.datetime.now()
#    bs = 10000
#    for i,d in enumerate(minibatch(data(dataset, poly2 = True), bs = bs)):
#        df = pd.DataFrame.from_dict(d)
#        print(datetime.datetime.now() - a)
#        df[['Expected']] = df[['Expected']].astype(int)
#        df.ix[df.Expected > 69,'Expected'] = 70
#        features = [c for c in df.columns if c not in ['Id', 'Set', 'Expected', 'TimeToEnd']] 
#        X = np.array(df.ix[:,features])
#        y = np.array(df.Expected)
#        ids = np.array(df.Id.astype('int'))
#        yield X,y,ids,features

#def predict(trainer, dataset):
#    
#    import datetime
#
#    df = pd.DataFrame()
#
#    a = datetime.datetime.now()
#    bs = 100000
#    for i,d in enumerate(minibatch(data(dataset), bs = bs)):
#        df = pd.concat([df, pd.DataFrame.from_dict(d)], ignore_index = True)
#        print(i * bs, datetime.datetime.now() - a)
#        break
#    features = [c for c in df.columns if c not in ['Id', 'Set', 'Expected', 'TimeToEnd']]
#    
#    df[['Expected']] = df[['Expected']].astype(int)
#    df.ix[df.Expected > 69,'Expected'] = 70
#    
#    X = np.array(df.ix[:,features])
#    y = np.array(df.Expected)
#    ids = np.array(df.Id.astype('int'))
#    return X,y,ids,features        

if __name__ == '__main__':
    base_classifier = SGDClassifier(loss='modified_huber', n_jobs = -1)
    transform = make_pipeline(Imputer(), StandardScaler())
    a = datetime.datetime.now()
    for epoch in range(1):
        for i,(X,y,ids,sets,features) in enumerate(minibatch(data(train_set, poly2 = True), bs = 1000)):
            if i%100 == 0 and i > 0:
                y_pred = base_classifier.predict_proba(transform.fit_transform(X))
                print(datetime.datetime.now() - a)
                print(score_crp(y_pred, y, ids).sum())
            else:
                base_classifier.partial_fit(transform.fit_transform(X), np.array(y), classes = range(71))
#
#    print(scoring(trainer, X_2, X_1, y_2, y_1, ids_2))
#    print(scoring(trainer, X_1, X_2, y_1, y_2, ids_1))

    #trainer.fit(X,y)
    #predict(trainer, test_set)
