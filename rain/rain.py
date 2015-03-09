#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

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
from sklearn.preprocessing import StandardScaler

# Data files
train_set = join(expanduser('~'), 'Downloads', 'train_2013.csv')
test_set = join(expanduser('~'), 'Downloads', 'test_2014.csv')
outfile = join(expanduser('~'), 'Downloads', 'test_results_2014.csv')

def data(filepath, poly2 = True, deriv = True, extra = True):
    """
    Process the data file and do transformations:
        * Split the time series.
        * Standardize the NAN values.
        * Calculate extra features if extra == True:
            ref: http://cimms.ou.edu/~schuur/disdrom/kyoto.pdf
        * Calculate time derivatives if deriv == True.
        * Calculate poly2 features if poly2 == True.
    """
    nan = float('nan')

    # values that are converted to nan
    nan_values = ['-99900.0','-99901.0','-99903.0','999.0','nan']
    
    # features that cannot have nan and are not used in poly2 or deriv
    not_features = ['Expected', 'Id']
    
    for row in DictReader(open(filepath)):
        
        # Extract the time series
        ntime = len(row['TimeToEnd'].split())

        # Split the row by the provided spaces into a time series
        d = {}
        for key, value in row.iteritems():
            if len(value.split()) > 1:
                d[key] = value.split()
            else:
                # Rows that contain a common value for all time steps
                d[key] = ntime * [value]

        # Calculate extra features
        if extra:
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

        # Return values
        # List of dicts with features as keys and time series list as value
        allr = []
        allexpected = []
        allId = []
        
        # Loop over the times and yield time steps, yield each ID
        oldr = {}
        for time in range(ntime):
            r = {}

            for key, value in d.items():

                if key in not_features: continue
                
                # Standardize the NAN values
                if value[time] in nan_values:
                    r[key] = nan
                    if deriv: r['%s_deriv' % key] = nan
                else:
                    r[key] = float(value[time])
                    
                    # Do not calculate derivatives (or just of some columns)
                    if not deriv or key in not_features: continue
                    
                    # Derivatives
                    if time > 0 and float(d['TimeToEnd'][time]) > float(d['TimeToEnd'][time - 1]):
                        r['%s_deriv' % key] = (r[key] - oldr[key])/(float(d['TimeToEnd'][time]) - float(d['TimeToEnd'][time - 1]))
                    else:
                        r['%s_deriv' % key] = 0.
                     
            # Interaction features
            if poly2:
                for (k1,v1),(k2,v2) in itertools.combinations_with_replacement(r.items(), 2):
                    if k1 in not_features or k2 in not_features: continue
                    r['%s_%s' % (k1,k2)] = v1*v2

            # Cap the expected values
            expected = int(float(d['Expected'][time]))
            if expected > 69: expected = 70
            
            allr.append(r)
            allexpected.append(expected)
            allId.append(int(float(d['Id'][time])))
            
            # Store current values to calculate derivatives
            oldr = r.copy()

        yield allr, allexpected, allId

def minibatch(data, bs = 1000):
    """ Group the Ids. """
    X = []
    y = []
    ids = []
    for i,(data,expected,Id) in enumerate(data):
        [X.append(d.values()) for d in data]
        [y.append(d) for d in expected]
        [ids.append(d) for d in Id]
        if i % bs == 0 and i > 0 or bs == 1:
            features = data[0].keys()
            yield X,y,ids,features
            X = []
            y = []
            ids = []
    yield X,y,ids,features

def group_and_mean(array, group_ids):
    """
    Group by "group_ids" and take the mean.
    TODO: can this be easily done without pandas? Of course!
    """
    
    # Check if ids are given in sorted order
    #assert sum([group_ids[i] != j for i,j in enumerate(sorted(group_ids))]) == 0
    df = pd.DataFrame(array)
    df['Groups'] = group_ids
    return np.array(df.groupby('Groups').mean())

def score_crp(y_pred, y_real, ids):
    """
    Gives the score based on the classification probability and expected values.
    """
    yp = np.array(group_and_mean(y_pred, ids)).cumsum(axis=1)
    ya = np.array(group_and_mean(y_real, ids)).flatten()
    x = range(70)
    return np.array([(yp[:,n] - (n >= ya))**2 for n in x]).T / len(x) / len(ya)       

if __name__ == '__main__':
    base_classifier = SGDClassifier(loss='modified_huber', n_jobs = -1)
    transform = make_pipeline(Imputer(), StandardScaler())
    a = datetime.datetime.now()
    for epoch in range(1):
        for i,(X,y,ids,features) in enumerate(minibatch(data(train_set))):
            if i%100 == 0 and i > 0:
                y_pred = base_classifier.predict_proba(transform.fit_transform(X))
                print(datetime.datetime.now() - a)
                print(score_crp(y_pred, y, ids).sum())
            else:
                base_classifier.partial_fit(transform.fit_transform(X), np.array(y), classes = range(71))
            break
        break
