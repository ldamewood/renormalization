#!/usr/bin/env python

from os.path import dirname, join
from csv import DictReader
from datetime import date
import gzip
import numpy

train = join(dirname(__file__),'train.gz')
test = join(dirname(__file__),'test.gz')

# All features
feature_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 
                'site_category', 'app_id', 'app_domain', 'app_category', 
                'device_id', 'device_ip', 'device_model', 'device_type', 
                'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 
                'C19', 'C20', 'C21', 'dayofweek']

# Features in base 16
_base16features = ['device_ip', 'device_id', 'device_model', 'site_id',
                    'app_id', 'app_category', 'site_category', 'app_domain', 
                    'site_domain']

_base = {fn:10 for fn in feature_names}
_base.update({fn:16 for fn in _base16features})

def _process_row(row):
    # id
    ID = row['id']
    del row['id']

    # process click
    y = row['click'] == '1'
    del row['click']

    # process hour
    year = int('20' + row['hour'][:2])
    month = int(row['hour'][2:4])
    day = int(row['hour'][4:6])
    dayofweek = str(date(year, month, day).weekday())
    row['hour'] = row['hour'][6:]
    row['dayofweek'] = dayofweek

    # process remaining features
    #features = dict([fname, int(row[fname], _base[fname])] for fname in feature_names)

    return ID, row, y    

def data(gzipfile, batchsize = 1000):    
    IDs = batchsize * [None]
    y = batchsize * [False]
    features = batchsize * [None]

    with gzip.open(gzipfile, 'rb') as gz:
        for t, row in enumerate(DictReader(gz)):
            if t % batchsize == 0:
                if t > 0: yield IDs,features,y
                i = 0
            IDs[i], features[i], y[i] = _process_row(row)
            i += 1

#def submit(learner):
#    with open('submission', 'w') as outfile:
#        outfile.write('id,click\n')
#        for ID,x,y in data(test, batchsize = 1):
#            p = learner.predict_proba(x)[0,1]
#            outfile.write('%s,%s\n' % (ID, str(p)))