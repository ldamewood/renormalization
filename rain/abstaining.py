#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is dumb

from __future__ import print_function
from sklearn.ensemble.base import BaseEnsemble
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import itemfreq

import numpy as np
import itertools

def hash_row(row):
    one = np.uint64(1)
    zero = np.uint64(0)
    acc = zero
    for i in row:
        acc = np.left_shift(acc, one)
        acc = np.bitwise_or(acc, one if i else zero)
    return acc

def unhash_row(h, cols = None):
    row = []
    h = np.uint64(h)
    one = np.uint64(1)
    zero = np.uint64(0)
    
    if cols is None:
        while h > 0:
            row.append(np.bitwise_and(h, one)==zero)
            h = np.right_shift(h, one)
    else:
        for col in range(cols):
            row.append(np.bitwise_and(h, one)==zero)
            h = np.right_shift(h, one)
    return row[::-1]

def weights(y):
    return {x : 1.*f/len(y) for x,f in itemfreq(y)}

def transform_X_for_hash(X, h = 0):
    cols = np.array(unhash_row(h, X.shape[1]))
    rows = ~np.isnan(X[:,cols]).any(axis=1)
    return X[rows][:,cols], rows, cols



class AbstainingClassifier(BaseEnsemble):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None,
                 verbose = False):

        super(AbstainingClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self._verbose = verbose
        self._poly2features = False

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Find all feature hashes
        # Feature: poly2 missing, etc hashes
        hashes = np.unique(np.array([hash_row(i) for i in np.isnan(X)], dtype = np.uint64))
        
        if self._poly2features:
            hashes = np.unique([np.bitwise_or(i,j) for i,j in list(itertools.product(hashes, hashes))])
        
        self._validate_estimator()
        
        self.estimators_ = {}
        self.estimator_weights_ = {}
        self.estimator_errors_ = {}
        
        for h in hashes:
            self.estimators_[h] = self._make_estimator(append = False)
            if self._verbose: print('Training on hash #%d' % h)

            X_trans, rows, cols = transform_X_for_hash(X, h)
            y_trans = y[rows]
            
            if np.unique(y_trans).shape[0] == 1:
                # Abstain from all data since there is only one class.
                # Possible improvement - could this tell us that these features don't do anything?
                self.estimator_weights_[h] = 0.
                del self.estimators_[h]
                continue
                
            self.estimators_[h].fit(X_trans,y_trans)
            incorrect = sum(self.estimators_[h].predict(X_trans) != y_trans)
            abstaining = X.shape[0] - rows.sum()
            wM = np.min(1. * incorrect / X.shape[0], 0.000001)
            wA = 1. * abstaining / X.shape[0]
            wC = 1 - wM - wA
            print(wC, wM, wA)
            self.estimator_weights_[h] = 1.#np.log(wC/wM) - np.log(len(self.estimators_[h].classes_) - 1)

        weightsum = sum(self.estimator_weights_.values())
        for h in self.estimator_weights_.iterkeys():
            self.estimator_weights_[h] /= weightsum
    
    def predict_proba(self, X):
        #hashes = np.array([hash_row(i) for i in np.isnan(X)], dtype=long)
        
        yres = np.zeros([X.shape[0], len(self.classes_)])

        for h in self.estimators_.iterkeys():
            weight = self.estimator_weights_[h]
            estimator = self.estimators_[h]
            
            X_trans, rows, cols = transform_X_for_hash(X, h)
            if X_trans.shape[0] < 2 or X_trans.shape[1] < 2: continue
            
            print(h,X_trans.shape)
            y_predict = estimator.predict_proba(X_trans)
            
            for cls in estimator.classes_:
                yres[rows][:,self.classes_ == cls] += weight * y_predict[:,estimator.classes_ == cls]
        return yres

    def predict(self, X):
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        #check_is_fitted(self, "n_classes_")
        #X = self._validate_X_predict(X)

        pred = None

        pred = np.zeros([X.shape[0], self.n_classes_])       
        for h in self.estimators_.iterkeys():
            X_trans, rows, cols = transform_X_for_hash(X, h)
            classes = self.estimators_[h].classes_[:, np.newaxis]
            n_classes = len(self.estimators_[h].classes_)
            pred[rows] += np.array([self.classes_ == self.estimators_[h].classes_[i] for i in (self.estimators_[h].predict(X_trans) == classes).T])
            pred[rows] *= self.estimator_weights_[h]

        #pred /= sum(self.estimator_weights_.values())
        #if n_classes == 2:
        #    pred[:, 0] *= -1
        #    return pred.sum(axis=1)
        return pred

if __name__ == '__main__':
    print(__doc__)
    
    # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
    # License: BSD 3 clause
    
    # Standard scientific Python imports
    import matplotlib.pyplot as plt
    
    # Import datasets, classifiers and performance metrics
    from sklearn import datasets, linear_model, metrics, svm
    
    # The digits dataset
    digits = datasets.load_digits()
    
    # The data that we are interested in is made of 8x8 images of digits, let's
    # have a look at the first 3 images, stored in the `images` attribute of the
    # dataset.  If we were working from image files, we could load them using
    # pylab.imread.  Note that each image must have the same size. For these
    # images, we know which digit they represent: it is given in the 'target' of
    # the dataset.
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)
    
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    
    data = digits.images.reshape((n_samples, -1))
    R = (np.random.random(data[:n_samples / 2].shape) > 0.01)
    data[:n_samples / 2] += np.power(R-1.,0.5)
    
    # Create a classifier: a support vector classifier
    baseclassifier = svm.SVC(gamma = 0.001)
    classifier = AbstainingClassifier(baseclassifier)
    
    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
    
    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples / 2:]
    predicted = classifier.predict(data[n_samples / 2:])
    
    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    
    images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)
    
    plt.show()