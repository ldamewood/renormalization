"""
This is incomplete
"""

import numpy as np

from sklearn.ensemble.forest import BaseForest
from sklearn.tree.tree import BaseDecisionTree
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_X_y

class AdaBoostAbstainingClassifier(AdaBoostClassifier):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        super(AdaBoostAbstainingClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        y : array-like of shape = [n_samples]
            The target values (class labels).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
                # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, 
                         force_all_finite = False, dtype=dtype)

        if sample_weight is None:
            abstainrows = np.isnan(X).sum(axis=1) > 0
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1.
            sample_weight[abstainrows] = 0.

        # Normalize existing weights
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        # Check that the sample weights sum is positive
        if sample_weight.sum() <= 0:
            raise ValueError(
                "Attempting to fit with a non-positive "
                "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self
        
    def _boost_real(self, iboost, X, y, sample_weight):
        raise NotImplementedError

    def _boost_discrete(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass
            
        abstaining = np.isnan(X).sum(axis=1) > 0

        X_trans = X[~abstaining]
        y_trans = y[~abstaining]
        sw_trans = sample_weight[~abstaining]
        estimator.fit(X_trans, y_trans, sample_weight=sw_trans)

        y_predict = estimator.predict(X_trans)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances classified
        incorrect = np.zeros(y.shape)
        incorrect[~abstaining] = y_predict != y_trans
        correct = np.zeros(y.shape)
        correct[~abstaining] = y_predict == y_trans

        n_classes = self.n_classes_

        # incorrect, correct, abstaining weights
        Wi = np.sum(incorrect) / X.shape[0]
        Wc = np.sum(correct) / X.shape[0]
        Wa = np.sum(abstaining) / X.shape[0]
        Z = Wa + 2. * np.sqrt(Wc/Wi)

        # Stop if classification is perfect
        if estimator_error <= 0 and abstaining.sum() == 0:
            return sample_weight, 1., 0.

        

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log(Wc/Wi) + 
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error