# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cbm_cpp
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class CBM(BaseEstimator):
    cpp: cbm_cpp.PyCBM

    def __init__(self,
        learning_rate_step_size:float = 1/100,
        max_iterations:int = 100,
        min_iterations_early_stopping:int = 20,
        epsilon_early_stopping:float = 1e-3,
        single_update_per_iteration:bool = True) -> None:

        self.learning_rate_step_size = learning_rate_step_size
        self.max_iterations = max_iterations
        self.min_iterations_early_stopping = min_iterations_early_stopping
        self.epsilon_early_stopping = epsilon_early_stopping
        self.single_update_per_iteration = single_update_per_iteration

    def fit(self,
            X: np.ndarray,
            y: np.ndarray
            ) -> 'CBM':

        X, y = check_X_y(X, y, y_numeric=True)

        y_mean = np.average(y)

        # determine max bin per categorical
        x_max = X.max(axis=0)

        if np.any(x_max > 255):
            raise ValueError("Maximum of 255 categories per features")

        self._cpp = cbm_cpp.PyCBM()
        self._cpp.fit(
            y.astype('uint32'), 
            X.astype('uint8'),
            y_mean, 
            x_max.astype('uint8'),
            self.learning_rate_step_size,
            self.max_iterations,
            self.min_iterations_early_stopping,
            self.epsilon_early_stopping,
            self.single_update_per_iteration)

        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray, explain: bool = False):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')

        return self._cpp.predict(X.astype('uint8'), explain)

    @property
    def weights(self):
        return self._cpp.weights