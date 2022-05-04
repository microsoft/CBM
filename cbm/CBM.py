# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cbm_cpp
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import List, Tuple, Union
from pandas.api.types import CategoricalDtype


class CBM(BaseEstimator):
    cpp: cbm_cpp.PyCBM

    def __init__(self,
        learning_rate_step_size:float = 1/100,
        max_iterations:int = 100,
        min_iterations_early_stopping:int = 20,
        epsilon_early_stopping:float = 1e-3,
        single_update_per_iteration:bool = True,
        metric: str = 'rmse',
        enable_bin_count: bool = False
        ) -> None:
        """Initialize the CBM model.

        Args:
            learning_rate_step_size (float, optional): [description]. Defaults to 1/100.
            max_iterations (int, optional): [description]. Defaults to 100.
            min_iterations_early_stopping (int, optional): [description]. Defaults to 20.
            epsilon_early_stopping (float, optional): [description]. Defaults to 1e-3.
            single_update_per_iteration (bool, optional): [description]. Defaults to True.
            date_features (List[str], optional): [description]. Defaults to ['day', 'month'].
            binning (Union[int, lambda x, optional): [description]. Defaults to 10. 
                The number of bins to create for continuous features. Supply lambda for flexible binning.
            metric (str): [description]. Used to determine when to stop. Defaults to 'rmse'. Options are rmse, smape, l1.
        """        

        self.learning_rate_step_size = learning_rate_step_size
        self.max_iterations = max_iterations
        self.min_iterations_early_stopping = min_iterations_early_stopping
        self.epsilon_early_stopping = epsilon_early_stopping
        self.single_update_per_iteration = single_update_per_iteration
        self.enable_bin_count = enable_bin_count

        self.metric = metric

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: np.ndarray
            ) -> "CBM":

        X, y = check_X_y(X, y, y_numeric=True)

        # pre-processing
        y_mean = np.average(y)

        # determine max bin per categorical
        x_max = X.max(axis=0)
        x_max_max = x_max.max()

        if x_max_max <= 255:
            self._x_type = "uint8"
        elif x_max_max <= 65535:
            self._x_type = "uint16"
        elif x_max_max <= 4294967295:  
            self._x_type = "uint32"
        else: 
            raise ValueError("Maximum of 255 categories per features")

        X = X.astype(self._x_type)

        self._cpp = cbm_cpp.PyCBM()
        self._cpp.fit(
            y.astype("uint32"), 
            X,
            y_mean, 
            x_max.astype("uint32"),
            self.learning_rate_step_size,
            self.max_iterations,
            self.min_iterations_early_stopping,
            self.epsilon_early_stopping,
            self.single_update_per_iteration,
            self.metric,
            self.enable_bin_count
            )

        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray, explain: bool = False):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        return self._cpp.predict(X.astype(self._x_type), explain)

    def update(self, weights: list, y_mean: float):
        if "_cpp" not in self.__dict__:
            self._cpp = cbm_cpp.PyCBM()

        x_max_max = max(map(len, weights))
        if x_max_max <= 255:
            self._x_type = "uint8"
        elif x_max_max <= 65535:
            self._x_type = "uint16"
        elif x_max_max <= 4294967295:  
            self._x_type = "uint32"
        else: 
            raise ValueError("Maximum of 255 categories per features")

        self._cpp.weights = weights
        self._cpp.y_mean = y_mean

        self.is_fitted_ = True

    @property
    def weights(self):
        return self._cpp.weights

    @property
    def y_mean(self):
        return self._cpp.y_mean

    @property
    def iterations(self):
        return self._cpp.iterations

    @property
    def bin_count(self):
        return self._cpp.bin_count