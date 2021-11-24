# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cbm_cpp
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import List, Union
from pandas.api.types import CategoricalDtype


class CBM(BaseEstimator):
    cpp: cbm_cpp.PyCBM

    def __init__(self,
        learning_rate_step_size:float = 1/100,
        max_iterations:int = 100,
        min_iterations_early_stopping:int = 20,
        epsilon_early_stopping:float = 1e-3,
        single_update_per_iteration:bool = True,
        date_features: Union[str, List[str]] = 'day,month',
        binning: Union[int, lambda x: int] = 10,
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
        """        

        self.learning_rate_step_size = learning_rate_step_size
        self.max_iterations = max_iterations
        self.min_iterations_early_stopping = min_iterations_early_stopping
        self.epsilon_early_stopping = epsilon_early_stopping
        self.single_update_per_iteration = single_update_per_iteration

        # lets make sure it's serializable
        if isinstance(date_features, list):
            date_features = ",".join(date_features)
        self.date_features = date_features
        self.binning = binning

    def get_date_features(self) -> List[str]:
        return self.date_features.split(",")

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: np.ndarray
            ) -> "CBM":

        # keep feature names around

        if isinstance(X, pd.DataFrame):
            self._feature_names = []
            self._feature_categories = []
            self._feature_bins = []

            X_numeric = []

            for col in X.columns:
                col_dtype = X[col].dtype

                if pd.api.types.is_datetime64_any_dtype(col_dtype):
                    for expansion in self.get_date_features():
                        import calendar

                        if expansion == 'day':
                            self._feature_names.append(f'{col}_day')
                            self._feature_categories.append(calendar.day_abbr)
                            self._feature_bins.append(None)

                            X_numeric.append(X[col].dt.dayofweek.values)

                        elif expansion == 'month':
                            self._feature_names.append(f'{col}_month')
                            self._feature_categories.append(calendar.month_abbr)
                            self._feature_bins.append(None)

                            X_numeric.append(X[col].dt.month.values)

                elif pd.api.types.is_float_dtype(col_dtype):
                    # deal with continuous features
                    bin_num = self.binning if isinstance(self.binning, int) else self.binning(X[col])

                    X_binned, bins  = pd.qcut(X[col].fillna(0), bin_num, retbins=True)

                    self._feature_names.append(col)
                    self._feature_categories.append(X_binned.cat.categories.astype(str).tolist())
                    self._feature_bins.append(bins)

                    X_numeric.append(pd.cut(X[col].fillna(0), bins, include_lowest=True).cat.codes)

                elif not pd.api.types.is_integer_dtype(col_dtype):
                    self._feature_names.append(col)

                    # convert to categorical
                    X_cat = (X[col]
                        .fillna('CBM_UnknownCategory')
                        .astype('category'))

                    # keep track of categories
                    self._feature_categories.append(X_cat.cat.categories.tolist())
                    self._feature_bins.append(None)

                    # convert to 0-based index
                    X_numeric.append(X_cat.cat.codes)
                else:
                    self._feature_names.append(col)
                    self._feature_categories.append(None)
                    self._feature_bins.append(None)

                    X_numeric.append(X[col])

            X = np.column_stack(X_numeric)
        else:
            self._feature_names = None

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
            )

        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray, explain: bool = False):
        if isinstance(X, pd.DataFrame):
            X_numeric = []

            offset = 0 # correct for date expansion
            for i, col in enumerate(X.columns):
                col_dtype = X[col].dtype

                if pd.api.types.is_datetime64_any_dtype(col_dtype):
                    for expansion in self.get_date_features():
                        if expansion == 'day':
                            X_numeric.append(X[col].dt.dayofweek.values)
                            offset += 1

                        elif expansion == 'month':
                            X_numeric.append(X[col].dt.month.values)
                            offset += 1

                    offset -= 1                           

                elif pd.api.types.is_float_dtype(col_dtype):
                    # re-use binning from training
                    X_numeric.append(pd.cut(X[col].fillna(0), self._feature_bins[i + offset], include_lowest=True).cat.codes)

                elif not pd.api.types.is_integer_dtype(col_dtype):
                    # convert to categorical
                    X_cat = (X[col]
                        .fillna('CBM_UnknownCategory')
                        # re-use categories from training
                        .astype(CategoricalDtype(categories=self._feature_categories[i + offset], ordered=True)))

                    # convert to 0-based index
                    X_numeric.append(X_cat.cat.codes)
                else:
                    X_numeric.append(X[col])

            X = np.column_stack(X_numeric)

        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        return self._cpp.predict(X.astype(self._x_type), explain)

    def update(self, weights: list, y_mean: float):
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

    def plot_importance(self, feature_names: list = None, **kwargs):
        check_is_fitted(self, "is_fitted_")

        if feature_names is None:
            feature_names = self._feature_names

        import matplotlib.pyplot as plt

        n_features = len(self.weights)

        n_cols = int(np.ceil( np.sqrt(n_features)))
        n_rows = int(np.floor(np.sqrt(n_features)))

        if n_cols * n_rows < n_features:
            n_rows += 1

        fig, ax = plt.subplots(n_rows, n_cols, sharex=True, **kwargs)

        fig.suptitle(f'Response mean: {self.y_mean:0.4f} | Iterations {self.iterations}')

        for f in range(n_features):
            w = np.array(self.weights[f])
            
            color = np.where(w < 1, 'xkcd:tomato', 'xkcd:green')

            ax_sub = ax[f // n_cols, f % n_cols]
            ax_sub.barh(range(len(w)), w - 1, color=color)

            ax_sub.set_title(feature_names[f] if feature_names is not None else f'Feature {f}')

            if self._feature_categories[f] is not None:
                ax_sub.set_yticks(range(len(self._feature_categories[f])))
                ax_sub.set_yticklabels(self._feature_categories[f])


    @property
    def weights(self):
        return self._cpp.weights

    @property
    def y_mean(self):
        return self._cpp.y_mean

    @property
    def iterations(self):
        return self._cpp.iterations
