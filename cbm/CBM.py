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
        date_features: Union[str, List[str]] = 'day,month',
        binning: Union[int, lambda x: int] = 10,
        metric: str = 'rmse'
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

        # lets make sure it's serializable
        if isinstance(date_features, list):
            date_features = ",".join(date_features)
        self.date_features = date_features
        self.binning = binning
        self.metric = metric

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
            self.metric
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

    def _plot_importance_categorical(self, ax, feature_idx: int, vmin: float, vmax: float, is_continuous: bool):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("RdYlGn")

        # plot positive/negative impact (so 1.x to 0.x)
        weights = np.array(self.weights[feature_idx]) - 1

        alpha = 1
        if self._feature_bins[feature_idx] is not None or is_continuous:
            ax.plot(range(len(weights)), weights)
            alpha = 0.3

        weights_normalized = [x - vmin / (vmax - vmin) for x in weights]

        ax.bar(range(len(weights)), weights, color=cmap(weights_normalized), edgecolor='black', alpha=alpha)
        ax.set_ylim(vmin, vmax)

        # ax.barh(range(len(weights)), weights, color=cmap(weights_normalized), edgecolor='black', alpha=0.3)
        # ax.set_xlim(vmin, vmax)

        # ax_sub.set_title(feature_names[feature_idx] if feature_names is not None else f'Feature {feature_idx}')
        ax.set_ylabel('% change')

        if self._feature_names is not None:
            ax.set_xlabel(self._feature_names[feature_idx])

        if self._feature_categories[feature_idx] is not None:
            ax.set_xticks(range(len(self._feature_categories[feature_idx])))
            ax.set_xticklabels(self._feature_categories[feature_idx], rotation=45)

    def _plot_importance_interaction(self, ax, feature_idx: int, vmin: float, vmax: float):
        import matplotlib.pyplot as plt

        weights = np.array(self.weights[feature_idx]) - 1

        cat_df = pd.DataFrame(
            [(int(c.split('_')[0]), int(c.split('_')[1]), i) for i, c in enumerate(self._feature_categories[feature_idx])],
            columns=['f0', 'f1', 'idx'])
    
        cat_df.sort_values(['f0', 'f1'], inplace=True)
    
        cat_df_2d = cat_df.pivot(index='f0', columns='f1', values='idx')

        # resort index by mean weight value
        zi = np.array(weights)[cat_df_2d.to_numpy()]

        sort_order = np.argsort(np.max(zi, axis=1))
        cat_df_2d = cat_df_2d.reindex(cat_df_2d.index[sort_order])

        # construct data matrices
        xi = cat_df_2d.columns
        yi = cat_df_2d.index
        zi =  np.array(weights)[cat_df_2d.to_numpy()]
    
        im = ax.imshow(zi, cmap=plt.get_cmap("RdYlGn"), aspect='auto', vmin=vmin, vmax=vmax)
    
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('% change', rotation=-90, va="bottom")
    
        if self._feature_names is not None:
            names = self._feature_names[feature_idx].split('_X_')
            ax.set_ylabel(names[0])
            ax.set_xlabel(names[1])

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(xi)), labels=xi)
        ax.set_yticks(np.arange(len(yi)), labels=yi)

    def plot_importance(self, feature_names: list = None, continuous_features: list = None, **kwargs):
        """Plot feature importance.

        Args:
            feature_names (list, optional): [description]. If the model was trained using a pandas dataframe, the feature names are automatically
                extracted from the dataframe. If the model was trained using a numpy array, the feature names need to supplied.
            continuous_features (list, optional): [description]. Will change the plot accordingly.
        """        
        import matplotlib.pyplot as plt

        check_is_fitted(self, "is_fitted_")

        if feature_names is not None:
            self._feature_names = feature_names

        n_features = len(self.weights)

        n_cols = int(np.ceil( np.sqrt(n_features)))
        n_rows = int(np.floor(np.sqrt(n_features)))

        if n_cols * n_rows < n_features:
            n_rows += 1

        fig, ax = plt.subplots(n_rows, n_cols, **kwargs)
        for r in range(n_rows):
            for c in range(n_cols):
                ax[r, c].set_axis_off()

        fig.suptitle(f'Response mean: {self.y_mean:0.4f} | Iterations {self.iterations}')

        vmin = np.min([np.min(w) for w in self.weights]) - 1
        vmax = np.max([np.max(w) for w in self.weights]) - 1

        for feature_idx in range(n_features):
            ax_sub = ax[feature_idx // n_cols, feature_idx % n_cols]
            ax_sub.set_axis_on()

            # ax_sub.set_title(feature_names[feature_idx] if feature_names is not None else f'Feature {feature_idx}')
            if continuous_features is None:
                is_continuous = False
            else:
                if self._feature_names is not None:
                    is_continuous = self._feature_names[feature_idx] in continuous_features
                else:
                    is_continuous = feature_idx in continuous_features

            if self._feature_names is not None and '_X_' in self._feature_names[feature_idx]:
                self._plot_importance_interaction(ax_sub, feature_idx, vmin, vmax)
            else:
                self._plot_importance_categorical(ax_sub, feature_idx, vmin, vmax, is_continuous)

    @property
    def weights(self):
        return self._cpp.weights

    @property
    def y_mean(self):
        return self._cpp.y_mean

    @property
    def iterations(self):
        return self._cpp.iterations
