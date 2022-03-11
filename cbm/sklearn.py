# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import calendar
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in

from datetime import timedelta

# TODO
class TemporalSplit(TimeSeriesSplit):
    def __init__(self, step=timedelta(days=1), n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits)
        self.step = step
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def _create_date_ranges(self, start, end, step):
        start_ = start
        while start_ < end:
            end_ = start_ + step
            yield start_
            start_ = end_
            
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        
        date_range = list(self._create_date_ranges(X.index.min(), X.index.max(), self.step))
        n_samples =  len(date_range)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    np.where(np.logical_and(X.index >= date_range[train_end - self.max_train_size], X.index <= date_range[train_end - 1]))[0],
                    np.where(np.logical_and(X.index >= date_range[test_start], X.index <= date_range[test_start + test_size - 1]))[0]
                )
            else:
                yield (
                    np.where(X.index < date_range[train_end])[0],
                    np.where(np.logical_and(X.index >= date_range[test_start], X.index <= date_range[test_start + test_size - 1]))[0]
		)


# TODO: add unit test
class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, component = 'month' ):
        if component == 'weekday':
            self.categories_ = list(calendar.day_abbr)
            self.column_to_ordinal_ = lambda col: col.dt.weekday.values
        elif component == 'dayofyear':
            self.categories_ = list(range(1, 366))
            self.column_to_ordinal_ = lambda col: col.dt.dayofyear.values
        elif component == 'month':
            self.categories_ = list(calendar.month_abbr)
            self.column_to_ordinal_ = lambda col: col.dt.month.values
        else:
            raise ValueError('component must be either day or month')
        
        self.component = component
    
    def fit(self, X, y = None):
        self._validate_data(X, dtype="datetime64")

        return self
    
    def transform(self, X, y = None):
        return X.apply(self.column_to_ordinal_, axis=0)
