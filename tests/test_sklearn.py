# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV

import lightgbm as lgb
import timeit
import cbm

def test_nyc_bicycle_sklearn():
	# read data
	bic = pd.read_csv(
		'data/nyc_bb_bicyclist_counts.csv',
		parse_dates=['Date'])

	X_train = bic.drop('BB_COUNT', axis=1)
	y_train = bic['BB_COUNT']

	cats = make_column_transformer(
		# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
		# (OrdinalEncoder(dtype='int', handle_unknown='use_encoded_value', unknown_value=-1), # +1 in CBM code
		# ['store_nbr', 'item_nbr', 'onpromotion', 'family', 'class', 'perishable']),
	
		(cbm.DateEncoder('weekday', 'weekday'), ['Date']),
		(KBinsDiscretizer(n_bins=2, encode='ordinal'), ['HIGH_T', 'LOW_T']),
		(KBinsDiscretizer(n_bins=5, encode='ordinal'), ['PRECIP']),
	)

	cbm_model = cbm.CBM()
	pipeline = make_pipeline(cats, cbm_model)

	# print(pipeline.get_params().keys())

	cv = GridSearchCV(
		pipeline,
		param_grid={'columntransformer__kbinsdiscretizer-1__n_bins': np.arange(2, 15)},
		scoring=make_scorer(mean_squared_error, squared=False),
		cv=3
	)

	cv.fit(X_train, y_train)

	print(cv.cv_results_['mean_test_score'])
	print(cv.best_params_)