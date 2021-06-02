import pytest
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from interpret.glassbox import ExplainableBoostingRegressor

import lightgbm as lgb
import timeit
import cbm

def test_poisson_random():
    np.random.seed(42)

    # n = 1000 # test w/ 100, 1000, 10000, 100000
    # features = 2
    # bins = 2

    # y_base = np.random.poisson([[1, 3], [7, 20]], (n, features, bins))

    # x = np.random.randint(0, bins, (n, features), dtype='uint8')

    # y = np.zeros(n, dtype='uint32')
    # # TODO: figure out proper take, take_along_axis, ...
    # for i, idx in enumerate(x):
    #     y[i] = y_base[i, idx[0], idx[1]]


def test_nyc_bicycle():
    np.random.seed(42)

    # read data
    bic = pd.read_csv('data/nyc_bb_bicyclist_counts.csv')
    bic['Date'] = pd.to_datetime(bic['Date'])
    bic['Weekday'] = bic['Date'].dt.weekday

    y = bic['BB_COUNT'].values.astype('uint32')

    # train/test split
    split = int(len(y) * 0.8)
    train_idx = np.arange(0, split)
    test_idx  = np.arange(split + 1, len(y))

    y_train = y[train_idx]
    y_test  = y[test_idx]

    #### CBM

    # TODO: move to CBM.py and support pandas interface?
    # CBM can only handle categorical information
    # def histedges_equalN(x, nbin):
    #     npt = len(x)
    #     return np.interp(np.linspace(0, npt, nbin + 1),
    #                      np.arange(npt),
    #                      np.sort(x))

    # def histedges_equalN(x, nbin):
        # return pd.qcut(x, nbin)

    print()
    # some hyper-parameter che.. ehm tuning
    for bins in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        x = np.stack([
            bic['Weekday'].values,
            pd.qcut(bic['HIGH_T'], bins).cat.codes,
            pd.qcut(bic['LOW_T'], bins).cat.codes,
            pd.qcut(bic['PRECIP'], 5, duplicates='drop').cat.codes
        ],
            axis=1)\
                .astype('uint8')

        x_train = x[train_idx, ]
        x_test  = x[test_idx, ]

        start = timeit.timeit()

        # fit CBM model
        model = cbm.CBM(single_update_per_iteration=False)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)

        # y_pred_explain[:, 0]  --> predictions
        # y_pred_explain[:, 1:] --> explainations in-terms of multiplicative deviation from global mean
        y_pred_explain = model.predict(x_test, explain=True)

        # print("x", x_test[:3])
        # print("y", y_pred_explain[:3])
        # print("f", model.weights)

        # validate data predictions line up
        # print(np.all(y_pred[:, 0] == y_pred_explain[:,0]))

        print(f"CMB:          {mean_squared_error(y_test, y_pred, squared=False):1.4f} (train {mean_squared_error(y_train, y_pred_train, squared=False):1.4f}) bins={bins} {timeit.timeit() - start}sec")
        # print("weights", model.weights)
        # print(np.stack((y, y_pred))[:5,].transpose())   

    #### Poisson Regression

    # one-hot encode categorical
    start = timeit.timeit()

    x = bic['Weekday'].values.reshape((-1,1)).astype('uint8')

    enc = OneHotEncoder()
    enc.fit(x)
    x = enc.transform(x)

    x = np.hstack([x.todense(), bic[['HIGH_T', 'LOW_T', 'PRECIP']].values])

    clf = linear_model.PoissonRegressor()
    clf.fit(x[train_idx, ], y_train)

    y_pred = clf.predict(x[test_idx, ])
    print(f"Poisson Reg:  {mean_squared_error(y_test, y_pred, squared=False):1.4f} {timeit.timeit() - start}sec")
    # print(np.stack((y, y_pred))[:5,].transpose())   

    #### LightGBM

    start = timeit.timeit()

    # train_data = lgb.Dataset(x, label=y, categorical_feature=[0, 1])
    x = bic[['Weekday', 'HIGH_T', 'LOW_T', 'PRECIP']].values

    train_data = lgb.Dataset(x[train_idx, ], label=y_train, categorical_feature=[0])
    model = lgb.train({
        'objective': 'poisson',
        'metric': ['poisson', 'rmse'],
        'verbose': -1,
    }, train_data)

    y_pred = model.predict(x[test_idx, ])
    print(f"LightGBM Reg: {mean_squared_error(y_test, y_pred, squared=False):1.4f} {timeit.timeit() - start}sec")
    # print(np.stack((y, y_pred))[:5,].transpose())   


    #### EBM
    start = timeit.timeit()

    ebm = ExplainableBoostingRegressor(random_state=23, max_bins=8) #, outer_bags=25, inner_bags=25)
    ebm.fit(x[train_idx], y_train)

    y_pred = ebm.predict(x[test_idx,])
    print(f"EBM:          {mean_squared_error(y_test, y_pred, squared=False):1.4f} {timeit.timeit() - start}sec")