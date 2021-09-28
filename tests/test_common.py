# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sklearn.utils.estimator_checks import check_estimator

from cbm import CBM

def test_all_estimators():
    return check_estimator(CBM())