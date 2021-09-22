# Cyclic Boosting Machines

![Build](https://github.com/Microsoft/cbm/workflows/Build/badge.svg)
![Python](https://img.shields.io/pypi/pyversions/cyclicbm.svg)
[![codecov](https://codecov.io/gh/microsoft/CBM/branch/main/graph/badge.svg?token=VRppFx2o8v)](https://codecov.io/gh/microsoft/CBM)
[![PyPI version](https://badge.fury.io/py/cyclicbm.svg)](https://badge.fury.io/py/cyclicbm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Paper](https://img.shields.io/badge/academic-paper-7fdcf7)](https://arxiv.org/abs/2002.03425)

This is an efficient and Scikit-learn compatible implementation of the machine learning algorithm [Cyclic Boosting -- an explainable supervised machine learning algorithm](https://arxiv.org/abs/2002.03425), specifically for predicting count-data, such as sales and demand.

## Usage

```bash
pip install cyclicbm
```

```python
import cbm
import numpy as np

x_train: np.ndarray = ... # will be cast to uint8, so make sure you featurize before hand
y_train: np.ndarray = ... # will be cast to uint32

model = cbm.CBM()
model.fit(x_train, y_train)

x_test: np.numpy = ...
y_pred = model.predict(x_test)
```

## Explainability

The CBM model predicts by multiplying the global mean with each weight estimate for each bin and feature. Thus the weights can be interpreted as % increase or decrease from the global mean. e.g. a weight of 1.2 for the bin _Monday_ of the feature _Day-of-Week_ can be interpreted as a 20% increase of the target.

<img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i = \mu \cdot \product^{p}_{j=1} f^k_j"> with <img src="https://render.githubusercontent.com/render/math?math=k = \{x_{j,_i} \in b^k_j \}">

```python
model = cbm.CBM()
model.fit(x_train, y_train)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 
                         int(np.ceil(x_train.shape[1] / 2)),
                         figsize=(25, 20),
                         sharex=True)

for feature in np.arange(x_train.shape[1]):
    w = model.weights[feature]
    
    ax = axes[feature % 2, feature // 2]
    (ax.barh(x_train.iloc[:,feature].cat.categories.astype(str),
             np.array(w) - 1, # make sure it looks nice w/ bars go up and down from zero
             )
    )
    
    ax.set_title(x_train.columns[feature])
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
fig.tight_layout()
``` 

## Featurization

Categorical features can be passed as 0-based indices, with a maximum of 255 categories supported at the moment.

Continuous features need to be discretized. [pandas.qcut](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html) for equal-sized bins or [numpy.interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html) for equal-distant bins yield good results for us.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
