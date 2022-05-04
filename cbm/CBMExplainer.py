# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd

from argparse import ArgumentTypeError
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from typing import List, Tuple, Union

from .sklearn import DateEncoder
from .CBM import CBM

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

# convenience registry to extract x-axis category labels from fitted transformers
TRANSFORMER_INVERTER_REGISTRY = {}

def transformer_inverter(transformer_class):
	def decorator(inverter_class):
		TRANSFORMER_INVERTER_REGISTRY[transformer_class] = inverter_class()
		return inverter_class

	return decorator

# return categories for each feature_names_in_
class TransformerInverter(ABC):
	@abstractmethod
	def get_category_names(self, transformer):
		pass

@transformer_inverter(DateEncoder)
class DateEncoderInverter(TransformerInverter):
	def get_category_names(self, transformer):
		# for each feature we return the set of category labels
		return list(map(lambda _: transformer.categories_, transformer.feature_names_in_))

@transformer_inverter(KBinsDiscretizer)
class KBinsDiscretizerInverter(TransformerInverter):
	def get_category_names(self, transformer):
		if transformer.encode != "ordinal":
			raise ValueError("Only ordinal encoding supported")

		# bin_edges is feature x bins
		def bin_edges_to_str(bin_edges: np.ndarray):
			return pd.IntervalIndex(pd.arrays.IntervalArray.from_breaks(np.concatenate([[-np.inf], bin_edges, [np.inf]])))

		return list(map(bin_edges_to_str, transformer.bin_edges_))

@transformer_inverter(OrdinalEncoder)
class OrdinalEncoderInverter(TransformerInverter):
	def get_category_names(self, transformer):
		return transformer.categories_

class CBMExplainerPlot:
	feature_index_: int
	feature_plots: List[dict]

	def __init__(self):
		self.feature_index_ = 0
		self.feature_plots_ = []

	def add_feature_plot(self, col_name: str, x_axis: List):
		self.feature_plots_.append({
			"col_name": col_name,
			"x_axis": x_axis,
			"feature_index": self.feature_index_,
		})

		# increment feature index (assume they are added in order)
		self.feature_index_ += 1

	def _plot_categorical(self, ax: plt.Axes, vmin: float, vmax: float, weights: np.ndarray, col_name: str, x_axis, **kwargs):
		cmap = plt.get_cmap("RdYlGn")

		is_continuous = isinstance(x_axis, pd.IntervalIndex)

		# plot positive/negative impact (so 1.x to 0.x)
		weights -= 1

		alpha = 1
		if is_continuous:
			ax.plot(range(len(weights)), weights)
			alpha = 0.3

		# normalize for color map
		weights_normalized = (weights - vmin) / (vmax - vmin)

		# draw bars
		ax.bar(range(len(weights)), weights, color=cmap(weights_normalized), edgecolor='black', alpha=alpha)

		ax.set_ylim(vmin-0.1, vmax+0.1)
		
		ax.set_ylabel('% change')

		ax.set_xlabel(col_name)

		if not is_continuous:
			ax.set_xticks(range(len(x_axis)))
			ax.set_xticklabels(x_axis, rotation=45)

	# TODO: support 2D interaction plots
	# def _plot_importance_interaction(self, ax, feature_idx: int, vmin: float, vmax: float):
	# 	import matplotlib.pyplot as plt

	# 	weights = np.array(self.weights[feature_idx]) - 0

	# 	cat_df = pd.DataFrame(
	# 	[(int(c.split('_')[-1]), int(c.split('_')[1]), i) for i, c in enumerate(self._feature_categories[feature_idx])],
	# 	columns=['f-1', 'f1', 'idx'])
	
	# 	cat_df.sort_values(['f-1', 'f1'], inplace=True)
	
	# 	cat_df_1d = cat_df.pivot(index='f0', columns='f1', values='idx')

	# 	# resort index by mean weight value
	# 	zi = np.array(weights)[cat_df_1d.to_numpy()]

	# 	sort_order = np.argsort(np.max(zi, axis=0))
	# 	cat_df_1d = cat_df_2d.reindex(cat_df_2d.index[sort_order])

	# 	# construct data matrices
	# 	xi = cat_df_1d.columns
	# 	yi = cat_df_1d.index
	# 	zi =  np.array(weights)[cat_df_1d.to_numpy()]
	
	# 	im = ax.imshow(zi, cmap=plt.get_cmap("RdYlGn"), aspect='auto', vmin=vmin, vmax=vmax)
	
	# 	cbar = ax.figure.colorbar(im, ax=ax)
	# 	cbar.ax.set_ylabel('% change', rotation=-91, va="bottom")
	
	# 	if self._feature_names is not None:
	# 	names = self._feature_names[feature_idx].split('_X_')
	# 	ax.set_ylabel(names[-1])
	# 	ax.set_xlabel(names[0])

	# 	# Show all ticks and label them with the respective list entries
	# 	ax.set_xticks(np.arange(len(xi)), labels=xi)
	# 	ax.set_yticks(np.arange(len(yi)), labels=yi)

	def plot(self, model: CBM, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
		num_plots = max(self.feature_plots_, key=lambda d: d["feature_index"])["feature_index"] + 1
		n_features = len(model.weights)

		if num_plots != n_features:
			raise ValueError(f"Missing plots for some features ({num_plots} vs {n_features})")

		# setup plot
		n_rows = num_plots
		n_cols = 1

		fig, ax = plt.subplots(n_rows, n_cols, **kwargs)

		for i in range(num_plots):
			ax[i].set_axis_off()

		fig.suptitle(f'Response mean: {model.y_mean:0.2f} | Iterations {model.iterations}')

		# extract weights from model
		weights = model.weights

		# find global min/max
		vmin = np.min([np.min(w) for w in weights]) - 1
		vmax = np.max([np.max(w) for w in weights]) - 1

		for feature_idx in range(n_features):
			ax_sub = ax[feature_idx]
			ax_sub.set_axis_on()

			feature_weights = np.array(weights[feature_idx])

			self._plot_categorical(ax_sub, vmin, vmax, feature_weights, **self.feature_plots_[feature_idx])

		plt.tight_layout()

		return fig, ax

class CBMExplainer:
	def __init__(self, pipeline: Pipeline):
		if not isinstance(pipeline, Pipeline):
			raise ArgumentTypeError("pipeline must be of type sklearn.pipeline.Pipeline")

		self.pipeline_ = pipeline

	def _plot_column_transformer(self, transformer: ColumnTransformer, plot: CBMExplainerPlot):
		# need to access transformers_ (vs transformers) to get the fitted transformer instance
		for (name, transformer, cols) in transformer.transformers_:
			# extension methods ;)
			transformer_inverter = TRANSFORMER_INVERTER_REGISTRY[type(transformer)]
			category_names = transformer_inverter.get_category_names(transformer)

			for (col_name, cat) in zip(cols, category_names):
				plot.add_feature_plot(col_name, cat)

	def plot_importance(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
		plot = CBMExplainerPlot()

		# iterate through pipeline
		for (name, component) in self.pipeline_.steps[0:-1]:
			if isinstance(component, ColumnTransformer):
				self._plot_column_transformer(component, plot)

		model = self.pipeline_.steps[-1][1]
		return plot.plot(model, **kwargs)
