import cbm_cpp
import numpy as np

class CBM:
    cpp: cbm_cpp.PyCBM

    # TODO: scitkit-learn estimator?
    # def fit_pandas(self, df):
    # TODO include binning of continuous features

    def fit(self,
            y: np.ndarray, 
            x: np.ndarray,
            learning_rate_step_size:float = 1/100,
            max_iterations:int = 100,
            min_iterations_early_stopping:int = 20,
            epsilon_early_stopping:float = 1e-3,
            single_update_per_iteration:bool = True):
        y_mean = np.average(y)

        # determine max bin per categorical
        x_max = x.max(axis=0)

        if np.any(x_max > 255):
            raise ValueError("Maximum of 255 categories per features")

        self.cpp = cbm_cpp.PyCBM()
        self.cpp.fit(
            y.astype('uint32'), 
            x.astype('uint8'),
            y_mean, 
            x_max.astype('uint8'),
            learning_rate_step_size,
            max_iterations,
            min_iterations_early_stopping,
            epsilon_early_stopping,
            single_update_per_iteration)

    def predict(self, x: np.ndarray, explain: bool = False):
        return self.cpp.predict(x, explain)

    @property
    def weights(self):
        return self.cpp.weights