import cbm_cpp
import numpy as np

class CBM:
    cpp: cbm_cpp.PyCBM

    def fit(self,
            y: np.ndarray, 
            x: np.ndarray,
            learning_rate_step_size = 1/100,
            max_iterations = 100,
            epsilon_early_stopping = 1e-3):
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
            epsilon_early_stopping)

    def predict(self, x: np.ndarray):
        return self.cpp.predict(x)
