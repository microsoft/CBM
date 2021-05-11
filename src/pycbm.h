#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cbm.h"

namespace cbm {

    namespace py = pybind11;

    class PyCBM {

        CBM _cbm;

    public:
        void fit(
            py::buffer y_b,
            py::buffer x_b,
            double y_mean,
            py::buffer x_max_b,
            double learning_rate_step_size,
            size_t max_iterations,
            size_t min_iterations_early_stopping,
            double epsilon_early_stopping,
            bool single_update_per_iteration);

        py::array_t<double> predict(py::buffer x_b, bool explain);

        std::vector<std::vector<double>>& get_weights();
    };
}