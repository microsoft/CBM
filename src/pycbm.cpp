/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#include "pycbm.h"

#include <sstream>

namespace py = pybind11;

namespace cbm {
    PyCBM::PyCBM() {
    }

    PyCBM::PyCBM(const std::vector<std::vector<double>>& f, double y_mean) : _cbm(f, y_mean) {
    }

    void PyCBM::fit(
        py::buffer y_b,
        py::buffer x_b,
        double y_mean,
        py::buffer x_max_b,
        double learning_rate_step_size,
        size_t max_iterations,
        size_t min_iterations_early_stopping,
        double epsilon_early_stopping,
        bool single_update_per_iteration) {

        // can't check compare just the format as linux returns I, windows returns L when using astype('uint32')
        // https://docs.python.org/3/library/struct.html#format-characters
        py::buffer_info y_info = y_b.request();
        if (!(y_info.itemsize == 4 && (y_info.format == "I" ||
                                       y_info.format == "H" ||
                                       y_info.format == "N" ||
                                       y_info.format == "B" ||
                                       y_info.format == "L"))) {
            std::ostringstream oss;
            oss << "y must be of type unsigned integer/long with 4 bytes! Must use y.astype('uint32'). "
                << "Format: " << y_info.format << " Size: " << y_info.itemsize;
            throw std::runtime_error("");
        }

        if (y_info.ndim != 1)
            throw std::runtime_error("y must be 1-dimensional!");

        py::buffer_info x_info = x_b.request();
        if (!(x_info.itemsize == 1 && (x_info.format == "I" ||
                                       x_info.format == "H" ||
                                       x_info.format == "N" ||
                                       x_info.format == "B" ||
                                       x_info.format == "L"))) {
            std::ostringstream oss;
            oss << "x must be of type unsigned integer/long with 1 bytes! Must use x.astype('uint8'). "
                << "Format: " << x_info.format << " Size: " << x_info.itemsize;

            throw std::runtime_error(oss.str().c_str());
        }

        if (x_info.ndim != 2)
            throw std::runtime_error("x must be 2-dimensional!");

        py::buffer_info x_max_info = x_max_b.request();
        if (x_max_info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a uint8 array for x_max!");

        if (x_max_info.ndim != 1)
            throw std::runtime_error("Incompatible buffer dimension!");

        if (y_info.shape[0] != x_info.shape[0])
            throw std::runtime_error("len(y) != len(x)");

        // data
        uint32_t* y = static_cast<uint32_t*>(y_info.ptr);
        uint8_t* x_max = static_cast<uint8_t*>(x_max_info.ptr);
        char* x_data = static_cast<char*>(x_info.ptr);

        // dimensions
        ssize_t n_examples = y_info.shape[0];
        ssize_t n_features = x_info.shape[1];

        _cbm.fit(
            y, 
            x_data,
            x_info.strides[0],
            x_info.strides[1],
            n_examples,
            n_features,
            y_mean,
            x_max,
            learning_rate_step_size,
            max_iterations,
            min_iterations_early_stopping,
            epsilon_early_stopping,
            single_update_per_iteration);
    }

    py::array_t<double> PyCBM::predict(py::buffer x_b, bool explain) {
        // TODO: fix error messages
        py::buffer_info x_info = x_b.request();
        if (x_info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a x array!");

        if (x_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        char* x_data = static_cast<char*>(x_info.ptr);

        // TODO: handle ssize_t vs size_t
        ssize_t n_examples = x_info.shape[0];
        ssize_t n_features = x_info.shape[1];

        py::array_t<double, py::array::f_style> out_data(
            {(int)n_examples, explain ? (int)(1 + n_features) : 1}
        );

        if (explain)
            _cbm.predict<true>(x_data, x_info.strides[0], x_info.strides[1], n_examples, n_features, out_data.mutable_data());
        else
            _cbm.predict<false>(x_data, x_info.strides[0], x_info.strides[1], n_examples, n_features, out_data.mutable_data());

        return out_data;
    }


    const std::vector<std::vector<double>>& PyCBM::get_weights() const {
        return _cbm.get_weights();
    }

    void PyCBM::set_weights(std::vector<std::vector<double>>& w) {
        _cbm.set_weights(w);
    }

    float PyCBM::get_y_mean() const {
        return _cbm.get_y_mean();
    }

    void PyCBM::set_y_mean(float y_mean) {
        _cbm.set_y_mean(y_mean);
    }
};

PYBIND11_MODULE(cbm_cpp, m) {
    py::class_<cbm::PyCBM> estimator(m, "PyCBM");

    estimator.def(py::init([]() { return new cbm::PyCBM(); }))
             .def("fit", &cbm::PyCBM::fit)
             .def("predict", &cbm::PyCBM::predict)
             .def_property("y_mean", &cbm::PyCBM::get_y_mean, &cbm::PyCBM::set_y_mean)
             .def_property("weights", &cbm::PyCBM::get_weights, &cbm::PyCBM::set_weights)
             .def(py::pickle(
                [](const cbm::PyCBM &p) { // __getstate__
                    /* Return a tuple that fully encodes the state of the object */
                    return py::make_tuple(p.get_weights(), p.get_y_mean());
                },
                [](py::tuple t) { // __setstate__
                    if (t.size() != 2)
                        throw std::runtime_error("Invalid state!");

                    /* Create a new C++ instance */
                    cbm::PyCBM p(t[0].cast<std::vector<std::vector<double>>>(),
                            t[1].cast<float>());

                    return p;
                }
             ));
}