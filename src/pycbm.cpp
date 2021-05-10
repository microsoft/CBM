#include "pycbm.h"

namespace py = pybind11;

namespace cbm {

    void PyCBM::fit(
        py::buffer y_b,
        py::buffer x_b,
        double y_mean,
        py::buffer x_max_b,
        double learning_rate_step_size,
        size_t max_iterations,
        double epsilon_early_stopping) {

        // TODO: fix error messages
        py::buffer_info y_info = y_b.request();
        if (y_info.format != py::format_descriptor<uint32_t>::format())
            throw std::runtime_error("Incompatible format: expected a y array!");

        if (y_info.ndim != 1)
            throw std::runtime_error("Incompatible buffer dimension!");

        py::buffer_info x_info = x_b.request();
        if (x_info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a x array!");

        if (x_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        py::buffer_info x_max_info = x_max_b.request();
        if (x_max_info.format != py::format_descriptor<uint8_t>::format())
            throw std::runtime_error("Incompatible format: expected a x array!");

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
            epsilon_early_stopping);
    }

    std::vector<double> PyCBM::predict(py::buffer x_b) {
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

        return _cbm.predict(x_data, x_info.strides[0], x_info.strides[1], n_examples, n_features);
    }
};

PYBIND11_MODULE(cbm_cpp, m) {
    py::class_<cbm::PyCBM> estimator(m, "PyCBM");

    estimator.def(py::init([]() { return new cbm::PyCBM(); }))
             .def("fit", &cbm::PyCBM::fit)
             .def("predict", &cbm::PyCBM::predict);
}