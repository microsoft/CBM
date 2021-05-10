#pragma once

#include <vector>
#include <stdint.h>

namespace cbm {
    class CBM {
        // n_features x max_bin[j] (jagged)
        std::vector<std::vector<double>> _f;       
        double _y_mean;

    public:
        void fit(
            const uint32_t* y,
            const char* x_data,
            size_t x_stride0,
            size_t x_stride1,
            size_t n_examples,
            size_t n_features,
            double y_mean,
            const uint8_t* x_max,
            double learning_rate_step_size,
            size_t max_iterations,
            double epsilon_early_stopping);

        std::vector<double> predict(
            const char* x_data,
            size_t x_stride0,
            size_t x_stride1,
            size_t n_examples,
            size_t n_features);
    };
}