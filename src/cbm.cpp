#include "cbm.h"

#include <algorithm>
#include <functional>
#include <cmath>

namespace cbm {
    void CBM::update_y_hat_sum(
        std::vector<std::vector<uint64_t>>& y_hat_sum,
        std::vector<std::vector<uint8_t>>& x,
        size_t n_examples,
        size_t n_features) {

        // reset y_hat_sum
        for (size_t j=0;j<n_features;j++)
            std::fill(y_hat_sum[j].begin(), y_hat_sum[j].end(), 0);

        // compute y_hat and y_hat_sum
        for (size_t i=0;i<n_examples;i++) {
            // TODO: parallelize & vectorize
            // TODO: use log to stabilize?
            auto y_hat_i = _y_mean;
            for (size_t j=0;j<n_features;j++)
                y_hat_i *= _f[j][x[j][i]];

            for (size_t j=0;j<n_features;j++)
                y_hat_sum[j][x[j][i]] += y_hat_i;
        }
    }

    std::vector<std::vector<double>>& CBM::get_weights() {
        return _f;
    }

    void CBM::fit(
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
        size_t min_iterations_early_stopping,
        double epsilon_early_stopping,
        bool single_update_per_iteration) {

        _y_mean = y_mean;

        // allocation
        std::vector<std::vector<uint8_t>> x(n_features);          // n_features x n_examples
        std::vector<std::vector<uint8_t>> g(n_features);          // n_features x max_bin[j] (jagged)
        std::vector<std::vector<uint64_t>> y_sum(n_features);     // n_features x max_bin[j] (jagged)
        std::vector<std::vector<uint64_t>> y_hat_sum(n_features); // n_features x max_bin[j] (jagged)
        std::vector<std::vector<uint16_t>> bin_count(n_features);

        _f.resize(n_features);

        for (size_t j=0;j<n_features;j++) {
            uint8_t max_bin = x_max[j];

            g[j].resize(max_bin + 1);
            _f[j].resize(max_bin + 1, 1);
            y_sum[j].resize(max_bin + 1);
            y_hat_sum[j].resize(max_bin + 1);
            bin_count[j].resize(max_bin + 1);

            // alloc and store columnar
            x[j].reserve(n_examples);
            for (size_t i=0;i<n_examples;i++) {
                // https://docs.python.org/3/c-api/buffer.html#complex-arrays
                // strides are expressed in char not target type
                uint8_t x_ij = *reinterpret_cast<const uint8_t*>(x_data + i * x_stride0 + j * x_stride1);
                x[j].push_back(x_ij);

                y_sum[j][x_ij] += y[i];
                bin_count[j][x_ij]++;
            }
        }

        // iterations
        double learning_rate = learning_rate_step_size;
        double rmse0 = std::numeric_limits<double>::infinity();
 
        for (size_t t=0;t<max_iterations;t++,learning_rate+=learning_rate_step_size) {
            // cap at 1
            if (learning_rate > 1)
                learning_rate = 1;

            update_y_hat_sum(y_hat_sum, x, n_examples, n_features);

            // compute g
            // TODO: parallelize
            for (size_t j=0;j<n_features;j++) {
                for (size_t k=0;k<x_max[j];k++) {

                    // TODO: check if a bin is empty. might be better to remap/exclude the bins?
                    if (y_sum[j][k]) {
                        // improve stability
                        double g = (double)y_sum[j][k] / y_hat_sum[j][k]; // eqn. 2 (a)

                        // magic numbers found in Regularization section (worsen it quite a bit)
                        // double g = (2.0 * y_sum[j][k]) / (1.67834 * y_hat_sum[j][k]); // eqn. 2 (a)

                        if (learning_rate == 1)
                            _f[j][k] *= g;
                        else
                            _f[j][k] *= std::exp(learning_rate * std::log(g)); // eqn 2 (b) + eqn 4

                        if (!single_update_per_iteration)
                            update_y_hat_sum(y_hat_sum, x, n_examples, n_features);
                    }
                }
            }

            // compute RMSE
            double rmse = 0;
            for (size_t i=0;i<n_examples;i++) {
                // TODO: batch parallelization
                auto y_hat_i = _y_mean;
                for (size_t j=0;j<n_features;j++) {
                    y_hat_i *= _f[j][x[j][i]];
                }
            
                rmse += (y_hat_i - y[i]) * (y_hat_i - y[i]);
            }
            rmse = std::sqrt(rmse);

            // check for early stopping
            // TODO: expose minimum number of rounds
            if (t > min_iterations_early_stopping && 
                (rmse > rmse0 || (rmse0 - rmse) < epsilon_early_stopping)) {
                // TODO: record diagnostics?
                // printf("early stopping %1.4f vs %1.4f after t=%d\n", rmse, rmse0, (int)t);
                break;
            }
            rmse0 = rmse; 
        }
    }
}