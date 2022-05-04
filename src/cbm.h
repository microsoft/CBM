/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#pragma once

#include <vector>
#include <stdint.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <functional>
#include <iostream>
// #include <omp.h>
// #include <chrono>
// using namespace std::chrono;

namespace cbm
{
    float metric_RMSE(const uint32_t* y, const double* y_hat, size_t n_examples);
    float metric_SMAPE(const uint32_t* y, const double* y_hat, size_t n_examples);
    float metric_L1(const uint32_t* y, const double* y_hat, size_t n_examples);

    class CBM
    {
        // n_features x max_bin[j] (jagged)
        std::vector<std::vector<double>> _f;
        double _y_mean;

        size_t _iterations;
        std::vector<std::vector<uint32_t>> _bin_count;

        template<typename T>
        void update_y_hat(
            std::vector<double>& y_hat,
            std::vector<std::vector<T>> &x,
            size_t n_examples,
            size_t n_features)
        {
            // predict
            y_hat.assign(n_examples, _y_mean);

            #pragma omp parallel for schedule(static, 10000)
            for (size_t i = 0; i < n_examples; i++)
                for (size_t j = 0; j < n_features; j++)
                    y_hat[i] *= _f[j][x[j][i]];
        }

        template<typename T>
        void update_y_hat_sum(
            std::vector<double>& y_hat,
            std::vector<std::vector<uint64_t>> &y_hat_sum,
            std::vector<std::vector<T>> &x,
            size_t n_examples,
            size_t n_features)
        {
            update_y_hat(y_hat, x, n_examples, n_features);

            // reset y_hat_sum
            #pragma omp parallel for
            for (size_t j = 0; j < n_features; j++)
                std::fill(y_hat_sum[j].begin(), y_hat_sum[j].end(), 0);

            // compute y_hat and y_hat_sum
            #pragma omp parallel for
            for (size_t j = 0; j < n_features; j++)
                for (size_t i = 0; i < n_examples; i++)
                    // TODO: use log to stabilize?
                    y_hat_sum[j][x[j][i]] += y_hat[i];
        }

        template<typename T, bool enableBinCount>
        void fit_internal(
            const uint32_t *y,
            const char *x_data,
            size_t x_stride0,
            size_t x_stride1,
            size_t n_examples,
            size_t n_features,
            double y_mean,
            const uint32_t *x_max,
            double learning_rate_step_size,
            size_t max_iterations,
            size_t min_iterations_early_stopping,
            double epsilon_early_stopping,
            bool single_update_per_iteration,
            float (*metric)(const uint32_t*, const double*, size_t n_examples))
        {
            _y_mean = y_mean;

            // allocation
            std::vector<std::vector<T>> x(n_features);                // n_features x n_examples
            std::vector<std::vector<T>> g(n_features);                // n_features x max_bin[j] (jagged)
            std::vector<std::vector<uint64_t>> y_sum(n_features);     // n_features x max_bin[j] (jagged)
            std::vector<std::vector<uint64_t>> y_hat_sum(n_features); // n_features x max_bin[j] (jagged)
            std::vector<double> y_hat(n_examples);

            _f.resize(n_features);
            if (enableBinCount)
                _bin_count.resize(n_features);

            #pragma omp parallel for
            for (size_t j = 0; j < n_features; j++)
            {
                uint32_t max_bin = x_max[j];

                g[j].resize(max_bin + 1);
                _f[j].resize(max_bin + 1, 1);
                y_sum[j].resize(max_bin + 1);
                y_hat_sum[j].resize(max_bin + 1);

                if (enableBinCount)
                    _bin_count[j].resize(max_bin + 1, 0);

                // alloc and store columnar
                x[j].reserve(n_examples);
                for (size_t i = 0; i < n_examples; i++)
                {
                    // https://docs.python.org/3/c-api/buffer.html#complex-arrays
                    // strides are expressed in char not target type
                    T x_ij = *reinterpret_cast<const T *>(x_data + i * x_stride0 + j * x_stride1);
                    x[j].push_back(x_ij);

                    y_sum[j][x_ij] += y[i];

                    y_sum[j][x_ij] += y[i];

                    if (enableBinCount)
                        _bin_count[j][x_ij]++;
                }
            }

            // iterations
            double learning_rate = learning_rate_step_size;
            double rmse0 = std::numeric_limits<double>::infinity();

            for (_iterations = 0; _iterations < max_iterations; _iterations++, learning_rate += learning_rate_step_size)
            {
                // cap at 1
                if (learning_rate > 1)
                    learning_rate = 1;

                update_y_hat_sum(y_hat, y_hat_sum, x, n_examples, n_features);

                // compute g
                for (size_t j = 0; j < n_features; j++)
                {
                    for (size_t k = 0; k <= x_max[j]; k++)
                    {
                        // TODO: check if a bin is empty. might be better to remap/exclude the bins?
                        if (y_sum[j][k])
                        {
                            // improve stability
                            double g = (double)y_sum[j][k] / y_hat_sum[j][k]; // eqn. 2 (a)

                            // magic numbers found in Regularization section (worsen it quite a bit)
                            // double g = (2.0 * y_sum[j][k]) / (1.67834 * y_hat_sum[j][k]); // eqn. 2 (a)

                            if (learning_rate == 1)
                                _f[j][k] *= g;
                            else
                                _f[j][k] *= std::exp(learning_rate * std::log(g)); // eqn 2 (b) + eqn 4

                            if (!single_update_per_iteration) {
                                update_y_hat_sum(y_hat, y_hat_sum, x, n_examples, n_features);
                            }
                        }
                    }

                    // update_y_hat_sum after every feature
                    update_y_hat_sum(y_hat, y_hat_sum, x, n_examples, n_features);
                }

                // prediction
                update_y_hat(y_hat, x, n_examples, n_features);

                double rmse = metric(y, y_hat.data(), n_examples);

                // check for early stopping
                // TODO: expose minimum number of rounds
                if (_iterations > min_iterations_early_stopping &&
                    (rmse > rmse0 || (rmse0 - rmse) < epsilon_early_stopping))
                {
                    // TODO: record diagnostics?
                    // printf("early stopping %1.4f vs %1.4f after t=%d\n", rmse, rmse0, (int)t);
                    break;
                }
                rmse0 = rmse;
            }
        }

    public:
        CBM();
        CBM(const std::vector<std::vector<double>> &f, double y_mean);

        void fit(
            const uint32_t *y,
            const char *x_data,
            size_t x_stride0,
            size_t x_stride1,
            size_t n_examples,
            size_t n_features,
            double y_mean,
            const uint32_t *x_max,
            double learning_rate_step_size,
            size_t max_iterations,
            size_t min_iterations_early_stopping,
            double epsilon_early_stopping,
            bool single_update_per_iteration,
            uint8_t x_bytes_per_feature,
            float (*metric)(const uint32_t*, const double*, size_t n_examples),
            bool enable_bin_count);

        template <bool explain, typename T>
        void predict(
            const char *x_data,
            size_t x_stride0,
            size_t x_stride1,
            size_t n_examples,
            size_t n_features,
            double *out_data)
        {

            if (n_features != _f.size())
                throw std::runtime_error("Features need to match!");

            // column-wise oriented output data
            double *out_y_hat = out_data;
            std::fill(out_y_hat, out_y_hat + n_examples, _y_mean);

            #pragma omp parallel for schedule(static, 10000)
            for (size_t i = 0; i < n_examples; i++)
            {
                double &y_hat_i = *(out_y_hat + i);

                for (size_t j = 0; j < n_features; j++)
                {
                    // TODO: simd gather?
                    T x_ij = *reinterpret_cast<const T *>(x_data + i * x_stride0 + j * x_stride1);
                    y_hat_i *= _f[j][x_ij];

                    if (explain)
                    {
                        *(out_data + (j + 1) * n_examples + i) = _f[j][x_ij];
                    }
                }
            }
        }

        const std::vector<std::vector<double>> &get_weights() const;
        void set_weights(std::vector<std::vector<double>> &);

        float get_y_mean() const;
        void set_y_mean(float mean);

        size_t get_iterations() const;

        const std::vector<std::vector<uint32_t>> &get_bin_count() const;
    };
}