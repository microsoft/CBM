/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#pragma once

#include <vector>
#include <stdint.h>
#include <stdexcept>

namespace cbm
{
    class CBM
    {
        // n_features x max_bin[j] (jagged)
        std::vector<std::vector<double>> _f;
        double _y_mean;

        void update_y_hat_sum(
            std::vector<std::vector<uint64_t>> &y_hat_sum,
            std::vector<std::vector<uint8_t>> &x,
            size_t n_examples,
            size_t n_features);

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
            const uint8_t *x_max,
            double learning_rate_step_size,
            size_t max_iterations,
            size_t min_iterations_early_stopping,
            double epsilon_early_stopping,
            bool single_update_per_iteration);

        template <bool explain>
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

            // TODO: batch parallelization
            for (size_t i = 0; i < n_examples; i++)
            {
                double &y_hat_i = *(out_y_hat + i);

                for (size_t j = 0; j < n_features; j++)
                {
                    // TODO: simd gather?
                    uint8_t x_ij = *reinterpret_cast<const uint8_t *>(x_data + i * x_stride0 + j * x_stride1);
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
    };
}