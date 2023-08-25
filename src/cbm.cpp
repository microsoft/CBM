/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#include "cbm.h"

namespace cbm
{
    CBM::CBM() : _iterations(0)
    {
    }

    CBM::CBM(const std::vector<std::vector<double>> &f, double y_mean) :
        _f(f), _y_mean(y_mean), _iterations(0)
    {
    }

    const std::vector<std::vector<double>> &CBM::get_weights() const
    {
        return _f;
    }

    void CBM::set_weights(std::vector<std::vector<double>> &w)
    {
        _f = w;
    }

    float CBM::get_y_mean() const
    {
        return _y_mean;
    }

    void CBM::set_y_mean(float y_mean)
    {
        _y_mean = y_mean;
    }

    size_t CBM::get_iterations() const
    {
        return _iterations;
    }

    const std::vector<std::vector<uint32_t>> & CBM::get_bin_count() const {
        return _bin_count;
    }

    void CBM::fit(
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
        bool enable_bin_count)
    {
        switch (x_bytes_per_feature)
        {
            case 1:
                if (enable_bin_count)
                    fit_internal<uint8_t, true>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration, metric);
                else
                    fit_internal<uint8_t, false>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration, metric);
                break;
            case 2:
                if (enable_bin_count)
                    fit_internal<uint16_t, true>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration, metric);
                else
                    fit_internal<uint16_t, false>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration, metric);
                break;
            case 4:
                if (enable_bin_count)
                    fit_internal<uint32_t, true>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration, metric);
                else
                    fit_internal<uint32_t, false>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration, metric);
                break;
        }
    }

    float metric_RMSE(const uint32_t* y, const double* y_hat, size_t n_examples)
    {
        double rmse = 0;
        #pragma omp parallel for schedule(static, 10000) reduction(+: rmse)
        for (size_t i = 0; i < n_examples; i++)
            rmse += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
        
        return std::sqrt(rmse);
    }

    float metric_SMAPE(const uint32_t* y, const double* y_hat, size_t n_examples)
    {
        double smape = 0;
        #pragma omp parallel for schedule(static, 10000) reduction(+: smape)
        for (size_t i = 0; i < n_examples; i++) {
            if (y[i] == 0 && y_hat[i] == 0)
                continue;
            smape += std::abs(y[i] - y_hat[i]) / (y[i] + y_hat[i]);
        }
        
        return (200 * smape) / n_examples;
    }


    float metric_L1(const uint32_t* y, const double* y_hat, size_t n_examples) 
    {
        double l1 = 0;
        #pragma omp parallel for schedule(static, 10000) reduction(+: l1)
        for (size_t i = 0; i < n_examples; i++)
            l1 += std::abs(y_hat[i] - y[i]);
        
        return l1 / n_examples;
    }
}