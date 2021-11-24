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
        uint8_t x_bytes_per_feature)
    {
        switch (x_bytes_per_feature)
        {
            case 1:
                fit_internal<uint8_t>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration);
                break;
            case 2:
                fit_internal<uint16_t>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration);
                break;
            case 4:
                fit_internal<uint32_t>(y, x_data, x_stride0, x_stride1, n_examples, n_features, y_mean, x_max, learning_rate_step_size, max_iterations, min_iterations_early_stopping, epsilon_early_stopping, single_update_per_iteration);
                break;
        }
    }
}