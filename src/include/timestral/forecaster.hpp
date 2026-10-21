#pragma once

#include "timestral/model_config.hpp"
#include "timestral/ort_predictor.hpp"

#include <vector>

namespace duckdb {
namespace timestral {

//! Options of `TimestralForecaster.forecast`; the defaults are the reference defaults.
struct ForecastOptions {
	bool symmetric_averaging = true;
	bool clip_nonnegative = true;
	double quantile_real_cap_k = 1e4;
	double extreme_value_threshold = 1e30;
};

//! One univariate series on a regular grid, as the forecaster reads it.
struct SeriesContext {
	//! The last `<= context_length` grid steps, oldest first; NaN marks a missing value.
	std::vector<float> values;
	//! At least one observed value in the WHOLE series, and all of them >= 0 (reference `nonnegative_context`).
	bool all_observed_nonnegative = false;
};

//! C++ port of `TimestralForecaster.forecast` (mistralai/timestral @ a534bdbd) for one univariate series without
//! covariates. Writes `prediction_length * NUM_QUANTILES` floats (row-major, sorted along the quantile axis) to
//! `out`. Returns false when the series has no observed value or the forecast contains NaN (the reference raises).
bool ForecastSeries(BlockPredictor &predictor, const ModelConfig &config, const SeriesContext &series,
                    idx_t prediction_length, const ForecastOptions &options, std::vector<float> &out);

} // namespace timestral
} // namespace duckdb
