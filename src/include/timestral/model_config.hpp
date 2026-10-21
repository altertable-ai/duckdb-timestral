#pragma once

#include "duckdb/common/typedefs.hpp"

#include <array>

namespace duckdb {
namespace timestral {

//! Number of quantile levels every Timestral checkpoint predicts.
static constexpr idx_t NUM_QUANTILES = 21;

//! Constants shared by all four Timestral checkpoints (identical in their config.json files).
struct ModelConfig {
	idx_t patch_size = 32;
	//! Maximum number of horizon patches decoded per forward pass.
	idx_t block_size = 32;
	//! Context cap in time steps; the most recent steps are kept.
	idx_t context_length = 32768;
	double minimum_scale = 1e-6;
	bool use_asinh = true;
	std::array<float, NUM_QUANTILES> quantile_levels = {0.01f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f,
	                                                    0.35f, 0.40f, 0.45f, 0.50f, 0.55f, 0.60f, 0.65f,
	                                                    0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f, 0.99f};
	idx_t median_index = 10;
};

inline const ModelConfig &DefaultModelConfig() {
	static const ModelConfig config;
	return config;
}

} // namespace timestral
} // namespace duckdb
