#pragma once

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace timestral {

//! One forward pass of the Timestral network (`TimestralModel.predict_block`, context-only path).
class BlockPredictor {
public:
	virtual ~BlockPredictor() = default;
	//! `inputs`/`observed` are `rows * t` row-major, `t` a multiple of the patch size; unobserved inputs are 0.
	//! Group ids must be distinct: the exported graph treats every row as its own series. Returns raw-space
	//! quantiles, `rows * (horizon_patches * patch_size) * NUM_QUANTILES` floats, sorted along the quantile axis.
	virtual std::vector<float> PredictBlock(const std::vector<float> &inputs, const std::vector<uint8_t> &observed,
	                                        const std::vector<int64_t> &group_ids, idx_t rows, idx_t t,
	                                        idx_t horizon_patches) = 0;
};

//! Runs an exported Timestral `.onnx` graph with ONNX Runtime (CPU). `PredictBlock` is thread-safe and runs
//! single-threaded: parallelism comes from DuckDB scheduling many forecasts at once.
class OrtBlockPredictor : public BlockPredictor {
public:
	explicit OrtBlockPredictor(const string &onnx_path);
	~OrtBlockPredictor() override;

	std::vector<float> PredictBlock(const std::vector<float> &inputs, const std::vector<uint8_t> &observed,
	                                const std::vector<int64_t> &group_ids, idx_t rows, idx_t t,
	                                idx_t horizon_patches) override;

	//! The linked ONNX Runtime version, e.g. "1.29.0".
	static string RuntimeVersion();

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace timestral
} // namespace duckdb
