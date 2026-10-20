// C++ port of `timestral/forecaster.py` and the parts of `timestral/scaler.py` it calls outside the network
// (mistralai/timestral @ a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff). Function names in comments refer to the
// reference. dtypes follow the reference: values and quantiles are float32, statistics are float64.

#include "timestral/forecaster.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

namespace duckdb {
namespace timestral {

namespace {

//! Packed `(R, T)` rows, row-major.
struct Rows {
	idx_t rows = 0;
	idx_t length = 0;
	std::vector<float> values;
	std::vector<uint8_t> observed;

	float *Values(idx_t r) {
		return values.data() + r * length;
	}
	uint8_t *Observed(idx_t r) {
		return observed.data() + r * length;
	}
};

//! Per-patch `(loc, scale)` of one row: `causal_patch_loc_scale` (each patch uses the causal float64
//! statistics at its last step, rounded to float32), with `stabilize_with_global = false`.
void CausalPatchLocScale(const float *x, const uint8_t *observed, idx_t length, const ModelConfig &config,
                         std::vector<float> &loc, std::vector<float> &scale) {
	auto ps = config.patch_size;
	loc.resize(length / ps);
	scale.resize(length / ps);
	double cum_w = 0, cum_x = 0, m2 = 0, previous_mean = 0;
	for (idx_t t = 0; t < length; t++) {
		double w = observed[t] ? 1.0 : 0.0;
		double v = static_cast<double>(x[t]);
		cum_w += w;
		cum_x += w * v;
		double denominator = std::max(cum_w, 1.0);
		double mean = cum_x / denominator;
		m2 += (v - previous_mean) * (v - mean) * w;
		previous_mean = mean;
		if (t % ps == ps - 1) {
			double variance = m2 / std::max(denominator - 1.0, 1.0);
			loc[t / ps] = static_cast<float>(mean);
			scale[t / ps] = static_cast<float>(std::max(std::sqrt(variance), config.minimum_scale));
		}
	}
}

struct Welford {
	double count = 0, mean = 0, m2 = 0;
};

//! `welford_combine`: merge state `b` into `a`; `(0, 0, 0)` is the identity.
Welford WelfordCombine(const Welford &a, const Welford &b) {
	Welford out;
	out.count = a.count + b.count;
	double w_safe = std::max(out.count, DBL_MIN);
	double delta = b.mean - a.mean;
	out.mean = a.mean + delta * b.count / w_safe;
	out.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / w_safe;
	if (out.count == 0) {
		out.mean = 0;
		out.m2 = 0;
	}
	return out;
}

//! `advancing_block_frames` for one row: per horizon patch `(loc, scale)` in float64. `sinh_median` is
//! `n_pred * patch_size`, the block's median in the last context patch's frame with the asinh undone.
void AdvancingBlockFrames(const float *context, const uint8_t *observed, idx_t length,
                          const std::vector<double> &sinh_median, idx_t n_pred, const ModelConfig &config,
                          std::vector<double> &locs, std::vector<double> &scales) {
	auto ps = config.patch_size;
	// Per-patch moments of the context, prefix-combined (the reference's Hillis-Steele scan, sequentially).
	Welford state;
	for (idx_t p = 0; p < length / ps; p++) {
		Welford patch;
		double sum = 0;
		for (idx_t j = 0; j < ps; j++) {
			if (observed[p * ps + j]) {
				patch.count += 1;
				sum += static_cast<double>(context[p * ps + j]);
			}
		}
		patch.mean = sum / std::max(patch.count, 1.0);
		for (idx_t j = 0; j < ps; j++) {
			if (observed[p * ps + j]) {
				double d = static_cast<double>(context[p * ps + j]) - patch.mean;
				patch.m2 += d * d;
			}
		}
		if (patch.count == 0) {
			patch.mean = 0;
			patch.m2 = 0;
		}
		state = WelfordCombine(state, patch);
	}
	double offset = state.mean;
	double clip = std::sqrt(static_cast<double>(FLT_MAX)) / 1e3;
	locs.resize(n_pred);
	scales.resize(n_pred);
	std::vector<double> folded(ps);
	for (idx_t h = 0; h < n_pred; h++) {
		double variance = std::max(state.m2 / std::max(state.count - 1.0, 1.0), 0.0);
		double scale = std::max(std::sqrt(variance), config.minimum_scale);
		locs[h] = state.mean;
		scales[h] = scale;
		double sum = 0;
		for (idx_t j = 0; j < ps; j++) {
			double v = sinh_median[h * ps + j] * scale + state.mean - offset;
			folded[j] = std::min(std::max(v, -clip), clip) + offset;
			sum += folded[j];
		}
		Welford block;
		block.count = static_cast<double>(ps);
		block.mean = sum / static_cast<double>(ps);
		for (idx_t j = 0; j < ps; j++) {
			double d = folded[j] - block.mean;
			block.m2 += d * d;
		}
		state = WelfordCombine(state, block);
	}
}

//! `clamp_nonfinite` over one quantile vector.
void ClampNonfinite(float *q) {
	float max_finite = -std::numeric_limits<float>::infinity();
	float min_finite = std::numeric_limits<float>::infinity();
	for (idx_t i = 0; i < NUM_QUANTILES; i++) {
		if (std::isfinite(q[i])) {
			max_finite = std::max(max_finite, q[i]);
			min_finite = std::min(min_finite, q[i]);
		}
	}
	for (idx_t i = 0; i < NUM_QUANTILES; i++) {
		if (q[i] == std::numeric_limits<float>::infinity()) {
			q[i] = max_finite;
		} else if (q[i] == -std::numeric_limits<float>::infinity()) {
			q[i] = min_finite;
		}
		if (std::isinf(q[i])) {
			q[i] = 0;
		}
	}
}

//! `_refine_block`: move a decoded block (`rows * n_pred*ps * Q`) from the anchor frame to the advancing frame.
void RefineBlock(std::vector<float> &quantiles, Rows &context, idx_t n_pred, const ModelConfig &config) {
	if (n_pred <= 1) {
		return;
	}
	auto ps = config.patch_size;
	auto block_len = n_pred * ps;
	std::vector<float> loc, scale;
	std::vector<double> sinh_median(block_len), locs, scales;
	for (idx_t r = 0; r < context.rows; r++) {
		CausalPatchLocScale(context.Values(r), context.Observed(r), context.length, config, loc, scale);
		float loc0 = loc.back();
		float scale0 = scale.back();
		float *q = quantiles.data() + r * block_len * NUM_QUANTILES;
		for (idx_t i = 0; i < block_len; i++) {
			sinh_median[i] = (static_cast<double>(q[i * NUM_QUANTILES + config.median_index]) - loc0) / scale0;
		}
		AdvancingBlockFrames(context.Values(r), context.Observed(r), context.length, sinh_median, n_pred, config, locs,
		                     scales);
		for (idx_t h = 0; h < n_pred; h++) {
			float loc_h = h == 0 ? loc0 : static_cast<float>(locs[h]);
			float scale_h = h == 0 ? scale0 : static_cast<float>(scales[h]);
			for (idx_t j = 0; j < ps; j++) {
				float *step = q + (h * ps + j) * NUM_QUANTILES;
				for (idx_t k = 0; k < NUM_QUANTILES; k++) {
					float sinh_norm = (step[k] - loc0) / scale0;
					step[k] = sinh_norm * scale_h + loc_h;
				}
				ClampNonfinite(step);
			}
		}
	}
}

//! `_block_ar_decode`: forecast block by block, feeding each block's median back as context.
//! Returns `rows * prediction_length * Q`.
std::vector<float> BlockArDecode(BlockPredictor &predictor, Rows context, idx_t prediction_length,
                                 const ModelConfig &config, const ForecastOptions &options) {
	auto ps = config.patch_size;
	std::vector<int64_t> group_ids(context.rows);
	for (idx_t r = 0; r < context.rows; r++) {
		group_ids[r] = static_cast<int64_t>(r);
	}
	std::vector<std::vector<float>> per_row(context.rows);
	idx_t remaining = (prediction_length + ps - 1) / ps;
	while (remaining > 0) {
		idx_t block = std::min(config.block_size, remaining);
		idx_t block_len = block * ps;
		auto quantiles = predictor.PredictBlock(context.values, context.observed, group_ids, context.rows,
		                                        context.length, block);
		RefineBlock(quantiles, context, block, config);
		for (idx_t r = 0; r < context.rows; r++) {
			auto begin = quantiles.begin() + static_cast<std::ptrdiff_t>(r * block_len * NUM_QUANTILES);
			per_row[r].insert(per_row[r].end(), begin,
			                  begin + static_cast<std::ptrdiff_t>(block_len * NUM_QUANTILES));
		}
		remaining -= block;
		if (remaining == 0) {
			break;
		}
		// Grow the context by the block's median (`replace_extreme_values`), all observed; then cap it.
		idx_t grown = context.length + block_len;
		idx_t keep = std::max(ps, (config.context_length / ps) * ps);
		idx_t new_length = std::min(grown, keep);
		idx_t drop = grown - new_length;
		Rows next;
		next.rows = context.rows;
		next.length = new_length;
		next.values.resize(next.rows * new_length);
		next.observed.resize(next.rows * new_length);
		for (idx_t r = 0; r < context.rows; r++) {
			const float *q = quantiles.data() + r * block_len * NUM_QUANTILES;
			for (idx_t i = drop; i < grown; i++) {
				float v;
				uint8_t m;
				if (i < context.length) {
					v = context.Values(r)[i];
					m = context.Observed(r)[i];
				} else {
					v = q[(i - context.length) * NUM_QUANTILES + config.median_index];
					if (!std::isfinite(v) || std::fabs(v) >= options.extreme_value_threshold) {
						v = 0;
					}
					m = 1;
				}
				next.Values(r)[i - drop] = v;
				next.Observed(r)[i - drop] = m;
			}
		}
		context = std::move(next);
	}
	std::vector<float> out(context.rows * prediction_length * NUM_QUANTILES);
	for (idx_t r = 0; r < context.rows; r++) {
		std::copy(per_row[r].begin(), per_row[r].begin() + static_cast<std::ptrdiff_t>(prediction_length * NUM_QUANTILES),
		          out.begin() + static_cast<std::ptrdiff_t>(r * prediction_length * NUM_QUANTILES));
	}
	return out;
}

//! `_cap_quantiles`: clamp to `[min - k*s, max + k*s]` of the observed context, `s` the last patch's scale.
void CapQuantiles(std::vector<float> &quantiles, Rows &rows, idx_t prediction_length, const ModelConfig &config,
                  double k) {
	std::vector<float> loc, scale;
	for (idx_t r = 0; r < rows.rows; r++) {
		float ctx_max = -std::numeric_limits<float>::infinity();
		float ctx_min = std::numeric_limits<float>::infinity();
		for (idx_t t = 0; t < rows.length; t++) {
			if (rows.Observed(r)[t]) {
				ctx_max = std::max(ctx_max, rows.Values(r)[t]);
				ctx_min = std::min(ctx_min, rows.Values(r)[t]);
			}
		}
		if (std::isinf(ctx_max)) {
			ctx_max = 0;
		}
		if (std::isinf(ctx_min)) {
			ctx_min = 0;
		}
		CausalPatchLocScale(rows.Values(r), rows.Observed(r), rows.length, config, loc, scale);
		float anchor = scale.back() * static_cast<float>(k);
		float lo = ctx_min - anchor;
		float hi = ctx_max + anchor;
		float *q = quantiles.data() + r * prediction_length * NUM_QUANTILES;
		for (idx_t i = 0; i < prediction_length * NUM_QUANTILES; i++) {
			if (q[i] < lo) {
				q[i] = lo;
			} else if (q[i] > hi) {
				q[i] = hi;
			}
		}
	}
}

} // namespace

bool ForecastSeries(BlockPredictor &predictor, const ModelConfig &config, const SeriesContext &series,
                    idx_t prediction_length, const ForecastOptions &options, std::vector<float> &out) {
	auto ps = config.patch_size;
	// `_as_context`: non-finite -> missing, stored as 0.
	auto total = series.values.size();
	std::vector<float> values(total);
	std::vector<uint8_t> observed(total);
	bool any_observed = false;
	for (idx_t t = 0; t < total; t++) {
		observed[t] = std::isfinite(series.values[t]) ? 1 : 0;
		values[t] = observed[t] ? series.values[t] : 0.0f;
		any_observed = any_observed || observed[t];
	}
	if (!any_observed || prediction_length == 0) {
		return false;
	}
	// `ffill_with_mask`: forward-fill unobserved positions; leading unobserved positions become 0.
	float last = 0;
	bool started = false;
	for (idx_t t = 0; t < total; t++) {
		if (observed[t]) {
			last = values[t];
			started = true;
		} else {
			values[t] = started ? last : 0.0f;
		}
	}
	// Keep the last `max_context` steps, then `pad_to_patch` on the left with unobserved zeros.
	idx_t start = total > config.context_length ? total - config.context_length : 0;
	idx_t kept = total - start;
	idx_t pad = (ps - kept % ps) % ps;
	idx_t num_rows = options.symmetric_averaging ? 2 : 1;
	Rows rows;
	rows.rows = num_rows;
	rows.length = kept + pad;
	rows.values.assign(num_rows * rows.length, 0.0f);
	rows.observed.assign(num_rows * rows.length, 0);
	for (idx_t r = 0; r < num_rows; r++) {
		// Row 1 is the mirrored pass: `-context` (symmetric averaging), decoded jointly as its own series.
		float sign = r == 0 ? 1.0f : -1.0f;
		for (idx_t i = 0; i < kept; i++) {
			rows.Values(r)[pad + i] = sign * values[start + i];
			rows.Observed(r)[pad + i] = observed[start + i];
		}
	}

	// `_forecast_raw` for both passes.
	auto quantiles = BlockArDecode(predictor, rows, prediction_length, config, options);
	if (options.quantile_real_cap_k > 0) {
		CapQuantiles(quantiles, rows, prediction_length, config, options.quantile_real_cap_k);
	}
	for (auto v : quantiles) {
		if (std::isnan(v)) {
			return false;
		}
	}

	auto count = prediction_length * NUM_QUANTILES;
	out.assign(quantiles.begin(), quantiles.begin() + static_cast<std::ptrdiff_t>(count));
	if (options.symmetric_averaging) {
		// Reflect the mirrored forecast back (`-Q(1 - p)`) and average.
		const float *mirrored = quantiles.data() + count;
		for (idx_t h = 0; h < prediction_length; h++) {
			for (idx_t k = 0; k < NUM_QUANTILES; k++) {
				float reflected = -mirrored[h * NUM_QUANTILES + (NUM_QUANTILES - 1 - k)];
				out[h * NUM_QUANTILES + k] = 0.5f * (out[h * NUM_QUANTILES + k] + reflected);
			}
		}
	}
	if (options.clip_nonnegative && series.all_observed_nonnegative) {
		for (auto &v : out) {
			v = std::max(v, 0.0f);
		}
	}
	return true;
}

} // namespace timestral
} // namespace duckdb
