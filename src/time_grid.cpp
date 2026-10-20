#include "timestral/time_grid.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/operator/subtract.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace duckdb {
namespace timestral {

static string StepToString(const GridStep &step) {
	return to_string(step.width);
}

static int64_t Distance(int64_t later, int64_t earlier) {
	int64_t result;
	if (!TrySubtractOperator::Operation(later, earlier, result)) {
		throw InvalidInputException("timestral: the time span of the series is too large");
	}
	return result;
}

static GridStep InferStep(const std::vector<std::pair<int64_t, double>> &rows) {
	std::map<int64_t, idx_t> gap_counts;
	for (idx_t i = 1; i < rows.size(); i++) {
		gap_counts[Distance(rows[i].first, rows[i - 1].first)]++;
	}
	GridStep step;
	idx_t best = 0;
	for (auto &entry : gap_counts) {
		// std::map iterates in increasing gap order, so ties keep the smallest gap.
		if (entry.second > best) {
			best = entry.second;
			step.width = entry.first;
		}
	}
	return step;
}

GridSeries BuildGrid(std::vector<std::pair<int64_t, double>> &rows, const GridStep *explicit_step,
                     idx_t context_length, const TimestampFormatter &format_ts) {
	GridSeries grid;
	if (rows.empty()) {
		return grid;
	}
	std::sort(rows.begin(), rows.end(),
	          [](const std::pair<int64_t, double> &a, const std::pair<int64_t, double> &b) { return a.first < b.first; });
	for (idx_t i = 1; i < rows.size(); i++) {
		if (rows[i].first == rows[i - 1].first) {
			throw InvalidInputException(
			    "timestral: duplicate timestamp %s in one series; aggregate first, e.g. with time_bucket",
			    format_ts(rows[i].first));
		}
	}
	if (explicit_step) {
		grid.step = *explicit_step;
	} else if (rows.size() < 2) {
		return grid;
	} else {
		grid.step = InferStep(rows);
	}
	if (grid.step.width <= 0) {
		throw InvalidInputException("timestral: freq must be positive");
	}

	grid.last_point = rows.back().first;
	auto width = grid.step.width;
	auto span = Distance(grid.last_point, rows.front().first) / width;
	idx_t window = static_cast<idx_t>(std::min<int64_t>(span + 1, static_cast<int64_t>(context_length)));
	grid.context.values.assign(window, std::nanf(""));
	bool any_observed = false;
	bool all_nonnegative = true;
	for (auto &row : rows) {
		auto distance = Distance(grid.last_point, row.first);
		if (distance % width != 0) {
			throw InvalidInputException("timestral: timestamp %s is not on the series grid (step %s); pass freq or "
			                            "use time_bucket",
			                            format_ts(row.first), StepToString(grid.step));
		}
		auto k = distance / width;
		// Values become float32 for the model, as in the reference; NaN and +/-inf are missing.
		auto value = static_cast<float>(row.second);
		if (std::isfinite(value)) {
			any_observed = true;
			all_nonnegative = all_nonnegative && value >= 0;
		}
		if (k < static_cast<int64_t>(window)) {
			grid.context.values[window - 1 - static_cast<idx_t>(k)] = value;
		}
	}
	grid.context.all_observed_nonnegative = any_observed && all_nonnegative;
	grid.ok = true;
	return grid;
}

int64_t GridPoint(int64_t last_point, const GridStep &step, int64_t k) {
	int64_t offset, result;
	if (!TryMultiplyOperator::Operation(step.width, k, offset) ||
	    !TryAddOperator::Operation(last_point, offset, result)) {
		throw OutOfRangeException("timestral: forecast timestamps are out of range");
	}
	return result;
}

} // namespace timestral
} // namespace duckdb
