#pragma once

#include "timestral/forecaster.hpp"

#include "duckdb/common/types/interval.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace duckdb {
namespace timestral {

//! Spacing of a series' grid. `width` is in the series' integer units (micros for temporal types).
struct GridStep {
	bool is_interval = false;
	int64_t width = 0;
	interval_t interval {0, 0, 0};
};

struct GridSeries {
	SeriesContext context;
	//! The grid point of the last row; the forecast starts one step after it.
	int64_t last_point = 0;
	GridStep step;
	//! False when the grid cannot be built (fewer than 2 distinct timestamps and no explicit step).
	bool ok = false;
};

//! Formats a timestamp in the series' SQL type, for error messages.
using TimestampFormatter = std::function<string(int64_t)>;

//! Sorts `rows` (ts, value; NaN = missing) and lays them on a regular grid ending at the last row. The step is
//! `explicit_step` or the most common positive gap (ties -> smallest). Only the last `context_length` grid steps
//! are materialised; missing grid points are NaN. Throws InvalidInputException on duplicate or off-grid timestamps.
GridSeries BuildGrid(std::vector<std::pair<int64_t, double>> &rows, const GridStep *explicit_step,
                     idx_t context_length, const TimestampFormatter &format_ts);

//! The grid point `k` steps after `last_point`.
int64_t GridPoint(int64_t last_point, const GridStep &step, int64_t k);

} // namespace timestral
} // namespace duckdb
