#pragma once

#include "timestral/forecaster.hpp"

#include "duckdb/common/types/interval.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace duckdb {
namespace timestral {

//! Spacing of a series' grid. Integer series use `width`. Temporal series (timestamps as epoch micros) set
//! `is_interval`: a fixed duration has `width` micros, a calendar step has `interval.months > 0` and `width == 0`.
//! Calendar grid points fall on day `min(anchor_day, days in month)` (so month-end series stay on month ends),
//! at the time of day of the last row; BuildGrid sets `anchor_day` to the series' largest day of month.
struct GridStep {
	bool is_interval = false;
	int64_t width = 0;
	interval_t interval {0, 0, 0};
	int32_t anchor_day = 1;

	bool IsCalendar() const {
		return is_interval && interval.months != 0;
	}
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

struct GridOptions {
	//! The step to use; nullptr infers the most common positive gap (ties -> smallest).
	const GridStep *explicit_step = nullptr;
	//! Timestamps are epoch micros (an inferred step is reported as an INTERVAL).
	bool temporal = false;
	idx_t context_length = 0;
	TimestampFormatter format_ts;
	//! Appended to the off-grid error message when the step was inferred / given explicitly.
	string off_grid_hint;
	string explicit_step_hint;
};

//! Sorts `rows` (ts, value; NaN = missing) and lays them on a regular grid ending at the last row. Only the last
//! `context_length` grid steps are materialised; missing grid points are NaN; older rows only feed
//! `all_observed_nonnegative`. Throws InvalidInputException on duplicate or off-grid timestamps.
GridSeries BuildGrid(std::vector<std::pair<int64_t, double>> &rows, const GridOptions &options);

//! The grid point `k` steps after `last_point` (k may be negative). Calendar points are computed from the month
//! index directly, never accumulated.
int64_t GridPoint(int64_t last_point, const GridStep &step, int64_t k);

} // namespace timestral
} // namespace duckdb
