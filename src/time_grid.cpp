#include "timestral/time_grid.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/timestamp.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace duckdb {
namespace timestral {

static string StepToString(const GridStep &step) {
	if (step.is_interval) {
		return Interval::ToString(step.IsCalendar() ? step.interval : Interval::FromMicro(step.width));
	}
	return to_string(step.width);
}

static int64_t Distance(int64_t later, int64_t earlier) {
	int64_t result;
	if (!TrySubtractOperator::Operation(later, earlier, result)) {
		throw InvalidInputException("timestral: the time span of the series is too large");
	}
	return result;
}

static GridStep InferStep(const std::vector<std::pair<int64_t, double>> &rows, bool temporal) {
	std::map<int64_t, idx_t> gap_counts;
	for (idx_t i = 1; i < rows.size(); i++) {
		gap_counts[Distance(rows[i].first, rows[i - 1].first)]++;
	}
	GridStep step;
	step.is_interval = temporal;
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

static int64_t MonthIndex(int64_t micros, int32_t *day_out = nullptr) {
	int32_t year, month, day;
	Date::Convert(Timestamp::GetDate(timestamp_t(micros)), year, month, day);
	if (day_out) {
		*day_out = day;
	}
	return int64_t(year) * 12 + month - 1;
}

static int64_t TimeOfDay(int64_t micros) {
	date_t date;
	dtime_t time;
	Timestamp::Convert(timestamp_t(micros), date, time);
	return time.value;
}

//! Day `min(anchor_day, days in month)` of the month `month_index`, at `time_of_day`.
static int64_t MonthPoint(int64_t month_index, int32_t anchor_day, int64_t time_of_day) {
	auto year = month_index >= 0 ? month_index / 12 : -((-month_index + 11) / 12);
	auto month = static_cast<int32_t>(month_index - year * 12) + 1;
	if (year > NumericLimits<int32_t>::Maximum() || year < NumericLimits<int32_t>::Minimum()) {
		throw OutOfRangeException("timestral: forecast timestamps are out of range");
	}
	auto y = static_cast<int32_t>(year);
	auto day = MinValue<int32_t>(anchor_day, Date::MonthDays(y, month));
	date_t date;
	timestamp_t result;
	if (!Date::TryFromDate(y, month, day, date) || !Timestamp::TryFromDatetime(date, dtime_t(time_of_day), result)) {
		throw OutOfRangeException("timestral: forecast timestamps are out of range");
	}
	return result.value;
}

//! Number of steps from `ts` up to `last`, or false when `ts` is not on the grid.
static bool StepIndex(int64_t last, int64_t ts, const GridStep &step, int64_t &k) {
	if (!step.IsCalendar()) {
		auto distance = Distance(last, ts);
		if (distance % step.width != 0) {
			return false;
		}
		k = distance / step.width;
		return true;
	}
	auto ts_month = MonthIndex(ts);
	auto months = MonthIndex(last) - ts_month;
	if (months % step.interval.months != 0) {
		return false;
	}
	k = months / step.interval.months;
	return MonthPoint(ts_month, step.anchor_day, TimeOfDay(last)) == ts;
}

GridSeries BuildGrid(std::vector<std::pair<int64_t, double>> &rows, const GridOptions &options) {
	GridSeries grid;
	if (rows.empty()) {
		return grid;
	}
	std::sort(rows.begin(), rows.end(), [](const std::pair<int64_t, double> &a, const std::pair<int64_t, double> &b) {
		return a.first < b.first;
	});
	for (idx_t i = 1; i < rows.size(); i++) {
		if (rows[i].first == rows[i - 1].first) {
			throw InvalidInputException(
			    "timestral: duplicate timestamp %s in one series; aggregate first, e.g. with time_bucket",
			    options.format_ts(rows[i].first));
		}
	}
	if (options.explicit_step) {
		grid.step = *options.explicit_step;
	} else if (rows.size() < 2) {
		return grid;
	} else {
		grid.step = InferStep(rows, options.temporal);
	}
	if (!grid.step.IsCalendar() && grid.step.width <= 0) {
		throw InvalidInputException("timestral: EVERY must be positive");
	}
	if (grid.step.IsCalendar()) {
		grid.step.anchor_day = 1;
		for (auto &row : rows) {
			int32_t day;
			MonthIndex(row.first, &day);
			grid.step.anchor_day = MaxValue(grid.step.anchor_day, day);
		}
	}
	auto &hint = options.explicit_step ? options.explicit_step_hint : options.off_grid_hint;

	grid.last_point = rows.back().first;
	int64_t span;
	if (!StepIndex(grid.last_point, rows.front().first, grid.step, span)) {
		span = -1; // reported by the loop below
	}
	idx_t window = static_cast<idx_t>(
	    std::max<int64_t>(1, std::min<int64_t>(span + 1, static_cast<int64_t>(options.context_length))));
	grid.context.values.assign(window, std::nanf(""));
	bool any_observed = false;
	bool all_nonnegative = true;
	for (auto &row : rows) {
		int64_t k;
		if (!StepIndex(grid.last_point, row.first, grid.step, k)) {
			throw InvalidInputException("timestral: timestamp %s is not on the series grid (step %s); %s",
			                            options.format_ts(row.first), StepToString(grid.step), hint);
		}
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
	if (step.IsCalendar()) {
		int64_t months, month_index;
		if (!TryMultiplyOperator::Operation(int64_t(step.interval.months), k, months) ||
		    !TryAddOperator::Operation(MonthIndex(last_point), months, month_index)) {
			throw OutOfRangeException("timestral: forecast timestamps are out of range");
		}
		return MonthPoint(month_index, step.anchor_day, TimeOfDay(last_point));
	}
	int64_t offset, result;
	if (!TryMultiplyOperator::Operation(step.width, k, offset) ||
	    !TryAddOperator::Operation(last_point, offset, result)) {
		throw OutOfRangeException("timestral: forecast timestamps are out of range");
	}
	return result;
}

} // namespace timestral
} // namespace duckdb
