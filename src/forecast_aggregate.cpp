#include "timestral/forecast_aggregate.hpp"

#include "timestral/forecaster.hpp"
#include "timestral/model_store.hpp"
#include "timestral/time_grid.hpp"

#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression.hpp"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace duckdb {
namespace timestral {

static constexpr int64_t MAX_HORIZON = 100000;
static constexpr int64_t MICROS_PER_DAY = 86400000000LL;
static const char *const QUANTILE_FIELDS[NUM_QUANTILES] = {"q01", "q05", "q10", "q15", "q20", "q25", "q30",
                                                           "q35", "q40", "q45", "q50", "q55", "q60", "q65",
                                                           "q70", "q75", "q80", "q85", "q90", "q95", "q99"};

static LogicalType ForecastEntryType(const LogicalType &ts_type) {
	child_list_t<LogicalType> fields {{"ts", ts_type}, {"forecast", LogicalType::DOUBLE}};
	for (auto name : QUANTILE_FIELDS) {
		fields.emplace_back(name, LogicalType::DOUBLE);
	}
	return LogicalType::LIST(LogicalType::STRUCT(std::move(fields)));
}

static bool IsTemporal(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return true;
	default:
		return false;
	}
}

struct ForecastBindData : public FunctionData {
	idx_t horizon = 0;
	LogicalType ts_type;
	bool has_step = false;
	GridStep step;
	//! Series forecast in parallel per finalize call (DuckDB's `threads` setting at bind time).
	idx_t parallelism = 1;
	//! Model size as given (setting or options); kept to restore the predictor after deserialization.
	string model;
	shared_ptr<BlockPredictor> predictor;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ForecastBindData>();
		copy->horizon = horizon;
		copy->ts_type = ts_type;
		copy->has_step = has_step;
		copy->step = step;
		copy->parallelism = parallelism;
		copy->model = model;
		copy->predictor = predictor;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ForecastBindData>();
		return horizon == other.horizon && ts_type == other.ts_type && has_step == other.has_step &&
		       step.is_interval == other.step.is_interval && step.width == other.step.width &&
		       step.interval == other.step.interval && parallelism == other.parallelism && predictor == other.predictor;
	}
};

struct ForecastState {
	//! Buffered (ts, value) rows of one series; NaN marks a NULL value.
	std::vector<std::pair<int64_t, double>> *rows;
};

//===--------------------------------------------------------------------===//
// Timestamps: INTEGER/BIGINT as is, temporal types as epoch micros
//===--------------------------------------------------------------------===//
static int64_t ReadTimestamp(const UnifiedVectorFormat &format, const LogicalType &type, idx_t idx) {
	switch (type.id()) {
	case LogicalTypeId::INTEGER:
		return UnifiedVectorFormat::GetData<int32_t>(format)[idx];
	case LogicalTypeId::BIGINT:
		return UnifiedVectorFormat::GetData<int64_t>(format)[idx];
	case LogicalTypeId::DATE: {
		auto date = UnifiedVectorFormat::GetData<date_t>(format)[idx];
		if (!Date::IsFinite(date)) {
			throw InvalidInputException("timestral: infinite dates are not supported");
		}
		return Timestamp::FromDatetime(date, dtime_t(0)).value;
	}
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ: {
		auto ts = UnifiedVectorFormat::GetData<timestamp_t>(format)[idx];
		if (!Timestamp::IsFinite(ts)) {
			throw InvalidInputException("timestral: infinite timestamps are not supported");
		}
		return ts.value;
	}
	default:
		throw InternalException("timestral: unsupported ts type %s", type.ToString());
	}
}

static Value TimestampValue(const LogicalType &type, int64_t ts) {
	switch (type.id()) {
	case LogicalTypeId::INTEGER: {
		int32_t narrow;
		if (!TryCast::Operation(ts, narrow)) {
			throw OutOfRangeException("timestral: forecast timestamp %lld is out of range for INTEGER", ts);
		}
		return Value::INTEGER(narrow);
	}
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(ts);
	case LogicalTypeId::DATE:
		return Value::DATE(Timestamp::GetDate(timestamp_t(ts)));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(timestamp_t(ts));
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(timestamp_tz_t(ts));
	default:
		throw InternalException("timestral: unsupported ts type %s", type.ToString());
	}
}

static string OffGridHint(const LogicalType &type, bool explicit_step) {
	if (!IsTemporal(type)) {
		return explicit_step ? "use time_bucket to align the timestamps" : "pass freq or use time_bucket";
	}
	string hint = explicit_step ? "use time_bucket to align the timestamps"
	                            : "pass an explicit freq, e.g. {'freq': INTERVAL '1 month'} (calendar frequencies are "
	                              "not inferred), or use time_bucket";
	if (type.id() == LogicalTypeId::TIMESTAMP_TZ) {
		hint += ". TIMESTAMPTZ values are UTC instants, so local-time data shifts across daylight saving changes: "
		        "cast to TIMESTAMP in the local time zone first, or use time_bucket";
	}
	return hint;
}

//===--------------------------------------------------------------------===//
// Aggregate callbacks
//===--------------------------------------------------------------------===//
static idx_t ForecastStateSize(const AggregateFunction &) {
	return sizeof(ForecastState);
}

static void ForecastInitialize(const AggregateFunction &, data_ptr_t state) {
	reinterpret_cast<ForecastState *>(state)->rows = nullptr;
}

static void ForecastUpdate(Vector inputs[], AggregateInputData &aggr_input, idx_t input_count, Vector &states,
                           idx_t count) {
	auto &bind = aggr_input.bind_data->Cast<ForecastBindData>();
	UnifiedVectorFormat ts_format, value_format, state_format;
	inputs[0].ToUnifiedFormat(count, ts_format);
	inputs[1].ToUnifiedFormat(count, value_format);
	states.ToUnifiedFormat(count, state_format);
	auto values = UnifiedVectorFormat::GetData<double>(value_format);
	auto state_ptrs = UnifiedVectorFormat::GetData<ForecastState *>(state_format);
	for (idx_t i = 0; i < count; i++) {
		auto ts_idx = ts_format.sel->get_index(i);
		if (!ts_format.validity.RowIsValid(ts_idx)) {
			continue;
		}
		auto &state = *state_ptrs[state_format.sel->get_index(i)];
		if (!state.rows) {
			state.rows = new std::vector<std::pair<int64_t, double>>();
		}
		auto value_idx = value_format.sel->get_index(i);
		double value = value_format.validity.RowIsValid(value_idx) ? values[value_idx] : std::nan("");
		state.rows->emplace_back(ReadTimestamp(ts_format, bind.ts_type, ts_idx), value);
	}
}

static void ForecastCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sources = FlatVector::GetData<ForecastState *>(source);
	auto targets = FlatVector::GetData<ForecastState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[i];
		auto &dst = *targets[i];
		if (!src.rows) {
			continue;
		}
		if (!dst.rows) {
			dst.rows = new std::vector<std::pair<int64_t, double>>(*src.rows);
		} else {
			dst.rows->insert(dst.rows->end(), src.rows->begin(), src.rows->end());
		}
	}
}

//! The forecast of one series, computed before it is written to the result vector.
struct SeriesForecast {
	bool ok = false;
	GridSeries grid;
	std::vector<float> quantiles;
};

static void ComputeForecast(ForecastState &state, const ForecastBindData &bind, SeriesForecast &out) {
	if (!state.rows) {
		return;
	}
	const auto &config = DefaultModelConfig();
	GridOptions options;
	options.explicit_step = bind.has_step ? &bind.step : nullptr;
	options.temporal = IsTemporal(bind.ts_type);
	options.context_length = config.context_length;
	options.format_ts = [&](int64_t ts) {
		return TimestampValue(bind.ts_type, ts).ToString();
	};
	options.off_grid_hint = OffGridHint(bind.ts_type, false);
	options.explicit_step_hint = OffGridHint(bind.ts_type, true);
	out.grid = BuildGrid(*state.rows, options);
	out.ok = out.grid.ok &&
	         ForecastSeries(*bind.predictor, config, out.grid.context, bind.horizon, ForecastOptions(), out.quantiles);
}

static void WriteForecast(const SeriesForecast &forecast, const ForecastBindData &bind, Vector &result,
                          idx_t result_idx) {
	if (!forecast.ok) {
		if (result.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			ConstantVector::SetNull(result, true);
		} else {
			FlatVector::SetNull(result, result_idx, true);
		}
		return;
	}
	const auto &config = DefaultModelConfig();
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + bind.horizon);
	auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t h = 0; h < bind.horizon; h++) {
		auto row = offset + h;
		auto ts = GridPoint(forecast.grid.last_point, forecast.grid.step, static_cast<int64_t>(h + 1));
		entries[0]->SetValue(row, TimestampValue(bind.ts_type, ts));
		auto q = forecast.quantiles.data() + h * NUM_QUANTILES;
		FlatVector::GetData<double>(*entries[1])[row] = q[config.median_index];
		for (idx_t k = 0; k < NUM_QUANTILES; k++) {
			FlatVector::GetData<double>(*entries[2 + k])[row] = q[k];
		}
	}
	ListVector::SetListSize(result, offset + bind.horizon);
	FlatVector::GetData<list_entry_t>(result)[result_idx] = list_entry_t(offset, bind.horizon);
}

//! A process-wide pool of forecasting threads. DuckDB often finalizes all groups of a small hash table from a
//! single thread, so the series of one finalize call are spread over this pool. Its size (the core count) bounds
//! the threads created whatever the number of concurrent finalize calls. It is never destroyed: joining threads
//! during process or DLL teardown can hang.
class ForecastPool {
public:
	static ForecastPool &Instance() {
		static auto *pool = new ForecastPool();
		return *pool;
	}

	//! Runs `job` on the calling thread and on up to `helpers` pool threads, and returns once all have finished.
	//! `job` must return when there is no work left (several copies run concurrently).
	void Run(idx_t helpers, const std::function<void()> &job) {
		helpers = MinValue<idx_t>(helpers, threads.size());
		std::mutex done_lock;
		std::condition_variable done;
		idx_t running = helpers;
		{
			std::lock_guard<std::mutex> guard(lock);
			for (idx_t i = 0; i < helpers; i++) {
				queue.emplace_back([&]() {
					job();
					std::lock_guard<std::mutex> done_guard(done_lock);
					if (--running == 0) {
						done.notify_one();
					}
				});
			}
		}
		available.notify_all();
		job();
		std::unique_lock<std::mutex> done_guard(done_lock);
		done.wait(done_guard, [&] { return running == 0; });
	}

private:
	ForecastPool() {
		auto count = MaxValue<idx_t>(std::thread::hardware_concurrency(), 1);
		for (idx_t i = 0; i < count; i++) {
			try {
				threads.emplace_back([this] { WorkerLoop(); });
			} catch (std::exception &) {
				// Out of threads (e.g. a container pids limit): work with what was created; callers always help.
				break;
			}
		}
		for (auto &thread : threads) {
			thread.detach();
		}
	}

	void WorkerLoop() {
		while (true) {
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> guard(lock);
				available.wait(guard, [&] { return !queue.empty(); });
				task = std::move(queue.front());
				queue.pop_front();
			}
			task();
		}
	}

	std::mutex lock;
	std::condition_variable available;
	std::deque<std::function<void()>> queue;
	std::vector<std::thread> threads;
};

//! Forecasts `states[0..count)` with up to `bind.parallelism` threads, then writes them in order.
static void FinalizeStates(ForecastState **states, idx_t count, const ForecastBindData &bind, Vector &result,
                           idx_t offset) {
	auto threads = MaxValue<idx_t>(MinValue<idx_t>(bind.parallelism, count), 1);
	// Bounded chunks keep at most a few forecasts per thread in memory (a horizon can be 100,000 steps).
	auto chunk_size = threads * 4;
	std::vector<SeriesForecast> forecasts;
	for (idx_t chunk_start = 0; chunk_start < count; chunk_start += chunk_size) {
		auto chunk_count = MinValue<idx_t>(chunk_size, count - chunk_start);
		forecasts.assign(chunk_count, SeriesForecast());
		std::atomic<idx_t> next {0};
		std::atomic<bool> failed {false};
		std::mutex error_lock;
		ErrorData error;
		auto job = [&]() {
			while (!failed) {
				auto i = next++;
				if (i >= chunk_count) {
					return;
				}
				try {
					ComputeForecast(*states[chunk_start + i], bind, forecasts[i]);
				} catch (std::exception &ex) {
					std::lock_guard<std::mutex> guard(error_lock);
					if (!failed) {
						error = ErrorData(ex);
						failed = true;
					}
				}
			}
		};
		if (threads == 1 || chunk_count == 1) {
			job();
		} else {
			ForecastPool::Instance().Run(MinValue<idx_t>(threads, chunk_count) - 1, job);
		}
		if (failed) {
			error.Throw();
		}
		for (idx_t i = 0; i < chunk_count; i++) {
			WriteForecast(forecasts[i], bind, result, offset + chunk_start + i);
		}
	}
}

static void ForecastFinalize(Vector &states, AggregateInputData &aggr_input, Vector &result, idx_t count,
                             idx_t offset) {
	auto &bind = aggr_input.bind_data->Cast<ForecastBindData>();
	if (states.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		FinalizeStates(ConstantVector::GetData<ForecastState *>(states), 1, bind, result, 0);
		return;
	}
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	FinalizeStates(FlatVector::GetData<ForecastState *>(states), count, bind, result, offset);
}

static void ForecastDestroy(Vector &states, AggregateInputData &, idx_t count) {
	auto state_ptrs = FlatVector::GetData<ForecastState *>(states);
	for (idx_t i = 0; i < count; i++) {
		delete state_ptrs[i]->rows;
		state_ptrs[i]->rows = nullptr;
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//
static GridStep ParseFreq(const Value &freq, const LogicalType &ts_type) {
	GridStep step;
	if (!IsTemporal(ts_type)) {
		if (!freq.type().IsIntegral()) {
			throw BinderException("timestral: freq for integer time steps must be an integer");
		}
		step.width = freq.GetValue<int64_t>();
		if (step.width <= 0) {
			throw BinderException("timestral: freq must be positive");
		}
		return step;
	}
	if (freq.type().id() != LogicalTypeId::INTERVAL) {
		throw BinderException("timestral: freq for %s values must be an INTERVAL", ts_type.ToString());
	}
	auto interval = freq.GetValue<interval_t>();
	step.is_interval = true;
	step.interval = interval;
	if (interval.months != 0) {
		if (interval.days != 0 || interval.micros != 0) {
			throw BinderException("timestral: freq must be either a fixed duration or a whole number of months");
		}
		if (interval.months < 0) {
			throw BinderException("timestral: freq must be positive");
		}
		return step;
	}
	if (!Interval::TryGetMicro(interval, step.width) || step.width <= 0) {
		throw BinderException("timestral: freq must be positive");
	}
	if (ts_type.id() == LogicalTypeId::DATE && step.width % MICROS_PER_DAY != 0) {
		throw BinderException("timestral: freq for DATE values must be a whole number of days or months");
	}
	return step;
}

static unique_ptr<FunctionData> ForecastBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<ForecastBindData>();
	bind->ts_type = arguments[0]->return_type;
	bind->parallelism = MaxValue<idx_t>(NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads()), 1);
	auto &horizon_expr = *arguments[2];
	if (!horizon_expr.IsFoldable()) {
		throw BinderException("timestral: horizon must be a constant");
	}
	auto horizon = ExpressionExecutor::EvaluateScalar(context, horizon_expr);
	if (horizon.IsNull()) {
		throw BinderException("timestral: horizon must not be NULL");
	}
	auto horizon_value = horizon.GetValue<int64_t>();
	if (horizon_value < 1 || horizon_value > MAX_HORIZON) {
		throw BinderException("timestral: horizon must be between 1 and %lld, got %lld", MAX_HORIZON, horizon_value);
	}
	bind->horizon = static_cast<idx_t>(horizon_value);

	auto model = GetStringSetting(context, "timestral_model");
	if (arguments.size() > 3 && arguments[3]->return_type.id() == LogicalTypeId::SQLNULL) {
		// NULL options (the default of timestral_forecast_by) mean no options.
		Function::EraseArgument(function, arguments, 3);
	}
	if (arguments.size() > 3) {
		auto &options_expr = *arguments[3];
		if (!options_expr.IsFoldable() || options_expr.return_type.id() != LogicalTypeId::STRUCT) {
			throw BinderException("timestral: options must be a constant STRUCT, e.g. {'freq': INTERVAL '1 hour'}");
		}
		auto options = ExpressionExecutor::EvaluateScalar(context, options_expr);
		auto &fields = StructType::GetChildTypes(options.type());
		auto &values = StructValue::GetChildren(options);
		for (idx_t i = 0; i < fields.size(); i++) {
			auto key = StringUtil::Lower(fields[i].first);
			auto &value = values[i];
			if (key == "model") {
				if (!value.IsNull()) {
					model = value.ToString();
				}
			} else if (key == "freq") {
				if (!value.IsNull()) {
					bind->step = ParseFreq(value, bind->ts_type);
					bind->has_step = true;
				}
			} else {
				throw BinderException("timestral: unknown option '%s' (valid options: model, freq)", fields[i].first);
			}
		}
		Function::EraseArgument(function, arguments, 3);
	}
	// The model is resolved (and downloaded if needed) at bind time, so a missing model fails the query early.
	bind->model = model;
	bind->predictor = ModelStore::Instance().GetPredictor(context, model);
	Function::EraseArgument(function, arguments, 2);
	function.return_type = ForecastEntryType(bind->ts_type);
	return std::move(bind);
}

// Bind erases the constant horizon/options arguments, so plans are (de)serialized with the bind data instead of
// being re-bound.
static void ForecastSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                              const AggregateFunction &) {
	auto &bind = bind_data_p->Cast<ForecastBindData>();
	serializer.WriteProperty(100, "horizon", bind.horizon);
	serializer.WriteProperty(101, "ts_type", bind.ts_type);
	serializer.WriteProperty(102, "has_step", bind.has_step);
	serializer.WriteProperty(103, "step_is_interval", bind.step.is_interval);
	serializer.WriteProperty(104, "step_width", bind.step.width);
	serializer.WriteProperty(105, "step_months", bind.step.interval.months);
	serializer.WriteProperty(106, "step_days", bind.step.interval.days);
	serializer.WriteProperty(107, "step_micros", bind.step.interval.micros);
	serializer.WriteProperty(108, "model", bind.model);
}

static unique_ptr<FunctionData> ForecastDeserialize(Deserializer &deserializer, AggregateFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();
	auto bind = make_uniq<ForecastBindData>();
	bind->horizon = deserializer.ReadProperty<idx_t>(100, "horizon");
	bind->ts_type = deserializer.ReadProperty<LogicalType>(101, "ts_type");
	bind->has_step = deserializer.ReadProperty<bool>(102, "has_step");
	bind->step.is_interval = deserializer.ReadProperty<bool>(103, "step_is_interval");
	bind->step.width = deserializer.ReadProperty<int64_t>(104, "step_width");
	bind->step.interval.months = deserializer.ReadProperty<int32_t>(105, "step_months");
	bind->step.interval.days = deserializer.ReadProperty<int32_t>(106, "step_days");
	bind->step.interval.micros = deserializer.ReadProperty<int64_t>(107, "step_micros");
	bind->model = deserializer.ReadProperty<string>(108, "model");
	bind->parallelism = MaxValue<idx_t>(NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads()), 1);
	bind->predictor = ModelStore::Instance().GetPredictor(context, bind->model);
	function.return_type = ForecastEntryType(bind->ts_type);
	return std::move(bind);
}

void RegisterForecastAggregate(ExtensionLoader &loader) {
	AggregateFunctionSet set("timestral_forecast");
	for (auto &ts_type : {LogicalType::BIGINT, LogicalType::INTEGER, LogicalType::DATE, LogicalType::TIMESTAMP,
	                      LogicalType::TIMESTAMP_TZ}) {
		for (bool with_options : {false, true}) {
			vector<LogicalType> arguments {ts_type, LogicalType::DOUBLE, LogicalType::BIGINT};
			if (with_options) {
				arguments.push_back(LogicalType::ANY);
			}
			AggregateFunction function(arguments, ForecastEntryType(ts_type), ForecastStateSize, ForecastInitialize,
			                           ForecastUpdate, ForecastCombine, ForecastFinalize,
			                           FunctionNullHandling::SPECIAL_HANDLING, nullptr, ForecastBind, ForecastDestroy);
			function.name = "timestral_forecast";
			function.SetSerializeCallback(ForecastSerialize);
			function.SetDeserializeCallback(ForecastDeserialize);
			function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
			set.AddFunction(function);
		}
	}
	loader.RegisterFunction(set);
}

} // namespace timestral
} // namespace duckdb
