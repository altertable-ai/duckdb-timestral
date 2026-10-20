#include "timestral/forecast_aggregate.hpp"

#include "timestral/forecaster.hpp"
#include "timestral/model_store.hpp"
#include "timestral/time_grid.hpp"

#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

#include <cmath>

namespace duckdb {
namespace timestral {

static constexpr int64_t MAX_HORIZON = 100000;
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

struct ForecastBindData : public FunctionData {
	idx_t horizon = 0;
	LogicalType ts_type;
	shared_ptr<BlockPredictor> predictor;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ForecastBindData>();
		copy->horizon = horizon;
		copy->ts_type = ts_type;
		copy->predictor = predictor;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ForecastBindData>();
		return horizon == other.horizon && ts_type == other.ts_type && predictor == other.predictor;
	}
};

struct ForecastState {
	//! Buffered (ts, value) rows of one series; NaN marks a NULL value.
	std::vector<std::pair<int64_t, double>> *rows;
};

//===--------------------------------------------------------------------===//
// Timestamp conversion between the SQL type and the grid's int64 representation
//===--------------------------------------------------------------------===//
static int64_t ReadTimestamp(const UnifiedVectorFormat &format, const LogicalType &type, idx_t idx) {
	switch (type.id()) {
	case LogicalTypeId::INTEGER:
		return UnifiedVectorFormat::GetData<int32_t>(format)[idx];
	case LogicalTypeId::BIGINT:
		return UnifiedVectorFormat::GetData<int64_t>(format)[idx];
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
	default:
		throw InternalException("timestral: unsupported ts type %s", type.ToString());
	}
}

static string FormatTimestamp(const LogicalType &type, int64_t ts) {
	return to_string(ts);
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

static void FinalizeOne(ForecastState &state, const ForecastBindData &bind, Vector &result, idx_t result_idx) {
	if (!state.rows) {
		FlatVector::SetNull(result, result_idx, true);
		return;
	}
	const auto &config = DefaultModelConfig();
	auto grid = BuildGrid(*state.rows, nullptr, config.context_length,
	                      [&](int64_t ts) { return FormatTimestamp(bind.ts_type, ts); });
	std::vector<float> quantiles;
	if (!grid.ok || !ForecastSeries(*bind.predictor, config, grid.context, bind.horizon, ForecastOptions(), quantiles)) {
		FlatVector::SetNull(result, result_idx, true);
		return;
	}
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + bind.horizon);
	auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t h = 0; h < bind.horizon; h++) {
		auto row = offset + h;
		entries[0]->SetValue(row, TimestampValue(bind.ts_type, GridPoint(grid.last_point, grid.step,
		                                                                 static_cast<int64_t>(h + 1))));
		auto q = quantiles.data() + h * NUM_QUANTILES;
		FlatVector::GetData<double>(*entries[1])[row] = q[config.median_index];
		for (idx_t k = 0; k < NUM_QUANTILES; k++) {
			FlatVector::GetData<double>(*entries[2 + k])[row] = q[k];
		}
	}
	ListVector::SetListSize(result, offset + bind.horizon);
	FlatVector::GetData<list_entry_t>(result)[result_idx] = list_entry_t(offset, bind.horizon);
}

static void ForecastFinalize(Vector &states, AggregateInputData &aggr_input, Vector &result, idx_t count,
                             idx_t offset) {
	auto &bind = aggr_input.bind_data->Cast<ForecastBindData>();
	if (states.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		FinalizeOne(**ConstantVector::GetData<ForecastState *>(states), bind, result, 0);
		return;
	}
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	auto state_ptrs = FlatVector::GetData<ForecastState *>(states);
	for (idx_t i = 0; i < count; i++) {
		FinalizeOne(*state_ptrs[i], bind, result, i + offset);
	}
}

static void ForecastDestroy(Vector &states, AggregateInputData &, idx_t count) {
	auto state_ptrs = FlatVector::GetData<ForecastState *>(states);
	for (idx_t i = 0; i < count; i++) {
		delete state_ptrs[i]->rows;
		state_ptrs[i]->rows = nullptr;
	}
}

static unique_ptr<FunctionData> ForecastBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<ForecastBindData>();
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
	bind->ts_type = arguments[0]->return_type;
	// The model is resolved (and downloaded if needed) at bind time, so a missing model fails the query early.
	bind->predictor = ModelStore::Instance().GetPredictor(context, GetStringSetting(context, "timestral_model"));
	Function::EraseArgument(function, arguments, 2);
	function.return_type = ForecastEntryType(bind->ts_type);
	return std::move(bind);
}

void RegisterForecastAggregate(ExtensionLoader &loader) {
	AggregateFunctionSet set("timestral_forecast");
	for (auto &ts_type : {LogicalType::BIGINT, LogicalType::INTEGER}) {
		AggregateFunction function({ts_type, LogicalType::DOUBLE, LogicalType::BIGINT}, ForecastEntryType(ts_type),
		                           ForecastStateSize, ForecastInitialize, ForecastUpdate, ForecastCombine,
		                           ForecastFinalize, FunctionNullHandling::SPECIAL_HANDLING, nullptr, ForecastBind,
		                           ForecastDestroy);
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		set.AddFunction(function);
	}
	loader.RegisterFunction(set);
}

} // namespace timestral
} // namespace duckdb
