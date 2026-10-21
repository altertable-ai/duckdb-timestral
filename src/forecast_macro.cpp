#include "timestral/forecast_aggregate.hpp"

#include "duckdb/catalog/default/default_table_functions.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

namespace duckdb {
namespace timestral {

// One row per (group, step). Column names arrive as strings and are matched exactly with COLUMNS lambdas,
// so 'store' does not also select 'store_id'. DuckDB allows one COLUMNS per expression, hence the inner rename.
static const DefaultTableMacro FORECAST_BY_MACRO = {DEFAULT_SCHEMA,
                                                    "timestral_forecast_by",
                                                    {"tbl", "group_by", "ts", "value", "horizon", nullptr},
                                                    {{"options", "NULL"}, {nullptr, nullptr}},
                                                    R"(
SELECT COLUMNS(lambda c: c = group_by), unnest(__timestral_forecast, recursive := true)
FROM (
    SELECT COLUMNS(lambda c: c = group_by), timestral_forecast(__timestral_ts, __timestral_value, horizon, options)
        AS __timestral_forecast
    FROM (
        SELECT COLUMNS(lambda c: c = group_by), COLUMNS(lambda c: c = ts) AS __timestral_ts,
               COLUMNS(lambda c: c = value) AS __timestral_value
        FROM query_table(tbl)
    )
    GROUP BY ALL
)
)"};

void RegisterForecastMacro(ExtensionLoader &loader) {
	auto info = DefaultTableFunctionGenerator::CreateTableMacroInfo(FORECAST_BY_MACRO);
	loader.RegisterFunction(*info);
}

} // namespace timestral
} // namespace duckdb
