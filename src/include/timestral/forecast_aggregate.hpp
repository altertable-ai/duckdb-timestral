#pragma once

namespace duckdb {
class ExtensionLoader;

namespace timestral {

//! Registers `timestral_forecast(ts, value, horizon)`.
void RegisterForecastAggregate(ExtensionLoader &loader);

//! Registers the `timestral_forecast_by(tbl, group_by, ts, value, horizon, options := NULL)` table macro.
void RegisterForecastMacro(ExtensionLoader &loader);

} // namespace timestral
} // namespace duckdb
