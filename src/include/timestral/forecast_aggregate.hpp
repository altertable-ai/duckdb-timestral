#pragma once

namespace duckdb {
class ExtensionLoader;

namespace timestral {

//! Registers `timestral_forecast(ts, value, horizon)`.
void RegisterForecastAggregate(ExtensionLoader &loader);

} // namespace timestral
} // namespace duckdb
