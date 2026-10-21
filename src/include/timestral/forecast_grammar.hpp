#pragma once

namespace duckdb {
class DatabaseInstance;

namespace timestral {

//! Registers the "timestral" grammar extension (`FROM t TIMESTRAL_FORECAST (...)` and
//! `timestral_forecast(value ORDER BY ts HORIZON n ...)`) and activates it on every connection.
void RegisterForecastGrammar(DatabaseInstance &db);

} // namespace timestral
} // namespace duckdb
