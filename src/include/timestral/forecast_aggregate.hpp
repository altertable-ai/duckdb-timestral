#pragma once

#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
class ExtensionLoader;

namespace timestral {

//! Registers `timestral_forecast(ts, value, horizon [, options])`.
void RegisterForecastAggregate(ExtensionLoader &loader);

//! The value fields of a forecast entry: `forecast`, then `q01` to `q99`.
vector<string> ForecastValueFields();
//! The name of a forecast entry's time field for an input column name: the name itself, or `ts` when it is empty
//! or collides with a value field.
string ForecastTimeField(const string &input_name);

} // namespace timestral
} // namespace duckdb
