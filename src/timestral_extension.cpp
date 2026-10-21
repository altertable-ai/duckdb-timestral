#define DUCKDB_EXTENSION_MAIN

#include "timestral_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "timestral/forecast_aggregate.hpp"
#include "timestral/forecast_grammar.hpp"
#include "timestral/model_store.hpp"
#include "timestral/ort_predictor.hpp"

namespace duckdb {

static void TimestralVersionFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &children = StructVector::GetEntries(result);
	TimestralExtension extension;
	auto extension_version = extension.Version();
	children[0].SetValue(0, Value(extension_version.empty() ? "dev" : extension_version));
	children[1].SetValue(0, Value(timestral::OrtBlockPredictor::RuntimeVersion()));
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
}

static void LoadInternal(ExtensionLoader &loader) {
	auto version_type =
	    LogicalType::STRUCT({{"extension", LogicalType::VARCHAR}, {"onnxruntime", LogicalType::VARCHAR}});
	ScalarFunction version("timestral_version", {}, version_type, TimestralVersionFun);
	version.SetStability(FunctionStability::CONSISTENT);
	loader.RegisterFunction(version);

	timestral::RegisterModelStore(loader);
	timestral::RegisterForecastAggregate(loader);
	timestral::RegisterForecastGrammar(loader.GetDatabaseInstance());
}

void TimestralExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string TimestralExtension::Name() {
	return "timestral";
}

std::string TimestralExtension::Version() const {
#ifdef EXT_VERSION_TIMESTRAL
	return EXT_VERSION_TIMESTRAL;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(timestral, loader) {
	duckdb::LoadInternal(loader);
}
}
