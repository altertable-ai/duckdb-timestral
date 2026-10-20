#include "timestral/ort_predictor.hpp"

#include "timestral/model_config.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#ifdef _WIN32
#include "duckdb/common/windows_util.hpp"
#endif

#include <onnxruntime_cxx_api.h>

namespace duckdb {
namespace timestral {

static Ort::Env &SharedEnv() {
	// One environment per process; it owns ONNX Runtime's logging and global thread state.
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "timestral");
	return env;
}

struct OrtBlockPredictor::Impl {
	unique_ptr<Ort::Session> session;
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

OrtBlockPredictor::OrtBlockPredictor(const string &onnx_path, idx_t intra_op_threads) : impl(make_uniq<Impl>()) {
	try {
		Ort::SessionOptions options;
		options.SetIntraOpNumThreads(NumericCast<int>(intra_op_threads));
		options.SetInterOpNumThreads(1);
		options.SetExecutionMode(ORT_SEQUENTIAL);
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		// DuckDB already keeps its own threads busy; spinning ONNX Runtime workers would only steal CPU.
		options.AddConfigEntry("session.intra_op.allow_spinning", "0");
#ifdef _WIN32
		auto wide_path = WindowsUtil::UTF8ToUnicode(onnx_path.c_str());
		impl->session = make_uniq<Ort::Session>(SharedEnv(), wide_path.c_str(), options);
#else
		impl->session = make_uniq<Ort::Session>(SharedEnv(), onnx_path.c_str(), options);
#endif
	} catch (const Ort::Exception &ex) {
		throw IOException("timestral: could not load model '%s': %s", onnx_path, ex.what());
	}
}

OrtBlockPredictor::~OrtBlockPredictor() = default;

std::vector<float> OrtBlockPredictor::PredictBlock(const std::vector<float> &inputs,
                                                   const std::vector<uint8_t> &observed,
                                                   const std::vector<int64_t> &group_ids, idx_t rows, idx_t t,
                                                   idx_t horizon_patches) {
	D_ASSERT(inputs.size() == rows * t && observed.size() == rows * t && group_ids.size() == rows);
	const auto &config = DefaultModelConfig();
	auto observed_bool = make_unsafe_uniq_array<bool>(rows * t);
	for (idx_t i = 0; i < rows * t; i++) {
		observed_bool[i] = observed[i] != 0;
	}
	std::vector<float> horizon_token(horizon_patches, 0.0f);
	int64_t matrix_shape[] = {NumericCast<int64_t>(rows), NumericCast<int64_t>(t)};
	int64_t rows_shape[] = {NumericCast<int64_t>(rows)};
	int64_t horizon_shape[] = {NumericCast<int64_t>(horizon_patches)};

	try {
		auto &mem = impl->memory_info;
		Ort::Value values[] = {
		    Ort::Value::CreateTensor<float>(mem, const_cast<float *>(inputs.data()), inputs.size(), matrix_shape, 2),
		    Ort::Value::CreateTensor<bool>(mem, observed_bool.get(), rows * t, matrix_shape, 2),
		    Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t *>(group_ids.data()), rows, rows_shape, 1),
		    Ort::Value::CreateTensor<float>(mem, horizon_token.data(), horizon_patches, horizon_shape, 1)};
		const char *input_names[] = {"inputs", "observed", "group_ids", "horizon_token"};
		const char *output_names[] = {"quantiles"};
		auto outputs = impl->session->Run(Ort::RunOptions {nullptr}, input_names, values, 4, output_names, 1);
		auto expected = rows * horizon_patches * config.patch_size * NUM_QUANTILES;
		auto info = outputs[0].GetTensorTypeAndShapeInfo();
		if (info.GetElementCount() != expected) {
			throw InternalException("timestral: model returned %llu values, expected %llu", info.GetElementCount(),
			                        expected);
		}
		auto data = outputs[0].GetTensorData<float>();
		return std::vector<float>(data, data + expected);
	} catch (const Ort::Exception &ex) {
		throw InvalidInputException("timestral: model inference failed: %s", ex.what());
	}
}

string OrtBlockPredictor::RuntimeVersion() {
	return OrtGetApiBase()->GetVersionString();
}

} // namespace timestral
} // namespace duckdb
