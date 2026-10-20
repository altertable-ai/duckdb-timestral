#include "timestral/model_store.hpp"

#include "timestral/model_manifest.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "mbedtls_wrapper.hpp"

#include <random>

namespace duckdb {
namespace timestral {

static constexpr const char *DEFAULT_MODEL_URL = "https://assets.altertable.ai/timestral/v1";
static constexpr const char *DEFAULT_CACHE_DIR = "~/.duckdb/timestral/v1";
static constexpr idx_t DOWNLOAD_CHUNK = 8 * 1024 * 1024;

string GetStringSetting(ClientContext &context, const string &name) {
	Value value;
	if (!context.TryGetCurrentSetting(name, value) || value.IsNull()) {
		return string();
	}
	return value.ToString();
}

ModelStore &ModelStore::Instance() {
	// Never destroyed: ONNX Runtime sessions must not outlive their Ort::Env, and tearing either down during process
	// or DLL unload is unsafe, so both live until the process exits.
	static auto *store = new ModelStore();
	return *store;
}

static_assert(sizeof(MODEL_MANIFEST) / sizeof(MODEL_MANIFEST[0]) <= 4, "ModelStore::entry_locks is too small");

static idx_t ManifestIndex(const ModelManifestEntry &entry) {
	return static_cast<idx_t>(&entry - MODEL_MANIFEST);
}

string ModelStore::NormalizeSize(const string &size) {
	auto lower = StringUtil::Lower(size);
	for (auto &entry : MODEL_MANIFEST) {
		if (lower == entry.size) {
			return lower;
		}
	}
	vector<string> sizes;
	for (auto &entry : MODEL_MANIFEST) {
		sizes.push_back(entry.size);
	}
	throw InvalidInputException("timestral: unknown model size '%s' (valid sizes: %s)", size,
	                            StringUtil::Join(sizes, ", "));
}

const ModelManifestEntry &ModelStore::Manifest(const string &normalized_size) {
	for (auto &entry : MODEL_MANIFEST) {
		if (normalized_size == entry.size) {
			return entry;
		}
	}
	throw InternalException("timestral: size '%s' is not normalized", normalized_size);
}

static string CacheDir(ClientContext &context) {
	auto dir = GetStringSetting(context, "timestral_cache_dir");
	if (dir.empty()) {
		dir = DEFAULT_CACHE_DIR;
	}
	return FileSystem::GetFileSystem(context).ExpandPath(dir);
}

string ModelStore::CachePath(ClientContext &context, const ModelManifestEntry &entry) {
	return FileSystem::GetFileSystem(context).JoinPath(CacheDir(context), entry.file_name);
}

string ModelStore::ModelUrl(ClientContext &context, const ModelManifestEntry &entry) {
	auto base = GetStringSetting(context, "timestral_model_url");
	if (base.empty()) {
		base = DEFAULT_MODEL_URL;
	}
	while (!base.empty() && base.back() == '/') {
		base.pop_back();
	}
	return base + "/" + entry.file_name;
}

bool ModelStore::IsCached(ClientContext &context, const ModelManifestEntry &entry) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto path = CachePath(context, entry);
	if (!fs.FileExists(path)) {
		return false;
	}
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	return NumericCast<uint64_t>(handle->GetFileSize()) == entry.file_bytes;
}

static string Sha256Hex(duckdb_mbedtls::MbedTlsWrapper::SHA256State &state) {
	char hex[64];
	state.FinishHex(hex);
	return string(hex, 64);
}

void ModelStore::VerifyCachedFile(FileSystem &fs, const string &path, const ModelManifestEntry &entry) {
	{
		lock_guard<mutex> guard(lock);
		if (verified_paths.count(path)) {
			return;
		}
	}
	duckdb_mbedtls::MbedTlsWrapper::SHA256State sha;
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto buffer = make_unsafe_uniq_array<data_t>(DOWNLOAD_CHUNK);
	while (true) {
		auto read = handle->Read(buffer.get(), DOWNLOAD_CHUNK);
		if (read <= 0) {
			break;
		}
		sha.AddBytes(buffer.get(), NumericCast<idx_t>(read));
	}
	handle.reset();
	auto actual = Sha256Hex(sha);
	if (actual != entry.sha256) {
		fs.TryRemoveFile(path);
		throw IOException("timestral: checksum mismatch for %s (expected %s, got %s); the corrupted file '%s' was "
		                  "removed, run the query again to download it",
		                  entry.file_name, entry.sha256, actual, path);
	}
	lock_guard<mutex> guard(lock);
	verified_paths.insert(path);
}

void ModelStore::Download(ClientContext &context, const ModelManifestEntry &entry, const string &target) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto url = ModelUrl(context, entry);
	if (StringUtil::StartsWith(url, "http://") || StringUtil::StartsWith(url, "https://")) {
		ExtensionHelper::TryAutoLoadExtension(context, "httpfs");
	}
	fs.CreateDirectoriesRecursive(CacheDir(context));
	// A name unique to this download, so concurrent downloads (other processes, other databases) never write to
	// the same file; the final rename is atomic.
	std::random_device random;
	auto tmp = StringUtil::Format("%s.tmp.%08x%08x", target, random(), random());
	duckdb_mbedtls::MbedTlsWrapper::SHA256State sha;
	try {
		auto source = fs.OpenFile(url, FileFlags::FILE_FLAGS_READ);
		auto sink = fs.OpenFile(tmp, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
		auto buffer = make_unsafe_uniq_array<data_t>(DOWNLOAD_CHUNK);
		while (true) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			auto read = source->Read(buffer.get(), DOWNLOAD_CHUNK);
			if (read <= 0) {
				break;
			}
			sha.AddBytes(buffer.get(), NumericCast<idx_t>(read));
			sink->Write(buffer.get(), NumericCast<idx_t>(read));
		}
		sink->Sync();
	} catch (std::exception &ex) {
		fs.TryRemoveFile(tmp);
		ErrorData error(ex);
		throw IOException("timestral: could not download %s: %s. To use a local copy instead, SET "
		                  "timestral_model_path = '/path/to/%s'",
		                  url, error.RawMessage(), entry.file_name);
	}
	auto actual = Sha256Hex(sha);
	if (actual != entry.sha256) {
		fs.TryRemoveFile(tmp);
		throw IOException("timestral: checksum mismatch for %s (expected %s, got %s) after downloading %s",
		                  entry.file_name, entry.sha256, actual, url);
	}
	fs.MoveFile(tmp, target);
	lock_guard<mutex> guard(lock);
	verified_paths.insert(target);
}

string ModelStore::ResolvePath(ClientContext &context, const ModelManifestEntry &entry) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto explicit_path = GetStringSetting(context, "timestral_model_path");
	if (!explicit_path.empty()) {
		explicit_path = fs.ExpandPath(explicit_path);
		if (!fs.FileExists(explicit_path)) {
			throw IOException("timestral: timestral_model_path '%s' does not exist", explicit_path);
		}
		return explicit_path;
	}
	auto path = CachePath(context, entry);
	if (IsCached(context, entry)) {
		VerifyCachedFile(fs, path, entry);
		return path;
	}
	Download(context, entry, path);
	return path;
}

shared_ptr<BlockPredictor> ModelStore::GetPredictor(ClientContext &context, const string &size) {
	auto &entry = Manifest(NormalizeSize(size));
	// One download/verification at a time per size; other sizes and cached models are not held up by it.
	lock_guard<mutex> entry_guard(entry_locks[ManifestIndex(entry)]);
	auto path = ResolvePath(context, entry);
	lock_guard<mutex> guard(lock);
	auto it = sessions.find(path);
	if (it != sessions.end()) {
		return it->second;
	}
	auto predictor = make_shared_ptr<OrtBlockPredictor>(path);
	sessions[path] = predictor;
	return predictor;
}

//===--------------------------------------------------------------------===//
// Settings
//===--------------------------------------------------------------------===//
static void ValidateModelSetting(ClientContext &context, SetScope scope, Value &parameter) {
	if (!parameter.IsNull()) {
		ModelStore::NormalizeSize(parameter.ToString());
	}
}

//===--------------------------------------------------------------------===//
// timestral_models()
//===--------------------------------------------------------------------===//
struct ModelsGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

static unique_ptr<FunctionData> ModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	names = {"size", "file_name", "url", "sha256", "file_bytes", "cached", "local_path"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::UBIGINT, LogicalType::BOOLEAN, LogicalType::VARCHAR};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> ModelsInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<ModelsGlobalState>();
}

static void ModelsFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ModelsGlobalState>();
	idx_t count = 0;
	constexpr idx_t total = sizeof(MODEL_MANIFEST) / sizeof(MODEL_MANIFEST[0]);
	while (state.offset < total && count < STANDARD_VECTOR_SIZE) {
		auto &entry = MODEL_MANIFEST[state.offset++];
		output.SetValue(0, count, Value(entry.size));
		output.SetValue(1, count, Value(entry.file_name));
		output.SetValue(2, count, Value(ModelStore::ModelUrl(context, entry)));
		output.SetValue(3, count, Value(entry.sha256));
		output.SetValue(4, count, Value::UBIGINT(entry.file_bytes));
		output.SetValue(5, count, Value::BOOLEAN(ModelStore::IsCached(context, entry)));
		output.SetValue(6, count, Value(ModelStore::CachePath(context, entry)));
		count++;
	}
	output.SetCardinality(count);
}

void RegisterModelStore(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("timestral_model", "Default Timestral model size: 5m, 28m, 123m or 209m",
	                          LogicalType::VARCHAR, Value("5m"), ValidateModelSetting);
	config.AddExtensionOption("timestral_model_url", "Base URL the Timestral .onnx files are downloaded from",
	                          LogicalType::VARCHAR, Value(DEFAULT_MODEL_URL));
	config.AddExtensionOption("timestral_cache_dir", "Directory downloaded Timestral models are cached in",
	                          LogicalType::VARCHAR, Value(DEFAULT_CACHE_DIR));
	config.AddExtensionOption("timestral_model_path",
	                          "Explicit local .onnx file to use instead of the download cache (no checksum check)",
	                          LogicalType::VARCHAR, Value(""));

	TableFunction models("timestral_models", {}, ModelsFunction, ModelsBind, ModelsInit);
	loader.RegisterFunction(models);
}

} // namespace timestral
} // namespace duckdb
