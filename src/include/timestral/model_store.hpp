#pragma once

#include "timestral/ort_predictor.hpp"

#include "duckdb.hpp"

namespace duckdb {
class ExtensionLoader;

namespace timestral {

struct ModelManifestEntry;

//! Resolves, downloads, verifies and caches Timestral models, and keeps one ONNX Runtime session per
//! (file, thread count) for the whole process.
class ModelStore {
public:
	static ModelStore &Instance();

	//! Returns the predictor for `size`: `timestral_model_path` if set, otherwise the cached file (downloaded
	//! into `timestral_cache_dir` from `timestral_model_url` if missing). Cached and downloaded files are
	//! checked against the SHA-256 in the manifest.
	shared_ptr<BlockPredictor> GetPredictor(ClientContext &context, const string &size);

	//! '5M' -> '5m'; throws InvalidInputException for an unknown size.
	static string NormalizeSize(const string &size);
	static const ModelManifestEntry &Manifest(const string &normalized_size);

	//! `<cache_dir>/<file_name>` with `~` expanded.
	static string CachePath(ClientContext &context, const ModelManifestEntry &entry);
	static string ModelUrl(ClientContext &context, const ModelManifestEntry &entry);
	//! True if the cache holds a file with the final name and the manifest size (contents not checked).
	static bool IsCached(ClientContext &context, const ModelManifestEntry &entry);

private:
	string ResolvePath(ClientContext &context, const ModelManifestEntry &entry);
	void VerifyCachedFile(FileSystem &fs, const string &path, const ModelManifestEntry &entry);
	void Download(ClientContext &context, const ModelManifestEntry &entry, const string &target);

	mutex lock;
	//! Paths whose checksum was verified in this process.
	unordered_set<string> verified_paths;
	unordered_map<string, shared_ptr<BlockPredictor>> sessions;
};

//! Registers the `timestral_*` settings and the `timestral_models()` table function.
void RegisterModelStore(ExtensionLoader &loader);

string GetStringSetting(ClientContext &context, const string &name);

} // namespace timestral
} // namespace duckdb
