# Timestral DuckDB Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A DuckDB extension, `timestral`, that runs Mistral's Timestral zero-shot forecasting models on CPU through ONNX Runtime, exposed as the aggregate `timestral_forecast(ts, value, horizon [, options])`.

**Architecture:** An offline Python script exports each Timestral checkpoint's `predict_block` (the whole network, with scaling, masks and denormalisation) to ONNX. The extension downloads or caches the `.onnx` file, runs it with a statically linked ONNX Runtime, and ports Mistral's ~300-line `TimestralForecaster` wrapper (block rollout, refine frames, capping, symmetric averaging, clipping) to C++. A DuckDB aggregate does the per-series buffering and time-grid handling.

**Tech Stack:** C++17, DuckDB v1.5.6 extension template (`extension-ci-tools`), vcpkg `onnxruntime` (overlay ports), Python 3.12 + `uv` for the export (torch 2.14.1, onnx 1.23.2, onnxscript 0.7.2, onnxruntime 1.31.0, `timestral` @ `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`).

**Spec:** `docs/superpowers/specs/2026-10-20-timestral-extension-design.md`

## Spike results this plan relies on (2026-10-20, scratchpad, 5M model)

- `torch.onnx.export(dynamo=True, opset_version=20)` of `TimestralModel._predict_context_only` works with dynamic `rows`, `context = 32·patches` and `horizon_patches`. The only change needed is replacing `cummax` on a bool tensor with `cumsum(int32) > 0` (see Task 2). The masks stay in the graph, so the spec §5.2 fallback is not needed.
- ONNX Runtime vs PyTorch max relative error: about 2e-6 typically, 3.5e-5 at T=2048, H=32. The graph uses only standard ops (MatMul, Softmax, Unique, ScatterND, CumSum, Asinh, Trilu…). The 5M `.onnx` file is 22.8 MB.
- Latency for 5M, T=2016, H=96: about 2.3 ms per row with 1 intra-op thread, regardless of batch size (R=2: 4.6 ms; R=64: 130 ms). 8 threads only help large batches (1.4 ms per row).
- **Plan decision (deviation from spec §6.5, justified by the numbers):** no cross-series batching. Each series is one `Run` with 2 rows (the context and its negation for symmetric averaging, `group_ids = {0, 1}`). Parallelism comes from DuckDB running aggregate finalisation across partitions. This also removes the mixed-length padding risk.
- Prior art: community extension `anofox_tabfm` (MIT, github.com/DataZooDE/anofox-tabfm @ `ff5af557ca31a1ef0cf63c3345789f68d62c4dd7`) statically links vcpkg ONNX Runtime 1.29.0 in community CI. It uses overlay ports `vcpkg_ports/{onnxruntime,onnx,flatbuffers}` (the vcpkg-built `flatc` segfaults in DuckDB's AlmaLinux 8 image) and excludes `wasm_mvp;wasm_eh;wasm_threads;windows_amd64_mingw;linux_amd64_musl;osx_amd64`. On GCC 14 Linux it needs `-Wl,--allow-multiple-definition`.

## Global Constraints

- Extension name `timestral`; all SQL objects are prefixed `timestral_`.
- DuckDB v1.5.6 (submodule `duckdb` @ `069cc9f9`, `extension-ci-tools` @ `v1.5.6`).
- ONNX Runtime is linked statically from vcpkg, CPU execution provider only. No libtorch, no Python at query time.
- Remove the template's OpenSSL dependency. SHA-256 uses `duckdb_mbedtls::MbedTlsWrapper::SHA256State` (as `httpfs` does). HTTP goes through DuckDB's `FileSystem` with `httpfs` auto-loaded.
- Model sizes: `5m` (default), `28m`, `123m`, `209m`, matched case-insensitively.
- Hugging Face revisions: 5M `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`, 28M `49b26409ac87b5935539a5d8ffe7f397b720db47`, 123M `5a147287addda85f6acd8da3fb8858fe9af7e8f7`, 209M `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`.
- Asset URL: `https://assets.altertable.ai/timestral/v1/timestral-<size>.onnx` (files uploaded manually by the requester).
- Settings: `timestral_model` (VARCHAR, `'5m'`), `timestral_model_url` (VARCHAR, `'https://assets.altertable.ai/timestral/v1'`), `timestral_cache_dir` (VARCHAR, `'~/.duckdb/timestral/v1'`), `timestral_model_path` (VARCHAR, `''` meaning unset), `timestral_threads` (BIGINT, `1`).
- Model constants (identical in all 4 configs): patch 32, block_size 32, context_length 32768, minimum_scale 1e-6, use_asinh true, 21 quantile levels `0.01, 0.05, 0.10, …, 0.95, 0.99`, median index 10.
- Forecaster defaults match the reference: `symmetric_averaging = true`, `clip_nonnegative = true`, `quantile_real_cap_k = 1e4`, `extreme_value_threshold = 1e30`, `max_context = 32768`. Values go into the model as float32. Use float64 wherever the reference does (causal statistics, advancing frames).
- `horizon` must be in `[1, 100000]` (plan decision; the bound protects memory).
- Output field names: `ts, forecast, q01, q05, q10, q15, q20, q25, q30, q35, q40, q45, q50, q55, q60, q65, q70, q75, q80, q85, q90, q95, q99` (all DOUBLE except `ts`).
- Excluded platforms: `wasm_mvp;wasm_eh;wasm_threads;windows_amd64_mingw;linux_amd64_musl`, plus `osx_amd64` only if it fails to build (record why).
- Weights licence notice (Mistral Modified MIT, including the $20M monthly revenue restriction) goes in the README and next to the hosted files.

## Review Focus

1. **Non-finite values** (`'inf'::DOUBLE`, `'nan'::DOUBLE`) are treated as missing, like NULL, exactly as the reference's `isfinite` mask does. The query must not crash or emit NaN. Test in Task 5.
2. **Huge time gaps** (one row in 1970, the rest in 2026 at 1-second steps) must not materialise a giant grid. Only the last 32,768 grid steps are built, and the earlier rows only feed the non-negativity flag. Test in Task 6.
3. **Concurrent first use** (`SET threads = 8`, 2,000 groups, fresh process) must create exactly one ONNX session per size without races or crashes. Test in Task 5.
4. **Interrupted downloads** leave only `*.tmp` files, which are never treated as cached. Test in Task 4.
5. **TIMESTAMPTZ daily series crossing DST**: values are UTC instants, so 23 h and 25 h gaps fall off a 1-day grid. The error must say so and suggest casting to TIMESTAMP or using `time_bucket`. Test in Task 6.

---

## File Structure

```
CMakeLists.txt, Makefile, extension_config.cmake, vcpkg.json   (renamed; ORT wired)
vcpkg_ports/{onnxruntime,onnx,flatbuffers}/                     (overlay ports copied from anofox_tabfm, with attribution)
scripts/export/pyproject.toml, uv.lock                          (pinned export env)
scripts/export/export_onnx.py                                   (export + verify + manifest + header)
scripts/export/make_fixtures.py                                 (parity fixtures from TimestralForecaster)
src/timestral_extension.cpp, src/include/timestral_extension.hpp  (LoadInternal: settings, functions)
src/include/timestral/model_manifest.hpp                        (GENERATED by export_onnx.py; committed)
src/include/timestral/model_config.hpp                          (ModelConfig + constants)
src/model_store.cpp, src/include/timestral/model_store.hpp      (resolve / download / verify / cache / session cache)
src/ort_predictor.cpp, src/include/timestral/ort_predictor.hpp  (BlockPredictor interface + ONNX Runtime impl)
src/forecaster.cpp, src/include/timestral/forecaster.hpp        (C++ port of TimestralForecaster; no DuckDB types)
src/time_grid.cpp, src/include/timestral/time_grid.hpp          (sorting, step inference, grid, future timestamps)
src/forecast_aggregate.cpp, src/include/timestral/forecast_aggregate.hpp
src/forecast_macro.cpp                                          (timestral_forecast_by table macro)
test/sql/*.test (parity_<size>.test generated), test/data/parity/<size>/*.csv
description.yml, README.md, docs/                              (community-extension draft, user docs)
.github/workflows/MainDistributionPipeline.yml, .github/workflows/parity.yml
```

---

### Task 1: Rename the template and link ONNX Runtime

**Files:**
- Modify: everything `scripts/bootstrap-template.py timestral` touches (`CMakeLists.txt`, `Makefile`, `extension_config.cmake`, `src/*`, `test/sql/*`, workflows); delete `.github/workflows/ExtensionTemplate.yml` (the bootstrap does this)
- Modify: `vcpkg.json` (drop `openssl`; add `{"name": "onnxruntime", "default-features": false}`; overlay-ports `./vcpkg_ports` before `./extension-ci-tools/vcpkg_ports`)
- Create: `vcpkg_ports/{onnxruntime,onnx,flatbuffers}/` copied from anofox_tabfm @ `ff5af557`, plus `vcpkg_ports/README.md` crediting DataZoo GmbH (MIT) and stating why each overlay exists
- Modify: `CMakeLists.txt` (`find_package(onnxruntime CONFIG REQUIRED)`, link `onnxruntime::onnxruntime` into both targets, add the `include/onnxruntime` subdir as anofox's `cmake/ort.cmake` does, remove OpenSSL)
- Modify: `extension_config.cmake` (Linux `-Wl,--allow-multiple-definition` block, as in anofox; `LOAD_TESTS`)
- Test: `test/sql/timestral.test`

**Interfaces:**
- Produces: the scalar `timestral_version() -> STRUCT(extension VARCHAR, onnxruntime VARCHAR)`, where `onnxruntime` comes from `OrtGetApiBase()->GetVersionString()`. It's a smoke test that ORT is linked, and useful in bug reports.

- [ ] **Step 1:** `git submodule update --init --recursive`, then `python3 scripts/bootstrap-template.py timestral`. Commit this mechanical rename alone (`chore: bootstrap template as timestral`).
- [ ] **Step 2: Write the failing test** in `test/sql/timestral.test` (replace the waddle content):

```
require timestral

query I
SELECT timestral_version().onnxruntime LIKE '1.%';
----
true
```

- [ ] **Step 3:** `make && make test`. Expected: the test FAILS (function missing). The first vcpkg build of ORT takes a long time, so run it in the background.
- [ ] **Step 4:** Implement `timestral_version()`, the vcpkg and CMake changes, and remove the template's sample functions and OpenSSL.
- [ ] **Step 5:** `make && make test`. Expected: PASS. Also run `ldd`/`otool -L` on `build/release/extension/timestral/timestral.duckdb_extension` and confirm there's no `libonnxruntime` dynamic dependency.
- [ ] **Step 6:** Commit (`feat: link ONNX Runtime and add timestral_version()`).

---

### Task 2: ONNX export pipeline

**Files:**
- Create: `scripts/export/pyproject.toml`, `scripts/export/uv.lock`, `scripts/export/export_onnx.py`, `scripts/export/README.md`
- Create (generated, committed): `src/include/timestral/model_manifest.hpp`
- Output dir (gitignored): `scripts/export/out/` → `timestral-{5m,28m,123m,209m}.onnx`, `manifest.json`, `LICENSE` (Mistral's weights licence, fetched from the HF repo at the pinned revision)

**Interfaces:**
- Produces: the ONNX graph contract that Task 4 consumes:
  - inputs `inputs` float32 `[rows, T]` (T a multiple of 32; unobserved positions 0), `observed` bool `[rows, T]`, `group_ids` int64 `[rows]`, `horizon_token` float32 `[H]` (a dummy whose length is `horizon_patches`, 1..32)
  - output `quantiles` float32 `[rows, H*32, 21]`, raw space, non-decreasing along the last axis
- Produces: `model_manifest.hpp`:

```cpp
namespace duckdb { namespace timestral {
struct ModelManifestEntry { const char *size; const char *file_name; const char *sha256; uint64_t file_bytes; const char *hf_revision; };
static constexpr const char *ASSET_VERSION = "v1";
static constexpr ModelManifestEntry MODEL_MANIFEST[] = { /* 5m, 28m, 123m, 209m */ };
}}
```

- [ ] **Step 1:** Create the uv project pinning `torch==2.14.1`, `onnx==1.23.2`, `onnxscript==0.7.2`, `onnxruntime==1.31.0`, `numpy`, and `timestral @ git+https://github.com/mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`. Python 3.12. Run `uv lock`.
- [ ] **Step 2: Implement `export_onnx.py [--sizes 5m,28m,123m,209m] [--out out/]`.** For each size, load with `TimestralForecaster.from_pretrained("mistralai/Timestral-<SIZE>", revision=<pinned>)` and apply the export-only patch and wrapper below. Export with `dynamo=True`, `opset_version=20`, and `dynamic_shapes` `rows: Dim(min=2, max=4096)`, `T: 32*Dim(min=1, max=1024)`, `H: Dim(min=1, max=32)`, using input/output names as in the contract.

```python
import timestral.model as tm
def _key_validity_export(obs_per_patch, attend_missing):
    # == masks.compute_enc_key_validity; bool cummax has no ONNX lowering, (cumsum > 0) is equivalent.
    if not attend_missing:
        return obs_per_patch
    return obs_per_patch | (obs_per_patch.to(torch.int32).cumsum(dim=-1) > 0)
tm.compute_enc_key_validity = _key_validity_export

class PredictBlock(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, inputs, observed, group_ids, horizon_token):
        return self.m._predict_context_only(inputs, observed, group_ids, horizon_token.shape[0])
```

- [ ] **Step 3: Built-in verification.** Before writing anything, the script runs ORT against the *unpatched* PyTorch `model.predict_block` on these cases (R, T, H): (1, 32, 1), (2, 32, 1), (3, 160, 4), (8, 2048, 32), (16, 9600, 7), (2, 32768, 32), each with 5% random missing values and a leading missing third on row 0. It exits non-zero if any max-abs error exceeds `1e-4 * max|ref|`. Then it writes `manifest.json` (`size`, `file_name`, `sha256`, `file_bytes`, `hf_revision`, plus the model constants from Global Constraints), copies `LICENSE`, and regenerates `model_manifest.hpp` from it.
- [ ] **Step 4:** Run `uv run python -I export_onnx.py` (all four sizes; the 209M export takes minutes). Expected: verification lines report every error ≤ 1e-4, and `out/` contains 4 `.onnx` files.
- [ ] **Step 5:** Run it a second time and record in `scripts/export/README.md` whether the SHA-256 values are byte-identical (deterministic export). If they aren't, the README says the committed header and the uploaded files must come from the same run. The README also gives the upload instructions: copy `out/*` to `assets.altertable.ai/timestral/v1/`.
- [ ] **Step 6:** Structure `export_onnx.py` so a second graph (the covariate path, `_predict_window`) can be added later as another wrapper class and manifest entry (spec §6.4). Don't export it now.
- [ ] **Step 7:** Commit the script, lockfile, README and generated header (`feat: ONNX export pipeline for Timestral checkpoints`). **Tell the requester the files are ready to upload.**

---

### Task 3: Parity fixtures

**Files:**
- Create: `scripts/export/make_fixtures.py`
- Create (generated, committed): `test/data/parity/<size>/<case>_input.csv` (`series_id, t, value`; `t` is an integer step; an empty `value` means NULL) and `test/data/parity/<size>/<case>_expected.csv` (`series_id, step, q01 … q99`; `step` is 1..horizon), for sizes `5m, 28m, 123m, 209m`. The inputs are identical across sizes.
- Create (generated, committed): `test/sql/parity_<size>.test`, one `query` block per case with the literal horizon (a horizon from a subquery isn't foldable). `parity_5m.test` uses `require-env TIMESTRAL_MODEL_PATH`; the larger sizes use `require-env TIMESTRAL_MODEL_DIR` (a directory holding all exported files), so they're optional but runnable (spec §7).

**Interfaces:**
- Produces: fixtures and parity test files consumed in Task 5. The expected values come from `TimestralForecaster.forecast(context, prediction_length=horizon)` with default options, PyTorch (not ONNX), with context float32 and NULL→NaN.

- [ ] **Step 1: Implement `make_fixtures.py [--sizes 5m,28m,123m,209m]`** (writes `test/data/parity/<size>/` and `test/sql/parity_<size>.test`) with a fixed RNG seed (`20261020`) and these cases (horizon in brackets):
  - `short_20` (20 steps) [24]
  - `one_patch` (32) [32]
  - `non_multiple` (1,000, random walk) [96]
  - `long_truncate` (40,000, seasonal + trend) [64]
  - `horizon_1` [1]
  - `rollout_1500` (5,000 steps) [1,500]
  - `interior_nulls` (10% NULLs) [48]
  - `leading_nulls` (first 300 NULL) [48]
  - `trailing_nulls` (last 10 NULL) [48]
  - `constant` (all 7.0) [32]
  - `nonnegative_counts` (Poisson, with zeros) [64]
  - `large_magnitude` (values around 1e9) [32]
  - `negative` (mean −500) [32]
  - `multi_series` (5 series of lengths 50, 333, 1,024, 2,000 and 4,096 in one file) [40]
- [ ] **Step 2:** Run it. Expected: 14 input/expected pairs per size and 4 test files. Inputs are about 2 MB; if the total exceeds 10 MB, write the input CSVs once in `test/data/parity/inputs/` instead of per size.
- [ ] **Step 3:** Commit (`test: Timestral 5M parity fixtures`).

---

### Task 4: Model store and ONNX Runtime predictor

**Files:**
- Create: `src/include/timestral/model_config.hpp`, `src/include/timestral/ort_predictor.hpp`, `src/ort_predictor.cpp`, `src/include/timestral/model_store.hpp`, `src/model_store.cpp`
- Modify: `src/timestral_extension.cpp` (register settings and `timestral_models()`)
- Test: `test/sql/model_store.test`, `test/sql/model_store_local.test`

**Interfaces:**
- Consumes: `MODEL_MANIFEST` (Task 2); the graph contract (Task 2).
- Produces:

```cpp
struct ModelConfig { idx_t patch_size = 32, block_size = 32, context_length = 32768; double minimum_scale = 1e-6;
                     bool use_asinh = true; std::array<float, 21> quantile_levels; idx_t median_index = 10; };
const ModelConfig &DefaultModelConfig();

class BlockPredictor {
public:
	virtual ~BlockPredictor() = default;
	// inputs/observed: rows*T row-major, T % patch_size == 0. Returns rows * (horizon_patches*patch_size) * 21 floats.
	virtual std::vector<float> PredictBlock(const std::vector<float> &inputs, const std::vector<uint8_t> &observed,
	                                        const std::vector<int64_t> &group_ids, idx_t rows, idx_t t,
	                                        idx_t horizon_patches) = 0;
};
class OrtBlockPredictor : public BlockPredictor { public: OrtBlockPredictor(const string &onnx_path, idx_t intra_op_threads); /* thread-safe Run */ };

class ModelStore {   // one per DatabaseInstance, stored as ObjectCacheEntry
public:
	static ModelStore &Get(ClientContext &context);
	// Normalizes size ('5M' -> '5m'), resolves path (timestral_model_path > cache > download), verifies SHA-256
	// for cached/downloaded files, creates the session once per (path, threads) under a mutex.
	shared_ptr<BlockPredictor> GetPredictor(ClientContext &context, const string &size);
	static string NormalizeSize(const string &size); // throws InvalidInputException listing valid sizes
};
```

- Produces the table function `timestral_models() -> (size VARCHAR, file_name VARCHAR, url VARCHAR, sha256 VARCHAR, file_bytes UBIGINT, cached BOOLEAN, local_path VARCHAR)` (plan decision: makes offline prep and the store testable).
- Download: `ExtensionHelper::TryAutoLoadExtension(context, "httpfs")` → stream `<model_url>/<file_name>` to `<cache_dir>/<file_name>.tmp` while hashing → compare → `MoveFile` to the final name. A mismatch deletes the tmp file and throws `IOException("timestral: checksum mismatch for <file> …")`. A download failure throws an error that mentions `timestral_model_path`. Expand `~` with `FileSystem::ExpandPath`.
- Cache validation: a file is "cached" only if it has the final name and the right size. Its SHA-256 is verified once per process on first load (a bad cached file → the same checksum error, and the file is deleted).
- `timestral_model_path` set → use that file as is, with no checksum (for custom exports and tests).

- [ ] **Step 1: Write the failing tests.** `test/sql/model_store.test` doesn't need a model:

```
require timestral
query I
SELECT count(*), bool_and(url LIKE 'https://assets.altertable.ai/timestral/v1/timestral-%.onnx') FROM timestral_models();
----
4	true

statement ok
SET timestral_cache_dir = '__TEST_DIR__/cache';

query II
SELECT size, cached FROM timestral_models() WHERE size = '5m';
----
5m	false

# Review focus 4: a leftover .tmp is not a cached model
statement ok
COPY (SELECT 'x') TO '__TEST_DIR__/cache/timestral-5m.onnx.tmp' (FORMAT csv);

query I
SELECT cached FROM timestral_models() WHERE size = '5m';
----
false

# corrupted cache file of the right name -> checksum error on use (needs Task 5 aggregate; add the query there)

statement error
SET timestral_model = '7m';
----
Invalid Input Error: timestral: unknown model size '7m'
```

  `test/sql/model_store_local.test` checks that `require-env TIMESTRAL_MODEL_PATH` and `SET timestral_model_path = '${TIMESTRAL_MODEL_PATH}'` succeed (session creation is exercised by Task 5).
  For the unknown-size error, validate the setting with a set-callback that calls `NormalizeSize`. Create the cache directory on demand.
- [ ] **Step 2:** `make test`. Expected: FAIL (no `timestral_models`).
- [ ] **Step 3:** Implement the interfaces above. ORT session options: intra-op threads = `timestral_threads`, inter-op 1, `ORT_SEQUENTIAL`, graph optimisation `ORT_ENABLE_ALL`. Use one shared `Ort::Env` per process.
- [ ] **Step 4:** `make test`. Expected: PASS.
- [ ] **Step 5:** Commit (`feat: model store, download cache and ONNX Runtime predictor`).

---

### Task 5: Forecaster port and `timestral_forecast` aggregate (integer time steps)

**Files:**
- Create: `src/include/timestral/forecaster.hpp`, `src/forecaster.cpp`, `src/include/timestral/time_grid.hpp`, `src/time_grid.cpp`, `src/include/timestral/forecast_aggregate.hpp`, `src/forecast_aggregate.cpp`
- Test: `test/sql/parity_5m.test`, `test/sql/forecast_basic.test`

**Interfaces:**
- Consumes: `BlockPredictor`, `ModelConfig`, `ModelStore::GetPredictor` (Task 4); fixtures (Task 3).
- Produces:

```cpp
struct ForecastOptions { bool symmetric_averaging = true; bool clip_nonnegative = true;
                         double quantile_real_cap_k = 1e4; double extreme_value_threshold = 1e30; };
struct SeriesContext { std::vector<float> values;  // last <= context_length grid steps; NaN = missing
                       bool all_observed_nonnegative; };  // over the WHOLE series (reference nonnegative_context)
// Returns false when the series has no observed value or the forecast contains NaN (-> SQL NULL).
// out: prediction_length * 21 floats, row-major, quantile axis sorted.
bool ForecastSeries(BlockPredictor &predictor, const ModelConfig &config, const SeriesContext &series,
                    idx_t prediction_length, const ForecastOptions &options, std::vector<float> &out);

struct GridStep { bool is_interval; int64_t width; interval_t interval; };  // Task 5 uses width only
struct GridSeries { SeriesContext context; int64_t last_point; GridStep step; bool ok; };
// rows: (ts as int64, value or NaN), unsorted. Throws InvalidInputException on duplicate / off-grid ts.
GridSeries BuildGrid(std::vector<std::pair<int64_t, double>> &rows, const GridStep *explicit_step, idx_t context_length);
int64_t GridPoint(int64_t last_point, const GridStep &step, int64_t k);
```

- SQL: `timestral_forecast(ts BIGINT|INTEGER, value <numeric>, horizon INTEGER) -> LIST(STRUCT(ts <ts type>, forecast DOUBLE, q01 … q99 DOUBLE))`. `horizon` must be a foldable constant in `[1, 100000]`, otherwise a `BinderException`. The state buffers `(int64 ts, double value)`. Combine appends. Finalize builds the grid → `ForecastSeries` (model from the `timestral_model` setting) → list, or NULL.

**Port notes for `ForecastSeries`** (mirror `forecaster.py` @ `a534bdbd` function by function; keep the names in comments):
- `_as_context`: non-finite → missing (value 0, mask false).
- `ffill_with_mask` → truncate to the last `max_context` → `pad_to_patch` (left-pad to a multiple of 32, unobserved).
- Pack rows `{context, -context}` with `group_ids {0, 1}` when `symmetric_averaging` is on, otherwise 1 row.
- `_block_ar_decode`: blocks of `min(32, remaining)` patches. Feed back `replace_extreme_values(median, 1e30)` with mask true. When the context exceeds 32,768, keep the last `max(32, (32768/32)*32)` steps.
- `_refine_block` + `advancing_block_frames` + `causal_patch_loc_scale` in float64. The Hillis–Steele scan may be a sequential prefix (single row).
- `_cap_quantiles` against the assembled (truncated, padded) context with `k = 1e4`, using the last patch's causal scale. `clamp_nonfinite`.
- Symmetric average: `0.5 * (q + reverse_q(-q_mirror))`.
- Clip at 0 when `all_observed_nonnegative` and at least one observed value exists. Any NaN in the result → return false.

**Grid in this task:** sort by ts; step = `explicit_step` or the most common positive gap (ties → smallest); fewer than 2 distinct ts and no explicit step → `ok = false` (NULL). A duplicate ts → `InvalidInputException("timestral: duplicate timestamp … in one series; aggregate first, e.g. with time_bucket")`. `(ts - last) % width != 0` → `InvalidInputException("timestral: timestamp … is not on the series grid (step …); pass freq or use time_bucket")`. Materialise only the steps `k ∈ (−context_length, 0]` relative to the last row, so rows older than that only update `all_observed_nonnegative`. Missing grid points are NaN.

- [ ] **Step 1: The failing parity tests already exist** (generated in Task 3). Each block loads a case's input CSV, forecasts by `series_id` with `t` as the ts, unnests, joins on (`series_id`, `step`), and asserts the error is within tolerance. Shown here as a loop for brevity; the generated file has literal horizons:

```
require timestral
require-env TIMESTRAL_MODEL_PATH
statement ok
SET timestral_model_path = '${TIMESTRAL_MODEL_PATH}';

foreach case short_20 one_patch non_multiple long_truncate horizon_1 rollout_1500 interior_nulls leading_nulls trailing_nulls constant nonnegative_counts large_magnitude negative multi_series

query I
WITH h AS (SELECT horizon FROM 'test/data/parity/5m/cases.csv' WHERE "case" = '${case}'),
     f AS (SELECT series_id, unnest(timestral_forecast(t, value, (SELECT horizon FROM h)), recursive := true) FROM 'test/data/parity/5m/${case}_input.csv' GROUP BY series_id),
     e AS (SELECT * FROM 'test/data/parity/5m/${case}_expected.csv')
SELECT bool_and(abs(f.q50 - e.q50) <= 1e-4 * s.scale AND abs(f.q01 - e.q01) <= 1e-4 * s.scale AND abs(f.q99 - e.q99) <= 1e-4 * s.scale AND f.forecast = f.q50)
       AND count(*) = (SELECT count(*) FROM e)
FROM (SELECT *, row_number() OVER (PARTITION BY series_id ORDER BY ts) AS step FROM f) f
JOIN e USING (series_id, step)
JOIN (SELECT series_id, greatest(max(abs(q50)), 1e-6) AS scale FROM e GROUP BY ALL) s USING (series_id);
----
true

endloop
```

  Check all 21 quantiles (the snippet shows 3). Tune the tolerance only with a written justification in the test header. Before committing, run all four sizes once with `TIMESTRAL_MODEL_DIR=$PWD/scripts/export/out` and record the result in the commit message.

  `test/sql/forecast_basic.test`:
  - output type and field names
  - `horizon` 0 and 100001 → bind errors
  - a non-constant horizon → bind error
  - a group of only NULL values → NULL
  - a single row → NULL
  - a duplicate ts → the error above
  - an off-grid ts (step 2 with one odd ts) → the error above
  - future ts = last + k·step, including when the last rows are NULL (forecast starts after the last row)
  - **(Review focus 1)** `'inf'::DOUBLE` and `'nan'::DOUBLE` values give the same result as NULL in their place
  - **(Review focus 3)** `SET threads = 8`, then 2,000 groups of 100 rows each return 2,000 non-NULL lists
  - (from Task 4) a corrupted cached `timestral-5m.onnx` with `timestral_model_path` unset → the checksum error
- [ ] **Step 2:** `TIMESTRAL_MODEL_PATH=$PWD/scripts/export/out/timestral-5m.onnx make test`. Expected: FAIL (no function).
- [ ] **Step 3:** Implement the time grid, the forecaster and the aggregate.
- [ ] **Step 4:** Run the same command. Expected: PASS. Record wall time for the `rollout_1500` and `long_truncate` cases in the commit message.
- [ ] **Step 5:** Commit (`feat: timestral_forecast aggregate with C++ port of TimestralForecaster`).

---

### Task 6: Temporal timestamps, `freq` and options

**Files:**
- Modify: `src/time_grid.cpp`, `src/include/timestral/time_grid.hpp`, `src/forecast_aggregate.cpp`
- Test: `test/sql/forecast_time.test`

**Interfaces:**
- Consumes: `BuildGrid`, `GridPoint`, `GridStep` (Task 5).
- Produces: overloads `timestral_forecast(ts TIMESTAMP|TIMESTAMPTZ|DATE|BIGINT|INTEGER, value, horizon [, options STRUCT])`, where `options` is a foldable struct with optional keys `model VARCHAR` and `freq INTERVAL` (temporal ts) or `freq BIGINT` (integer ts). Unknown keys → `BinderException` listing the valid ones. The output `ts` has the input ts type.
- Internals: TIMESTAMP/TIMESTAMPTZ use epoch micros (instants); DATE uses micros at midnight and casts back. An interval with only `micros`/`days` is fixed-width (1 day = 86,400,000,000 µs). An interval with only `months` is calendar: `k = month_diff / months`, and the grid point is verified with `Interval::Add(base, months·k)` computed from the last point, never accumulated step by step. Mixed months and days/micros intervals → `BinderException("timestral: freq must be either a fixed duration or a whole number of months")`. Inference is fixed-width only.

- [ ] **Step 1: Write the failing tests** in `test/sql/forecast_time.test`:
  - hourly TIMESTAMP data (inferred step 1 h): the output ts are the next 24 hours.
  - DATE daily data: the output type is DATE.
  - monthly DATE data on the 1st without `freq` → the off-grid error, whose message mentions `freq := INTERVAL '1 month'`.
  - the same data with `{'freq': INTERVAL '1 month'}`: the next 12 month starts.
  - month-end data (Jan 31, Feb 28/29, Mar 31…) with a 1-month freq where the last point is the 31st: future points follow `Interval::Add` from the last point.
  - `{'model': '5m'}` override works; `{'model': 'x'}` → the unknown-size error.
  - `{'freq': INTERVAL '1 month 2 days'}` → the mixed-interval error.
  - an unknown option key → error.
  - integer ts with `{'freq': 5}`.
  - a gap of 3 missing hours in the middle forecasts the same as the explicit NULL rows.
  - **(Review focus 2)** one row at `1970-01-01` plus 500 1-second rows in 2026 → a non-NULL result in under 5 s.
  - **(Review focus 5)** TIMESTAMPTZ daily at local midnight in `Europe/Paris` across the October DST change → the off-grid error, whose message mentions casting to TIMESTAMP or `time_bucket`.
- [ ] **Step 2:** Run the tests with the model path. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS, and the Task 5 tests still pass.
- [ ] **Step 5:** Commit (`feat: temporal timestamps, freq and options for timestral_forecast`).

---

### Task 7: `timestral_forecast_by` table macro

**Files:**
- Create: `src/forecast_macro.cpp` (registered from `LoadInternal` as a default table macro)
- Test: `test/sql/forecast_by.test`

**Interfaces:**
- Produces: `timestral_forecast_by(tbl VARCHAR, group_by VARCHAR, ts VARCHAR, value VARCHAR, horizon INTEGER, options := NULL)` returning the group column (under its original name), followed by `ts, forecast, q01 … q99`, one row per (group, step). The body uses `query_table(tbl)` and `COLUMNS(...)` with exact-name matching (anchor the regex, or use a lambda).

- [ ] **Step 1: Write the failing test:** it gives the same rows as the hand-written `GROUP BY` + `unnest` over a 3-group table. It also checks that a group column `store` doesn't also match a column named `store_id`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement. **If DuckDB v1.5.6 macros can't select columns by a string parameter cleanly, drop this task** (spec §6.3 allows it). Record why in the README's limitations and skip to Task 8.
- [ ] **Step 4:** Run it. Expected: PASS.
- [ ] **Step 5:** Commit (`feat: timestral_forecast_by table macro`).

---

### Task 8: CI, community-extension readiness, docs

**Files:**
- Modify: `.github/workflows/MainDistributionPipeline.yml` (`extension_name: timestral`; `exclude_archs` from Global Constraints)
- Create: `.github/workflows/parity.yml` (ubuntu-latest: builds with vcpkg binary caching, downloads `timestral-5m.onnx` from the asset URL, verifies its SHA-256 against `model_manifest.hpp`, runs `TIMESTRAL_MODEL_PATH=… make test`)
- Create: `description.yml` (draft for community-extensions: `name: timestral`, `language: C++`, `build: cmake`, `license: MIT`, `maintainers`, `excluded_platforms`, `repo.github` and `ref` placeholders, a `hello_world` example, and an `extended_description` with the licence warning)
- Create: `README.md` (replacing the template docs). Sections: what it is; install (build from source now, `INSTALL timestral FROM community` later); quick start; `timestral_forecast` reference; options and settings; offline use (`timestral_models()`, `timestral_model_path`); model sizes and speed/accuracy (Mistral's published GIFT-Eval/TIME numbers plus the measured per-series latency); limitations (no covariates or multivariate yet, no WASM, `TIMESTAMPTZ` semantics, calendar `freq` must be explicit); licences (extension MIT, Mistral Modified MIT weights with the **$20M monthly revenue restriction**, ONNX Runtime MIT, anofox overlay ports MIT).
- Delete: `docs/NEXT_README.md`, `docs/README.md`, `docs/UPDATING.md` (template docs), if the bootstrap didn't already.

- [ ] **Step 1:** Push the branch and run the distribution pipeline. Expected: all non-excluded archs build and `make test` passes (the model tests skip without `TIMESTRAL_MODEL_PATH`). If `osx_amd64` fails because of ORT, add it to the exclusions in both the workflow and `description.yml`, with the reason in the README.
- [ ] **Step 2:** Run `parity.yml`. Expected: PASS once the requester has uploaded the `v1` files. Until then it fails at the download step; say so in the PR description.
- [ ] **Step 3:** Record the loadable-extension size per platform (from CI artifacts) in the README.
- [ ] **Step 4:** Run the code-quality workflow (`format;tidy`), fix any findings, and commit (`ci: distribution, parity workflow and community-extension draft`; `docs: README`).
