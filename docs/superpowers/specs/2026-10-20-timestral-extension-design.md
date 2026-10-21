# DuckDB `timestral` extension — design

Date: 2026-10-20
Status: approved direction, pending plan

Items marked **(builder's call — decided by Claude)** were not discussed with the
requester; they are defaults chosen to keep the design complete and can be overridden.

## 1. Intent

Bring Mistral's Timestral zero-shot forecasting models into DuckDB so a forecast is a SQL
query: no Python, no GPU, no separate service. Timestral (released 2026-10-15) is a family
of patch-based transformer forecasters (5M / 28M / 123M / 209M parameters) that run on
CPU in milliseconds and return 21 quantiles, which maps naturally onto SQL rows.

The requester explicitly prefers **adding a dependency over reimplementing the model**.
The neural network therefore runs in ONNX Runtime; only the thin forecasting wrapper
around it is written in C++.

## 2. Goals (v1)

1. Univariate zero-shot forecasting of many series in one query (one series per group).
2. Probabilistic output: median plus all 21 model quantiles per forecast step.
3. Model size selectable (5M default, 28M, 123M, 209M).
4. Handles missing values (SQL `NULL`) and gaps in the time grid.
5. Any horizon ≥ 1 (beyond 1,024 steps via block-by-block rollout, as the reference does).
6. Numerically matches Mistral's reference Python implementation (`TimestralForecaster.forecast`
   with default options) within a small tolerance — this is the definition of "correct".
7. Builds and loads on DuckDB's native platforms: linux amd64/arm64, macOS amd64/arm64,
   windows amd64 (MSVC).

## 3. Non-goals (v1) **(builder's call — decided by Claude)**

- Covariates (past-only, known-future, categorical) and multivariate targets. The model
  supports them; they are the obvious v2. v1 must not paint the API into a corner (see §6.4).
- A backtest / evaluation function (e.g. rolling-origin MASE/CRPS in SQL). Natural v2.
- DuckDB-WASM and windows MinGW builds.
- GPU inference, fine-tuning, quantized (int8/fp16) models.
- Actually submitting to the DuckDB community-extensions repository. The requester will do that
  later; v1 must be **ready for it** (see §8).

## 4. Anti-goals

- **A forecast that silently differs from Mistral's.** If the C++ wrapper drifts from the
  reference (scaling, rollout, symmetric averaging…), users get a worse model with
  Timestral's name on it. Parity tests are mandatory, not optional.
- **Reimplementing the network in C++.** The transformer stays inside the ONNX graph.
- **Needing Python at query time.**
- **One bad series failing a whole query.** A series with no observed values yields a NULL
  forecast, not an error **(builder's call — decided by Claude)**.

## 5. Constraints and facts established

### 5.1 The model (verified from Hugging Face and github.com/mistralai/timestral)

- Weights: one `model.safetensors` (all F32) + `config.json` per size, at
  `huggingface.co/mistralai/Timestral-{5M,28M,123M,209M}`. Sizes: 21 MB, 111 MB, 490 MB, 837 MB.
- Patch size 32; context up to 32,768 steps (most recent kept); one forward pass forecasts up
  to `block_size` = 32 patches (1,024 steps).
- 21 quantile levels: 0.01, 0.05, 0.10, …, 0.95, 0.99 (from `config.json` `quantile_knots`).
- Reference code: `src/timestral/{model,layers,masks,scaler,forecaster,covariate_encoding}.py`,
  Apache-2.0, PyTorch.
- Weights licence: "Modified MIT" — redistribution allowed with Mistral's attribution and
  permission notice included; **no rights for companies above $20M consolidated monthly
  revenue**, and this explicitly extends to derivatives (our ONNX exports). See §9.

### 5.2 Where the ONNX boundary sits

The reference splits cleanly into:

- `TimestralModel.predict_block(inputs, observed, group_ids, horizon_patches)` — causal
  per-patch scaling + asinh, patch embedding, attention masks/positions, transformer, quantile
  head, denormalization. **This is the ONNX graph.** It is the bulk of the logic and all of the
  weights.
- `TimestralForecaster` (≈300 lines) — **re-implemented in C++** in the extension:
  - context shaping: NaN/NULL → mask, forward-fill, left-pad to a multiple of 32, truncate to
    `max_context`;
  - block-autoregressive decode (feed each block's median back as context);
  - `_refine_block` with `advancing_block_frames` (float64 Welford statistics);
  - `_cap_quantiles` (clamp to `[min − k·s, max + k·s]`, `k = 1e4`);
  - symmetric averaging ("clampsym": also forecast `−context`, reflect, average) — on by default
    in the reference, so on by default here;
  - non-negative clipping — on by default;
  - NaN-output detection.

  The C++ must follow the reference's dtypes (float64 where the reference uses float64).

Why not put more in the graph: the rollout loop length depends on the horizon and the refine
step contains a Python loop over horizon patches; exporting those reliably is harder than
porting ~300 lines of array arithmetic. Why not less: the masks/positions code
(`masks.py`) uses `torch.unique` and data-dependent shapes; it should go into the graph if the
exporter accepts it. **If export of the masks fails, the fallback is to compute masks and
positions in C++ and pass them as graph inputs** (builder decides after trying).

### 5.3 Dependencies

- ONNX Runtime via vcpkg (`onnxruntime` port, 1.31.0 at time of writing, supports all
  non-UWP targets). CPU execution provider only.
- Rejected: `libtorch` (vcpkg port has no static Windows build; DuckDB links extension deps
  statically; very large); `ggml` (would require hand-writing the forward pass).
- Remove the template's OpenSSL dependency. For checksums use DuckDB's bundled mbedtls
  SHA-256 wrapper **(builder's call — decided by Claude)**.
- HTTP downloads go through DuckDB's virtual filesystem (auto-loading `httpfs`), so the
  extension does not link its own HTTP client **(builder's call — decided by Claude)**.

### 5.4 Repository

This repo is an uninitialised copy of `duckdb/extension-template` (placeholder name
`waddle`, submodules `duckdb` @ 069cc9f and `extension-ci-tools` @ 3fd6109 not checked out).
Rename to `timestral` with `scripts/bootstrap-template.py`.

## 6. What gets built

### 6.1 Export pipeline (Python, offline, in-repo)

`scripts/export_onnx.py` (with its own pinned environment, e.g. `scripts/requirements.txt` or
`uv`):

1. Loads each checkpoint with Mistral's `timestral` package (`TimestralForecaster.from_pretrained`).
2. Exports `predict_block` (context-only path) to ONNX with dynamic axes for rows, context
   length and `horizon_patches` (or one graph per `horizon_patches` value if a dynamic
   decode length cannot be exported — builder's call).
3. Verifies the ONNX output against PyTorch on random inputs before writing.
4. Writes `timestral-<size>.onnx`, a manifest (`manifest.json`: model sizes, file names,
   SHA-256, source HF revision, `config.json` contents the C++ needs — patch size, block size,
   context length, quantile knots, minimum_scale, use_asinh), and Mistral's LICENSE.
5. Generates parity fixtures (§7).

The requester uploads the output to **assets.altertable.ai** manually. Layout
**(builder's call — decided by Claude)**: `https://assets.altertable.ai/timestral/v1/<files>`;
the version segment changes whenever the export changes.

### 6.2 Model loading in the extension

- On first use of a size, download `timestral-<size>.onnx` from the base URL, verify SHA-256
  against the manifest values compiled into the extension, store in a cache directory, then load.
  Later loads read the cache.
- One shared `Ort::Env` and one `Ort::Session` per model size per DuckDB instance, created
  lazily and reused across queries.
- Settings **(builder's call — decided by Claude)**:
  - `timestral_model` — default size (`'5m'`).
  - `timestral_model_url` — base URL (default the assets.altertable.ai path above).
  - `timestral_cache_dir` — default `~/.duckdb/timestral/v1`.
  - `timestral_model_path` — explicit local `.onnx` file, bypassing download (air-gapped use
    and tests).
  - `timestral_threads` — ONNX Runtime intra-op threads.
- Clear errors for: download failure (mention `timestral_model_path`), checksum mismatch,
  unknown model size.

### 6.3 SQL interface

The core is an **aggregate function**, so DuckDB's `GROUP BY` handles partitioning into series
and parallelism **(builder's call — decided by Claude)**:

```sql
SELECT store, unnest(timestral_forecast(ts, amount, 96), recursive := true)
FROM sales
GROUP BY store;
```

`timestral_forecast(ts, value, horizon [, options])`

- `ts`: `TIMESTAMP`, `TIMESTAMPTZ`, `DATE`, or an integer step index.
- `value`: any numeric, cast to `DOUBLE`; `NULL` = missing.
- `horizon`: constant integer ≥ 1.
- `options` (optional constant `STRUCT`): `model` (size, overrides the setting), `freq`
  (`INTERVAL`, or integer for integer `ts`).
- Returns `LIST(STRUCT(ts <same type as input>, forecast DOUBLE, q01, q05, q10, q15, q20, q25,
  q30, q35, q40, q45, q50, q55, q60, q65, q70, q75, q80, q85, q90, q95, q99 DOUBLE))`, one
  element per future step. `forecast` equals `q50` (the median); it is the column most users
  want and is named for discoverability.

Time-grid handling:

- Rows are sorted by `ts` inside the aggregate (no `ORDER BY` required from the user).
- Frequency: `freq` if given; otherwise inferred as the most common gap between consecutive
  timestamps. Calendar frequencies (month, quarter, year) are not inferred from fixed gaps;
  users pass `freq := INTERVAL '1 month'`.
- Missing grid points become missing values (the model accepts them).
- Duplicate timestamps, or timestamps off the grid, raise an error that suggests
  `time_bucket` to pre-aggregate.
- The context ends at the **last row of the group, even if its value is `NULL`**: trailing
  `NULL`s stay in the context as masked steps, exactly as trailing NaNs do in the reference.
  Future `ts` values = last row's grid point + k·freq. A side effect users can rely on: adding
  `NULL`-valued rows moves the forecast origin forward (e.g. to "today").
- Values are cast to float32 before entering the model, as the reference does.
- Series with no observed values, or whose forecast comes out NaN (the reference raises), return
  `NULL` **(builder's call — decided by Claude)**.

Convenience table macro **(builder's call — decided by Claude; drop it if DuckDB's macro
system can't express it cleanly)**:

```sql
FROM timestral_forecast_by('sales', group_by := 'store', ts := 'ts', value := 'amount', horizon := 96);
```

returning one row per (group, future step) with the struct fields as columns.

### 6.4 Leaving room for v2

Covariates and multivariate targets will need more inputs per row (e.g. extra argument(s) of
`STRUCT` type, and future rows with `NULL` target carrying known-future covariates). Keep the
argument list extensible: `options` is a struct, and nothing in v1 should depend on the
aggregate taking exactly two data columns. The export script should be structured so the
covariate path (`_predict_window`) can be exported later as a second graph.

### 6.5 Execution and performance

- The aggregate buffers `(ts, value)` per group; work happens in finalize.
- Finalize receives a vector of group states; batch several series into one ONNX `Run`
  (pack rows; `group_ids` keeps series independent). Series of different lengths are
  left-padded to a common length with unobserved steps; this does not change results, since
  leading unobserved patches are never valid attention keys and positions count from the first
  observed patch (parity tests should still cover a mixed-length batch). **Cap the batch** (start around 64 rows,
  builder tunes): the variate-attention mask is `(patches, 1, R, R)`, so memory grows
  quadratically with rows per call. Batch series of similar length to limit padding.
- ONNX Runtime threading vs DuckDB threading: avoid oversubscription (builder decides; a
  reasonable start is intra-op threads = `timestral_threads`, default 1, relying on DuckDB's
  parallel finalize across groups, and measuring with a single huge series).
- Symmetric averaging doubles the compute; it can share one `Run` with the main pass by packing
  `context` and `−context` together.

## 7. Correctness and testing

- **Parity fixtures**: the export script generates a set of input series and the reference
  outputs from `TimestralForecaster.forecast` (5M model, default options), saved as
  CSV/Parquet under `test/data/`. Cases must cover: short series (< 32 steps), exactly one
  patch, non-multiple of 32, long (> 32,768, exercising truncation), horizon 1, horizon
  > 1,024 (rollout), interior, leading and trailing NULLs, constant series, all-non-negative series
  (clipping), large-magnitude values (capping), negative series.
- **sqllogictest** (`test/sql/`) compares extension output against the fixtures with a tolerance
  (start at 1e-4 relative; builder justifies the final value).
- SQL-surface tests: frequency inference, `freq` override, gaps, duplicate-timestamp error,
  integer `ts`, `DATE` `ts`, NULL result for empty series, unknown model size error, settings.
- Tests load the model via `timestral_model_path` from a locally exported file; CI obtains the
  file either by downloading from assets.altertable.ai or by running the export (builder's call,
  based on CI time).
- The parity tests for 28M/123M/209M may be optional/slow, but must exist and pass once.

## 8. Build and CI

- Add `onnxruntime` to `vcpkg.json`; link statically into both static and loadable extension
  targets.
- Update `MainDistributionPipeline.yml` to exclude wasm and MinGW targets (and any other target
  where ONNX Runtime does not build, with a note why).
- Expect long first builds (ORT compiles from source); make sure CI caching is on.
- Record the resulting loadable-extension size per platform in the README.

### 8.1 Distribution: open-source community extension (later)

The requester will publish this as an open-source DuckDB community extension. Consequences:

- No custom deploy job or bucket credentials in this repo's CI; binaries are built and signed
  by the community-extensions CI, and users run `INSTALL timestral FROM community`.
- The repo stays compatible with the community build: standard `extension-ci-tools` makefile
  flow, all native deps from vcpkg (community CI uses its own pinned vcpkg commit, so check
  that the `onnxruntime` port builds there), and platform exclusions expressed so they can be
  copied into the community `description.yml`.
- Ship a draft `description.yml` (name, description, licence, maintainers, repo/ref,
  `excluded_platforms`, docs/hello-world example) in the repo for the later submission.
- Target the latest stable DuckDB release (the community repo builds against it), keeping
  this repo's CI on the same version.
- Risk to check early: ONNX Runtime's from-source build time against community CI limits.
- README install section: build from source now; `INSTALL timestral FROM community` once
  published.

## 9. Licensing and documentation

- Extension code: keep the template's MIT licence.
- README states, prominently: model weights are Mistral's under the Modified MIT licence,
  including the **$20M monthly revenue restriction** that applies to users of the extension;
  link to Mistral's licence and to Mistral for commercial licensing.
- The hosted `.onnx` files ship with Mistral's attribution + permission notice (LICENSE next to
  them on assets.altertable.ai).
- README covers: install/load, the aggregate and macro with examples, settings, offline use,
  model sizes vs speed/accuracy (Mistral's published numbers), limitations (no covariates yet).

## 10. Left to the builder

- Exact ONNX export technique (TorchScript vs dynamo exporter, opset), and whether masks live in
  the graph or in C++ (§5.2).
- Internal C++ structure, batching heuristics and thread defaults (§6.5).
- Tolerance values, fixture formats, CI strategy for obtaining the model.
- Whether the table macro is feasible (§6.3).

## 11. Open risk to retire first

The plan should start with a spike: export the 5M `predict_block` to ONNX and confirm ONNX
Runtime reproduces PyTorch's output on varied context lengths and horizons. Everything else
depends on that.
