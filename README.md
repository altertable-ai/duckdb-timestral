# timestral: zero-shot forecasting in DuckDB

`timestral` runs [Mistral's Timestral](https://mistral.ai/news/timestral/) time-series foundation models inside
DuckDB. A forecast is a SQL query: there is no training step, no Python and no GPU. The network runs on CPU through
a statically linked [ONNX Runtime](https://onnxruntime.ai/), and every forecast comes with 21 quantiles.

```sql
LOAD timestral;

SELECT store, ts, forecast, q10, q90
FROM timestral_forecast_by('sales', 'store', 'ts', 'amount', 24);
```

| store | ts                  | forecast | q10   | q90   |
|-------|---------------------|----------|-------|-------|
| 1     | 2026-01-21 20:00:00 | 10.98    | 10.96 | 11.00 |
| 1     | 2026-01-21 21:00:00 | 10.96    | 10.94 | 10.99 |
| …     | …                   | …        | …     | …     |

(`sales` here holds 500 hourly rows per store; the hello-world in `description.yml` builds it.)

> **Model licence.** This extension is MIT licensed, but the Timestral weights it downloads belong to Mistral AI
> and are released under Mistral's [Modified MIT licence](https://huggingface.co/mistralai/Timestral-5M/blob/main/LICENSE).
> That licence **grants no rights to a company whose consolidated monthly revenue (or its employer's) exceeds
> $20 million**, including through derivatives such as the ONNX files this extension uses. If that applies to you,
> contact Mistral AI (sales@mistral.ai) for a commercial licence before using the models.

## Install

Until it is published as a DuckDB community extension, build it from source (see [Building](#building)) and load
the resulting `build/release/extension/timestral/timestral.duckdb_extension`:

```sql
LOAD 'build/release/extension/timestral/timestral.duckdb_extension';
```

Once published, it will install with `INSTALL timestral FROM community; LOAD timestral;`.

On first use the extension downloads the selected model from `https://assets.altertable.ai/timestral/v1/`,
checks its SHA-256, and caches it in `~/.duckdb/timestral/v1/`.

## Forecasting

### `timestral_forecast(ts, value, horizon [, options])`

`timestral_forecast` is an aggregate: each group is one series. It returns a list with one entry per future step:

```sql
SELECT store, unnest(timestral_forecast(ts, amount, 24), recursive := true)
FROM sales
GROUP BY store;
```

| Argument | Type | Meaning |
|---|---|---|
| `ts` | `TIMESTAMP`, `TIMESTAMPTZ`, `DATE`, `BIGINT`, `INTEGER` | Time of each observation. Rows do not need to be ordered. |
| `value` | any numeric | Observed value. `NULL`, `NaN` and `±inf` count as missing. |
| `horizon` | constant integer, 1 to 100,000 | Number of future steps. |
| `options` | constant `STRUCT`, optional | `model`: `'5m'`, `'28m'`, `'123m'` or `'209m'`. `freq`: the step, an `INTERVAL` (or an integer for integer `ts`). |

Each entry is `STRUCT(ts, forecast, q01, q05, q10, …, q95, q99)`:
- `ts` has the input's type.
- `forecast` is the median (`q50`).
- `q01` to `q99` are the model's 21 quantiles.

How the time grid is built:
- **Step.** The step is the most common gap between consecutive timestamps, unless you pass `freq`. Monthly,
  quarterly and yearly data must pass `freq`, for example `{'freq': INTERVAL '1 month'}`, because calendar
  frequencies are not inferred from day counts.
  Calendar steps fall on the series' largest day of the month, clamped to each month's length, so month-end
  data stays on month ends (Jan 31, Feb 28, Mar 31…).
- **Gaps.** Missing grid points and `NULL` values are missing values, which the model handles.
- **Forecast start.** The forecast starts one step after the series' last row. If the last rows have `NULL`
  values, they still count as part of the history, so adding `NULL` rows moves the forecast start forward.
- **Duplicates and off-grid timestamps.** These raise an error. Pre-aggregate with `time_bucket` first.
- **Empty series.** A series with no observed value, or with a single row and no `freq`, returns `NULL`
  instead of failing the query.
- **Long histories.** Only the last 32,768 steps are used (the model's context length).

### `timestral_forecast_by(table, group_by, ts, value, horizon [, options := …])`

`timestral_forecast_by` is a table macro around the aggregate. It returns one row per (group, step), with the
group column under its own name followed by `ts, forecast, q01 … q99`. Column names are given as strings and
matched exactly.

```sql
FROM timestral_forecast_by('sales', 'store', 'ts', 'amount', 12, options := {'freq': INTERVAL '1 month'});
```

### Choosing a model

| Size | File | GIFT-Eval MASE ↓ | GIFT-Eval CRPS ↓ | 1,000 series, 2,016 steps of context → 96 steps* |
|---|---|---|---|---|
| `5m` (default) | 23 MB | 0.705 | 0.479 | 0.9 s |
| `28m` | 114 MB | 0.678 | 0.461 | 3.3 s |
| `123m` | 495 MB | 0.659 | 0.448 | 14 s |
| `209m` | 842 MB | 0.658 | 0.446 | 25 s |

\* Apple M4 Pro, 14 threads. The accuracy figures are Mistral's published GIFT-Eval results.

Use `SET timestral_model = '28m'` to change the default, or pass `{'model': '28m'}` to a single call.

## Settings

Forecasts run on DuckDB's own worker threads, one series per task, so `SET threads` controls the parallelism.

| Setting | Default | Meaning |
|---|---|---|
| `timestral_model` | `'5m'` | Default model size. |
| `timestral_model_url` | `'https://assets.altertable.ai/timestral/v1'` | Where model files are downloaded from. |
| `timestral_cache_dir` | `'~/.duckdb/timestral/v1'` | Where downloaded models are cached. |
| `timestral_model_path` | `''` | A local `.onnx` file to use instead of the download cache. No checksum check. |

### Offline use

`timestral_models()` lists every model with its URL, SHA-256, size and whether it is cached:

```sql
SELECT size, cached, local_path FROM timestral_models();
```

To use the extension on a machine without internet access, do either of these:
- Copy the `.onnx` file into `local_path`.
- Point `timestral_model_path` at the file.

## How it works

Mistral's reference implementation is PyTorch. `scripts/export/export_onnx.py` exports each checkpoint's
single-block forecast to ONNX: normalisation, patching, the transformer and the quantile head. The forecasting
logic around it is ported to C++ in `src/forecaster.cpp`, following the reference function by function. That logic
covers block-by-block rollout for long horizons, refined scaling, quantile capping, symmetric averaging and
non-negative clipping.

The parity tests in `test/sql/parity_*.test` compare all 21 quantiles with Mistral's PyTorch output on 14 cases.
Those cases include long contexts, horizons beyond 1,024 steps, missing values and very large magnitudes. The
largest difference is 1.1e-6 of the series scale.

## Limitations

- No covariates (past-only, known-future, categorical) and no joint multivariate forecasting yet. The model
  supports both, so they are a natural next step.
- Not available for DuckDB-WASM, Windows MinGW or Linux musl builds, because ONNX Runtime does not support them.
- `TIMESTAMPTZ` values are treated as UTC instants. Daily data at local midnight shifts by an hour across daylight
  saving changes and falls off the grid. Cast to `TIMESTAMP` in the local time zone first.
- The model is loaded, and downloaded if needed, when the query is planned. `EXPLAIN` therefore triggers the
  download too.

## Building

```bash
git clone --recurse-submodules https://github.com/altertable-ai/duckdb-timestral
cd duckdb-timestral
git clone https://github.com/microsoft/vcpkg.git ../vcpkg && ../vcpkg/bootstrap-vcpkg.sh
export VCPKG_TOOLCHAIN_PATH=$PWD/../vcpkg/scripts/buildsystems/vcpkg.cmake
GEN=ninja make release
```

The build requires C++17 or newer and uses the same standard for DuckDB and the extension.
The first build compiles ONNX Runtime from source through vcpkg, which takes a while. The patched ports in
`vcpkg_ports/` are explained in [`vcpkg_ports/README.md`](vcpkg_ports/README.md). The loadable extension is about
34 MB (macOS arm64).

Run the tests:

```bash
make test
```

Tests that need a model are skipped unless you point them at one. Export it with `scripts/export` (see its
[README](scripts/export/README.md)) or download it from the asset URL:

```bash
TIMESTRAL_MODEL_PATH=$PWD/scripts/export/out/timestral-5m.onnx make test
```

```bash
TIMESTRAL_MODEL_DIR=$PWD/scripts/export/out make test
```

The second command also runs the parity tests for the 28M, 123M and 209M models.

## Licences

- This extension: MIT (see `LICENSE`).
- Timestral weights and the ONNX files derived from them: Mistral AI, Modified MIT, with the revenue restriction
  above.
- Timestral reference code: Apache 2.0.
- ONNX Runtime: MIT.
- vcpkg overlay ports adapted from [anofox_tabfm](https://github.com/DataZooDE/anofox-tabfm): MIT.
