# Timestral ONNX export

These scripts turn Mistral's Timestral checkpoints into the ONNX graphs the `timestral` DuckDB extension runs,
and generate the parity fixtures the extension is tested against. You only need them to cut a new asset
version; users of the extension never run them.

```bash
cd scripts/export
uv sync
uv run python export_onnx.py      # all sizes -> out/, regenerates src/include/timestral/model_manifest.hpp
uv run python make_fixtures.py    # test/data/parity/<size>/ and test/sql/parity_<size>.test
```

Everything is pinned: the Mistral code (`timestral` @ `a534bdbd`), the Hugging Face revision of every
checkpoint (`REVISIONS` in `export_onnx.py`), and the torch/onnx/onnxruntime versions (`uv.lock`).

## What gets exported

`TimestralModel._predict_context_only`, the single-block forecast without covariates, is exported with dynamic
rows, context length and horizon patches. Its inputs and outputs are documented at the top of `export_onnx.py`.
One export-only patch is needed: a boolean `cummax` has no ONNX lowering, so it is replaced by the equivalent
`cumsum > 0`. Each graph is checked against the unpatched PyTorch model on 6 shapes before anything is
written. A run fails if the error exceeds 1e-4 of the output scale; the observed worst case is about 4e-5.

The forecasting wrapper around the graph (`TimestralForecaster`) is ported to C++ in `src/forecaster.cpp`.

## Determinism

Re-exporting gives byte-identical files (checked 2026-10-20: `timestral-5m.onnx` had the same SHA-256 across
two runs). Even so, the committed `model_manifest.hpp` and the uploaded files should come from the same run,
because the extension refuses files whose SHA-256 differs from the header.

## Publishing a new asset version

1. Bump `ASSET_VERSION` in `export_onnx.py` if the graphs change, and the default `timestral_model_url` and
   `timestral_cache_dir` in the extension to match.
2. Run both scripts and commit the regenerated header, fixtures and tests.
3. Upload everything in `out/` (the four `.onnx` files, `manifest.json` and Mistral's `LICENSE`) to
   `https://assets.altertable.ai/timestral/<ASSET_VERSION>/`. The licence must stay next to the weights: it is
   a condition of Mistral's Modified MIT licence.
