"""Export Timestral checkpoints to ONNX for the timestral DuckDB extension.

Each checkpoint's single-block forecast primitive (`TimestralModel._predict_context_only`: scaling,
patch embedding, masks, transformer, quantile head, denormalization) becomes one ONNX graph:

    inputs        float32 [rows, T]   T a multiple of 32; unobserved positions hold 0
    observed      bool    [rows, T]
    group_ids     int64   [rows]      rows with the same id are one series
    horizon_token float32 [H]         dummy; its length is the number of horizon patches (1..32)
    -> quantiles  float32 [rows, H*32, 21]  raw space, sorted along the last axis

The forecasting wrapper around it (block rollout, refine frames, capping, symmetric averaging,
clipping) is ported to C++ in the extension. Every export is checked against PyTorch before it is
written. Outputs land in `out/`: the .onnx files, manifest.json and Mistral's weights LICENSE, and
the extension's `src/include/timestral/model_manifest.hpp` is regenerated.

Usage: uv run python -I export_onnx.py [--sizes 5m,28m,123m,209m] [--out out]
"""

import argparse
import contextlib
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import timestral.model as timestral_model
from huggingface_hub import hf_hub_download
from timestral import TimestralForecaster

ASSET_VERSION = "v1"
REVISIONS = {
    "5m": ("mistralai/Timestral-5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"),
    "28m": ("mistralai/Timestral-28M", "49b26409ac87b5935539a5d8ffe7f397b720db47"),
    "123m": ("mistralai/Timestral-123M", "5a147287addda85f6acd8da3fb8858fe9af7e8f7"),
    "209m": ("mistralai/Timestral-209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"),
}
INPUT_NAMES = ["inputs", "observed", "group_ids", "horizon_token"]
OUTPUT_NAMES = ["quantiles"]
# (rows, context length, horizon patches) checked against PyTorch for every export.
VERIFY_CASES = [(1, 32, 1), (2, 32, 1), (3, 160, 4), (8, 2048, 32), (16, 9600, 7), (2, 32768, 32)]
VERIFY_TOLERANCE = 1e-4
REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER_PATH = REPO_ROOT / "src" / "include" / "timestral" / "model_manifest.hpp"


def _key_validity_export(obs_per_patch, attend_missing):
    # Same as masks.compute_enc_key_validity; bool cummax has no ONNX lowering, (cumsum > 0) is equivalent.
    if not attend_missing:
        return obs_per_patch
    return obs_per_patch | (obs_per_patch.to(torch.int32).cumsum(dim=-1) > 0)


@contextlib.contextmanager
def export_patches():
    original = timestral_model.compute_enc_key_validity
    timestral_model.compute_enc_key_validity = _key_validity_export
    try:
        yield
    finally:
        timestral_model.compute_enc_key_validity = original


class PredictBlock(torch.nn.Module):
    """Context-only `predict_block`, with the horizon carried as a tensor length so it stays dynamic."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, observed, group_ids, horizon_token):
        return self.model._predict_context_only(inputs, observed, group_ids, horizon_token.shape[0])


def export_graph(model, path: Path) -> None:
    wrapper = PredictBlock(model).eval()
    rows, t, h = 4, 32 * 8, 2
    example = (torch.randn(rows, t).cumsum(-1), torch.ones(rows, t, dtype=torch.bool), torch.arange(rows),
               torch.zeros(h))
    dims = {
        "rows": torch.export.Dim("rows", min=2, max=4096),
        "patches": torch.export.Dim("patches", min=1, max=1024),
        "horizon": torch.export.Dim("horizon", min=1, max=32),
    }
    with export_patches(), torch.inference_mode():
        program = torch.onnx.export(
            wrapper,
            example,
            dynamo=True,
            opset_version=20,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_shapes={
                "inputs": {0: dims["rows"], 1: 32 * dims["patches"]},
                "observed": {0: dims["rows"], 1: 32 * dims["patches"]},
                "group_ids": {0: dims["rows"]},
                "horizon_token": {0: dims["horizon"]},
            },
        )
    program.save(str(path))


def verify_graph(model, path: Path) -> None:
    """Compare ONNX Runtime against the unpatched PyTorch `predict_block`; raise on mismatch."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for rows, t, h in VERIFY_CASES:
        g = torch.Generator().manual_seed(rows * 100003 + t * 31 + h)
        values = torch.randn(rows, t, generator=g).cumsum(-1) * 10 + 100
        observed = torch.rand(rows, t, generator=g) > 0.05
        observed[0, : t // 3] = False
        inputs = torch.where(observed, values, torch.zeros_like(values))
        group_ids = torch.arange(rows)
        with torch.inference_mode():
            ref = model.predict_block(inputs, observed, group_ids, h).numpy()
        out = session.run(None, {"inputs": inputs.numpy(), "observed": observed.numpy(),
                                 "group_ids": group_ids.numpy(), "horizon_token": np.zeros(h, np.float32)})[0]
        err = float(np.abs(out - ref).max())
        limit = VERIFY_TOLERANCE * float(np.abs(ref).max())
        print(f"  verify rows={rows} T={t} H={h}: max abs err {err:.3e} (limit {limit:.3e})")
        if out.shape != ref.shape or not err <= limit:
            raise SystemExit(f"verification failed for {path.name}: rows={rows} T={t} H={h}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_header(entries: list[dict]) -> None:
    rows = "\n".join(
        f'    {{"{e["size"]}", "{e["file_name"]}", "{e["sha256"]}", {e["file_bytes"]}ULL, "{e["hf_revision"]}"}},'
        for e in entries
    )
    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(f"""// GENERATED by scripts/export/export_onnx.py; do not edit by hand.
#pragma once

#include <cstdint>

namespace duckdb {{
namespace timestral {{

struct ModelManifestEntry {{
	const char *size;
	const char *file_name;
	const char *sha256;
	uint64_t file_bytes;
	const char *hf_revision;
}};

static constexpr const char *ASSET_VERSION = "{ASSET_VERSION}";

static constexpr ModelManifestEntry MODEL_MANIFEST[] = {{
{rows.replace("    ", chr(9))}
}};

}} // namespace timestral
}} // namespace duckdb
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="5m,28m,123m,209m")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    sizes = [s.strip().lower() for s in args.sizes.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": []}
    by_size = {m["size"]: m for m in manifest["models"]}
    config = None
    for size in sizes:
        repo, revision = REVISIONS[size]
        print(f"exporting {repo}@{revision}")
        forecaster = TimestralForecaster.from_pretrained(repo, revision=revision, device="cpu")
        model = forecaster.model.eval()
        path = args.out / f"timestral-{size}.onnx"
        export_graph(model, path)
        verify_graph(model, path)
        c = model.config
        config = {"patch_size": c.patch_size, "block_size": c.block_size, "context_length": c.context_length,
                  "minimum_scale": c.minimum_scale, "use_asinh": c.use_asinh,
                  "quantile_knots": list(c.quantile_knots), "median_index": c.median_index}
        by_size[size] = {"size": size, "file_name": path.name, "sha256": sha256_of(path),
                         "file_bytes": path.stat().st_size, "hf_revision": revision, "config": config}
        shutil.copyfile(hf_hub_download(repo, "LICENSE", revision=revision), args.out / "LICENSE")

    entries = [by_size[s] for s in REVISIONS if s in by_size]
    manifest = {"asset_version": ASSET_VERSION, "models": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_header(entries)
    for e in entries:
        print(f"{e['file_name']}: {e['file_bytes']} bytes sha256 {e['sha256']}")


if __name__ == "__main__":
    main()
