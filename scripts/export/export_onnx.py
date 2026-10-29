"""Export Timestral checkpoints to ONNX for the timestral DuckDB extension.

Each checkpoint's single-block forecast primitive (`TimestralModel._predict_context_only`: scaling,
patch embedding, masks, transformer, quantile head, denormalization) becomes one ONNX graph:

    inputs        float32 [rows, T]   T a multiple of 32; unobserved positions hold 0
    observed      bool    [rows, T]
    group_ids     int64   [rows]      must be distinct: every row is its own (univariate) series
    horizon_token float32 [H]         dummy; its length is the number of horizon patches (1..32)
    -> quantiles  float32 [rows, H*32, 21]  raw space, sorted along the last axis

The graph is exported at opset 23 with a few rewrites that ONNX Runtime runs faster (see `export_patches`):
the fused QKV projection is reordered head-major, attention and RoPE become ONNX Attention and
RotaryEmbedding on 3-D (rows, patches, heads * head_dim) tensors (no transposes), and variate attention is
dropped because over a single-row series it is the identity on its values.

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
import torch.nn.functional as F
import timestral.layers as timestral_layers
import timestral.model as timestral_model
from huggingface_hub import hf_hub_download
from onnxscript import opset23
from timestral import TimestralForecaster

# v2: opset-23 graphs with the export rewrites; v1 files keep their checksums for older extension builds.
ASSET_VERSION = "v2"
OPSET = 23
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


# Attention and RoPE on 3-D (batch, seq, heads * head_dim) tensors, as custom ops so the exporter maps them to
# single ONNX nodes. The eager implementations are the reference math, used for tracing.
@torch.library.custom_op("timestral_export::attention", mutates_args=())
def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, heads: int) -> torch.Tensor:
    b, sq, d = q.shape
    sk = k.shape[1]
    out = F.scaled_dot_product_attention(q.view(b, sq, heads, d // heads).transpose(1, 2),
                                         k.view(b, sk, heads, d // heads).transpose(1, 2),
                                         v.view(b, sk, heads, d // heads).transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2).reshape(b, sq, d)


@_attention.register_fake
def _(q, k, v, mask, heads):
    return torch.empty_like(q)


@torch.library.custom_op("timestral_export::rope", mutates_args=())
def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int) -> torch.Tensor:
    """Non-interleaved RoPE over whole heads; `cos`/`sin` are `(batch, seq, head_dim / 2)`."""
    b, s, d = x.shape
    x4 = x.view(b, s, heads, d // heads)
    cos2 = torch.cat([cos, cos], dim=-1).unsqueeze(2)
    sin2 = torch.cat([sin, sin], dim=-1).unsqueeze(2)
    return (x4 * cos2 + timestral_layers._rotate_half(x4) * sin2).reshape(b, s, d)


@_rope.register_fake
def _(x, cos, sin, heads):
    return torch.empty_like(x)


TRANSLATIONS = {
    # Attention-23 emits optional present-key/value and qk outputs; only Y is used, so the exporter prunes the rest.
    torch.ops.timestral_export.attention.default:
        lambda q, k, v, mask, heads: opset23.Attention(q, k, v, mask, q_num_heads=heads, kv_num_heads=heads)[0],
    torch.ops.timestral_export.rope.default:
        lambda x, cos, sin, heads: opset23.RotaryEmbedding(x, cos, sin, num_heads=heads, interleaved=0),
}


def _attention_forward(self, x, mask, cos, sin):
    """`layers.Attention.forward` on head-major QKV weights (see `export_patches`), with the same math."""
    d, heads = self.embed_dim, self.num_heads
    if not self.is_time:
        # Every row is its own series, so the variate mask is the diagonal: softmax over one key is 1 and the
        # attention output is v itself.
        out = F.linear(x, self.wQKV.weight[2 * d:])
        if self._gate is not None:
            out = out * self._gate(x).sigmoid()
        return self.wO(out)
    q, k, v = self.wQKV(x).split(d, dim=-1)
    half = d // heads // 2
    q = torch.ops.timestral_export.rope(q, cos[..., :half], sin[..., :half], heads)
    k = torch.ops.timestral_export.rope(k, cos[..., :half], sin[..., :half], heads)
    if isinstance(mask, timestral_layers.TwoStreamMask):
        n = mask.enc_len
        out = torch.cat([torch.ops.timestral_export.attention(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n], heads),
                         torch.ops.timestral_export.attention(q[:, n:], k, v, mask.mask[:, :, n:, :], heads)], dim=1)
    else:
        out = torch.ops.timestral_export.attention(q, k, v, mask, heads)
    if self._gate is not None:
        out = out * self._gate(x).sigmoid()
    return self.wO(out)


@contextlib.contextmanager
def export_patches(model):
    """Graph rewrites for the export, undone on exit so verification runs the unpatched reference.

    The fused QKV weight rows are laid out `(qkv, head_dim, heads)`; reordering them to `(qkv, heads, head_dim)`
    makes q, k and v plain head-major slices of one MatMul (same dot products) instead of a 5-D permute.
    """
    original_validity = timestral_model.compute_enc_key_validity
    original_forward = timestral_layers.Attention.forward
    saved = {}
    timestral_model.compute_enc_key_validity = _key_validity_export
    try:
        for name, module in model.named_modules():
            if isinstance(module, timestral_layers.Attention):
                d, heads = module.embed_dim, module.num_heads
                saved[name] = module.wQKV.weight.data
                module.wQKV.weight.data = saved[name].view(3, d // heads, heads, d).permute(0, 2, 1, 3).reshape(3 * d, d).contiguous()
        timestral_layers.Attention.forward = _attention_forward
        yield
    finally:
        timestral_layers.Attention.forward = original_forward
        timestral_model.compute_enc_key_validity = original_validity
        for name, module in model.named_modules():
            if name in saved:
                module.wQKV.weight.data = saved[name]


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
    with export_patches(model), torch.inference_mode():
        program = torch.onnx.export(
            wrapper,
            example,
            dynamo=True,
            opset_version=OPSET,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            custom_translation_table=TRANSLATIONS,
            dynamic_shapes={
                "inputs": {0: dims["rows"], 1: 32 * dims["patches"]},
                "observed": {0: dims["rows"], 1: 32 * dims["patches"]},
                "group_ids": {0: dims["rows"]},
                "horizon_token": {0: dims["horizon"]},
            },
        )
        # Initializers are read from the module's parameters when saving, so save before the patches undo the
        # reordered QKV weights.
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
    # Laid out exactly as clang-format (DuckDB's .clang-format) formats it, so `make format-check` passes.
    rows = "\n".join(
        f'    {{"{e["size"]}", "{e["file_name"]}", "{e["sha256"]}", {e["file_bytes"]}ULL,\n     "{e["hf_revision"]}"}},'
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
{rows}
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
