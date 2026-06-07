#!/usr/bin/env python3
"""Convert SigLIP2 SO400M weights to MLX fp16 and cache them locally (ml-ycd.7).

A deterministic, idempotent wrapper around ``mlx_embeddings``' converter. By
default it writes to the exact local cache dir the accelerator auto-loads from
(``src.models.clip.siglip2_cache_dir``), so a one-time run makes the native
SigLIP2 backend use pre-converted fp16 weights (~2.2 GB) instead of downloading
the HF bf16 safetensors and converting on first use of every install.

CRITICAL: the output dir name MUST contain a 'patchNN-NNN' token — the
mlx-embeddings loader regex-parses the patch size from the path and crashes
otherwise (config.json omits patch_size). The default cache dir keeps the repo
basename, which satisfies this; a custom --mlx-path is validated up front. See
the ml-ycd.1 spike for the full rationale.

The converter copies config.json / *.json and saves tokenizer.json into the
output dir, so the result is self-contained (weights + config + tokenizer) and
``siglip2_dir_is_complete`` recognizes it.

Usage (from ml/, venv active):

    .venv/bin/python scripts/convert_siglip2_mlx.py                 # -> default cache
    .venv/bin/python scripts/convert_siglip2_mlx.py --force         # re-convert
    .venv/bin/python scripts/convert_siglip2_mlx.py --verify        # load + encode check
    .venv/bin/python scripts/convert_siglip2_mlx.py \
        --upload-repo mlx-community/siglip2-so400m-patch16-384       # publish (ml-yo9)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make ``src`` importable when run as a standalone script from anywhere.
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.models.clip import (  # noqa: E402
    MLX_EMBEDDINGS_MAP,
    siglip2_cache_dir,
    siglip2_dir_is_complete,
)

DEFAULT_HF_REPO = MLX_EMBEDDINGS_MAP["ViT-SO400M-16-SigLIP2-384__webli"]
_PATCH_TOKEN = re.compile(r"patch\d+-\d+")


def _verify_load(path: Path) -> None:
    """Load the converted dir and run a tiny image+text encode as a smoke test.

    Confirms the weights actually load via the production path and yield a
    1152-dim L2-normalized vector — catching a corrupt/incomplete convert that
    file-presence checks alone would miss. Heavy (loads ~2 GB), so opt-in.
    """
    import io

    import numpy as np
    from PIL import Image

    from src.models.clip import MLXClip

    print(f"[verify] loading {path} via the production backend ...")
    # Point the accelerator at this dir regardless of cache-resolution order.
    import os

    os.environ["ML_SIGLIP2_MLX_PATH"] = str(path)
    clip = MLXClip("ViT-SO400M-16-SigLIP2-384__webli")
    if not getattr(clip, "_use_mlx_embeddings", False):
        raise SystemExit(
            "[verify] FAILED: backend fell back off the native MLX path "
            "(check the load error logged above)"
        )

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 80, 200)).save(buf, format="JPEG")
    img = clip.encode_image(buf.getvalue())
    txt = clip.encode_text("a photo of a cat")
    for name, emb in (("image", img), ("text", txt)):
        norm = float(np.linalg.norm(emb))
        if emb.shape != (1152,) or abs(norm - 1.0) > 1e-3:
            raise SystemExit(
                f"[verify] FAILED: {name} embedding shape={emb.shape} norm={norm:.5f}"
            )
        print(f"[verify] {name}: shape={emb.shape} norm={norm:.5f} OK")
    print("[verify] OK")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--hf-path",
        default=DEFAULT_HF_REPO,
        help=f"HF repo id (or local dir) to convert. Default: {DEFAULT_HF_REPO}",
    )
    ap.add_argument(
        "--mlx-path",
        default=None,
        help="Output dir. Default: the accelerator's local cache dir "
        "(siglip2_cache_dir). Its name MUST contain a 'patchNN-NNN' token.",
    )
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Weight dtype to save. Default: float16 (~2.2 GB).",
    )
    ap.add_argument(
        "--upload-repo",
        default=None,
        help="Optional HF repo to publish the converted dir to (ml-yo9).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if a complete cache already exists.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="After converting, load the dir and run a tiny encode smoke test.",
    )
    args = ap.parse_args()

    out = Path(args.mlx_path) if args.mlx_path else siglip2_cache_dir(DEFAULT_HF_REPO)

    # Fail fast on the one mistake that crashes the loader later (ml-ycd.1).
    if not _PATCH_TOKEN.search(out.name):
        ap.error(
            f"--mlx-path dir name {out.name!r} lacks a 'patchNN-NNN' token; the "
            "mlx-embeddings loader regex-parses the patch size from the path and "
            "will crash. Use e.g. '.../siglip2-so400m-patch16-384'."
        )

    if siglip2_dir_is_complete(out) and not args.force:
        print(f"[skip] complete convert already at {out} (use --force to redo)")
        if args.verify:
            _verify_load(out)
        if args.upload_repo:
            print(
                "[note] --upload-repo with an existing convert: re-run with "
                "--force to convert+upload, or upload the dir manually."
            )
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)

    from mlx_embeddings.convert import convert

    print(f"[convert] {args.hf_path} -> {out} (dtype={args.dtype})")
    convert(
        hf_path=args.hf_path,
        mlx_path=str(out),
        dtype=args.dtype,
        upload_repo=args.upload_repo,
    )

    if not siglip2_dir_is_complete(out):
        raise SystemExit(
            f"[convert] FAILED: {out} is missing config.json / tokenizer.json / "
            "*.safetensors after conversion"
        )
    print(f"[convert] OK -> {out}")
    print("  files:", ", ".join(sorted(p.name for p in out.iterdir())))

    if args.verify:
        _verify_load(out)

    print(
        "\nThe accelerator now auto-loads this convert (no env var needed) as "
        "long as it stays in the default cache dir."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
