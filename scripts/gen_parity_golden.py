#!/usr/bin/env python3
"""Generate committed golden parity references from the upstream ONNX models.

Run-once, locally (needs HF download + onnxruntime). Output is committed:
  tests/fixtures/golden/{openai_clip,siglip2,face}.npz + .json

Usage:
  .venv/bin/python scripts/gen_parity_golden.py --targets openai_clip siglip2 face
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "scripts"))

FIX = ML_ROOT / "tests" / "fixtures"
GOLDEN = FIX / "golden"

OPENAI_MODEL = "ViT-B-32__openai"


def _load_clip_images() -> list[tuple[str, bytes]]:
    files = sorted((FIX / "clip").glob("*.jpg"))
    if not files:
        raise SystemExit(f"No CLIP fixtures in {FIX / 'clip'}; run fetch_pd_clip_fixtures.py first.")
    return [(f.name, f.read_bytes()) for f in files]


def _load_queries() -> list[str]:
    return [ln.strip() for ln in (FIX / "queries.txt").read_text().splitlines() if ln.strip()]


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise SystemExit(f"{name}: non-finite values in golden embeddings — refusing to commit.")
    norms = np.linalg.norm(arr, axis=1)
    if (norms < 1e-6).any():
        raise SystemExit(f"{name}: zero-norm embedding in golden — refusing to commit.")


def _write(stem: str, npz: dict, manifest: dict) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN / f"{stem}.npz", **npz)
    (GOLDEN / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[golden] wrote {stem}.npz + {stem}.json")


def gen_openai_clip() -> None:
    from clip_parity import _ONNX_REPO, embed_onnx

    images = _load_clip_images()
    queries = _load_queries()
    img, txt = embed_onnx(OPENAI_MODEL, images, queries)
    _check_finite("openai_clip image", img)
    _check_finite("openai_clip text", txt)
    _write(
        "openai_clip",
        {"image_embeds": img, "text_embeds": txt},
        {
            "model": OPENAI_MODEL,
            "onnx_repo": _ONNX_REPO[OPENAI_MODEL][0],
            "onnx_repo_commit": "FILL_IN_resolved_sha",
            "onnxruntime_version": ort.__version__,
            "dim": int(img.shape[1]),
            "images": [n for n, _ in images],
            "queries": queries,
            "preprocess": "siglip_image_pixels @224 + CLIP mean/std; clean_text(canonicalize=False) + open_clip ViT-B-32 tokenizer",
            "generated": date.today().isoformat(),
        },
    )


def gen_siglip2() -> None:
    from embedding_parity import ONNX_REPO_SIGLIP2, embed_onnx_siglip2

    images = _load_clip_images()
    queries = _load_queries()
    img, txt = embed_onnx_siglip2(images, queries)
    _check_finite("siglip2 image", img)
    _check_finite("siglip2 text", txt)
    _write(
        "siglip2",
        {"image_embeds": img, "text_embeds": txt},
        {
            "model": "ViT-SO400M-16-SigLIP2-384__webli",
            "onnx_repo": ONNX_REPO_SIGLIP2,
            "onnx_repo_commit": "FILL_IN_resolved_sha",
            "onnxruntime_version": ort.__version__,
            "dim": int(img.shape[1]),
            "images": [n for n, _ in images],
            "queries": queries,
            "preprocess": "siglip_image_pixels @384 + 0.5 mean/std; SiglipTextTokenizer",
            "generated": date.today().isoformat(),
        },
    )


# gen_face() added in a later task.
GENERATORS = {"openai_clip": gen_openai_clip, "siglip2": gen_siglip2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    args = ap.parse_args()
    for t in args.targets:
        print(f"\n=== generating golden: {t} ===")
        GENERATORS[t]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
