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


FACE_NUM_IDS = 3
FACE_MIN_PER_ID = 2
FACE_MAX_PER_ID = 2


def _save_face_fixtures(samples) -> None:
    """Persist the LFW subset under tests/fixtures/faces/<label>/ (committed)."""
    root = FIX / "faces"
    for s in samples:
        # s.name is "<path>" or "<label>.jpg"; group by identity label.
        ident = (s.label or "unknown").replace(" ", "_")
        d = root / ident
        d.mkdir(parents=True, exist_ok=True)
        fname = Path(s.name).name
        (d / fname).write_bytes(s.data)


def gen_face() -> None:
    from face_embedding_parity import load_lfw, top1_accuracy, upstream_embeddings

    samples = load_lfw(FACE_NUM_IDS, FACE_MIN_PER_ID, FACE_MAX_PER_ID)
    _save_face_fixtures(samples)
    records = upstream_embeddings(samples)
    if not records:
        raise SystemExit("face: upstream pipeline detected zero faces — cannot freeze golden.")
    emb = np.stack([r["embedding"] for r in records])
    _check_finite("face", emb)

    # Identify each face by a STABLE per-image key (identity + filename) that the
    # test reconstructs identically from the committed fixtures via
    # load_image_dir (whose Sample.name is "<ident>/<file>"). Positional ids are
    # NOT stable across load_lfw vs load_image_dir ordering.
    def _ident(label: str | None) -> str:
        return (label or "unknown").replace(" ", "_")

    img_keys = [f"{_ident(r['label'])}/{Path(r['name']).name}" for r in records]
    labels = [_ident(r["label"]) for r in records]
    bboxes = np.array([r["bbox"] for r in records], dtype=np.float32)
    # Integer image ids (faces grouped by source image) for same-image-excluded
    # top-1; derived from the stable keys so the grouping is order-independent.
    key_to_id = {k: i for i, k in enumerate(dict.fromkeys(img_keys))}
    img_ids = [key_to_id[k] for k in img_keys]
    # Golden top-1 retrieval accuracy of the upstream embeddings against itself
    # (same-image excluded) — the test asserts MLX stays within 0.02 of this.
    golden_top1 = top1_accuracy(emb, labels, img_ids, emb, labels, img_ids)
    _write(
        "face",
        {
            "embeddings": emb,
            "bboxes": bboxes,
            "labels": np.array(labels),
            "img_keys": np.array(img_keys),
            "golden_top1": np.array(golden_top1, dtype=np.float64),
        },
        {
            "model": "buffalo_l (SCRFD det_10g + ArcFace w600k_r50)",
            "onnxruntime_version": ort.__version__,
            "dim": int(emb.shape[1]),
            "num_faces": len(records),
            "identities": sorted({r["label"] for r in records}),
            "golden_top1_accuracy": float(golden_top1),
            "generated": date.today().isoformat(),
        },
    )


GENERATORS = {"openai_clip": gen_openai_clip, "siglip2": gen_siglip2, "face": gen_face}


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
