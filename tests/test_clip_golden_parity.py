"""Weights-gated CLIP parity: MLX production backend vs committed ONNX golden.

Skips unless prerequisites + golden artifacts are present (ML_RUN_PARITY=1 makes
that a hard failure). The conftest forces STUB_MODE=true for hermetic unit tests;
these tests load REAL models, so they run only on the opt-in/auto-detected path.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests._parity_gate import gate_reason

FIX = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIX / "golden"
THRESHOLD = 0.99

# (golden stem, Immich model name routed to the MLX backend)
CASES = [
    ("openai_clip", "ViT-B-32__openai"),
    ("siglip2", "ViT-SO400M-16-SigLIP2-384__webli"),
]


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def _load_fixtures():
    images = [(f.name, f.read_bytes()) for f in sorted((FIX / "clip").glob("*.jpg"))]
    queries = [ln.strip() for ln in (FIX / "queries.txt").read_text().splitlines() if ln.strip()]
    return images, queries


@pytest.mark.parametrize("stem,model_name", CASES, ids=[c[0] for c in CASES])
def test_clip_mlx_matches_onnx_golden(stem, model_name):
    reason = gate_reason(stem)
    if reason is not None:
        pytest.skip(reason)

    # Real models required; opt out of the conftest's forced STUB_MODE.
    os.environ["STUB_MODE"] = "false"

    gold = np.load(GOLDEN / f"{stem}.npz")
    g_img, g_txt = gold["image_embeds"], gold["text_embeds"]
    images, queries = _load_fixtures()
    assert len(images) == g_img.shape[0], "fixture/golden image count drift — regenerate golden"
    assert len(queries) == g_txt.shape[0], "fixture/golden query count drift — regenerate golden"

    from src.models.clip import get_clip_model

    forced = os.getenv("ML_RUN_PARITY") == "1"
    try:
        model = get_clip_model(model_name)
        mlx_img = np.stack([model.encode_image(b) for _, b in images])
        mlx_txt = np.stack([model.encode_text(q) for q in queries])
        model.unload()
    except Exception as e:  # noqa: BLE001 — translate to skip unless forced
        if forced:
            raise
        pytest.skip(f"MLX backend for {model_name} unavailable: {e}")

    img_cos = _cosine_rows(mlx_img, g_img)
    txt_cos = _cosine_rows(mlx_txt, g_txt)
    img_fail = [(images[i][0], float(img_cos[i])) for i in range(len(images)) if img_cos[i] < THRESHOLD]
    txt_fail = [(queries[j], float(txt_cos[j])) for j in range(len(queries)) if txt_cos[j] < THRESHOLD]
    assert not img_fail, f"{model_name}: image cosine below {THRESHOLD}: {img_fail}"
    assert not txt_fail, f"{model_name}: text cosine below {THRESHOLD}: {txt_fail}"
