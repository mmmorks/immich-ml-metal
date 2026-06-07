"""Weights-gated face parity: MLX/Apple-Vision fork vs committed upstream golden.

Asserts alignment-drift median cosine >= 0.90 and top-1 retrieval drop <= 0.02.
Apple Vision (the fork detector) is macOS-only; skips elsewhere.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._parity_gate import ParityUnavailable, gate_reason

FIX = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIX / "golden"
MEDIAN_COS_MIN = 0.90
TOP1_DROP_MAX = 0.02


def _require_macos():
    if sys.platform != "darwin":
        if os.getenv("ML_RUN_PARITY") == "1":
            raise ParityUnavailable("face parity needs Apple Vision (macOS only)")
        pytest.skip("face parity needs Apple Vision (macOS only)")


def test_face_mlx_matches_onnx_golden():
    reason = gate_reason("face")
    if reason is not None:
        pytest.skip(reason)
    _require_macos()
    os.environ["STUB_MODE"] = "false"

    # Load script helpers by path (scripts/ is not a package).
    spec = importlib.util.spec_from_file_location(
        "face_embedding_parity",
        Path(__file__).resolve().parents[1] / "scripts" / "face_embedding_parity.py",
    )
    fep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fep)

    gold = np.load(GOLDEN / "face.npz", allow_pickle=False)
    g_emb = gold["embeddings"]
    g_bbox = gold["bboxes"]
    g_labels = list(gold["labels"])
    g_img_ids = list(int(x) for x in gold["img_ids"])
    golden_top1 = float(gold["golden_top1"])

    samples = fep.load_image_dir(FIX / "faces")
    forced = os.getenv("ML_RUN_PARITY") == "1"

    # Fork pipeline: Apple Vision detect → ArcFace embed, per face.
    mlx_emb, mlx_bbox, mlx_labels, mlx_img_ids = [], [], [], []
    try:
        for img_id, s in enumerate(samples):
            for f in fep.fork_faces(s.data, nose_strategy="tip"):
                e = fep.embed(s.data, f["kps"])
                n = float(np.linalg.norm(e))
                mlx_emb.append((e / n if n > 0 else e).astype(np.float32))
                mlx_bbox.append(f["bbox"])
                mlx_labels.append(s.label)
                mlx_img_ids.append(img_id)
    except Exception as e:  # noqa: BLE001
        if forced:
            raise
        pytest.skip(f"MLX face pipeline unavailable: {e}")

    assert mlx_emb, "fork pipeline detected zero faces on committed fixtures"
    mlx_emb_arr = np.stack(mlx_emb)

    # Greedy-match fork faces to golden faces by bbox IoU (same image only).
    drift = []
    for gi in range(len(g_labels)):
        best_j, best_iou = -1, 0.0
        for j in range(len(mlx_bbox)):
            if mlx_img_ids[j] != g_img_ids[gi]:
                continue
            iou = fep.iou(tuple(g_bbox[gi]), tuple(mlx_bbox[j]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        assert best_j >= 0 and best_iou >= 0.3, (
            f"golden face {gi} ({g_labels[gi]}, img {g_img_ids[gi]}) has no fork match "
            f"(best IoU {best_iou:.2f}) — detection-set regression"
        )
        drift.append(float(np.dot(g_emb[gi], mlx_emb_arr[best_j])))

    median_cos = float(np.median(drift))
    assert median_cos >= MEDIAN_COS_MIN, (
        f"median alignment-drift cosine {median_cos:.4f} < {MEDIAN_COS_MIN}; per-face={sorted(drift)}"
    )

    mlx_top1 = fep.top1_accuracy(mlx_emb_arr, mlx_labels, mlx_img_ids, mlx_emb_arr, mlx_labels, mlx_img_ids)
    drop = golden_top1 - float(mlx_top1)
    assert drop <= TOP1_DROP_MAX, (
        f"top-1 retrieval dropped {drop:.4f} (golden {golden_top1:.4f} -> mlx {mlx_top1:.4f}); max {TOP1_DROP_MAX}"
    )
