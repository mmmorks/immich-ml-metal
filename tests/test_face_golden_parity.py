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
    g_keys = [str(k) for k in gold["img_keys"]]  # "<ident>/<file>" per face
    golden_top1 = float(gold["golden_top1"])

    samples = fep.load_image_dir(FIX / "faces")
    forced = os.getenv("ML_RUN_PARITY") == "1"

    # Fork pipeline: Apple Vision detect → ArcFace embed, per face. Each face is
    # tagged with the SAME stable image key the golden uses (Sample.name is
    # "<ident>/<file>"), so golden and fork align regardless of load order.
    mlx_emb, mlx_bbox, mlx_labels, mlx_keys = [], [], [], []
    try:
        for s in samples:
            for f in fep.fork_faces(s.data, nose_strategy="tip"):
                e = fep.embed(s.data, f["kps"])
                n = float(np.linalg.norm(e))
                mlx_emb.append((e / n if n > 0 else e).astype(np.float32))
                mlx_bbox.append(f["bbox"])
                mlx_labels.append(s.label)
                mlx_keys.append(s.name)
    except Exception as e:  # noqa: BLE001
        if forced:
            raise
        pytest.skip(f"MLX face pipeline unavailable: {e}")

    assert mlx_emb, "fork pipeline detected zero faces on committed fixtures"
    mlx_emb_arr = np.stack(mlx_emb)

    # Greedy-match golden faces to fork faces by bbox IoU, WITHIN the same source
    # image. SCRFD (golden) and Apple Vision (fork) routinely detect different
    # SETS of faces, so a golden face with no fork counterpart is a detection
    # difference, not an alignment-drift regression — it is left unmatched and
    # excluded from the median. We DO require that every committed image yields
    # at least one matched face (the fork must still find each main subject).
    drift = []
    matched_per_image: dict[str, int] = {k: 0 for k in g_keys}
    fork_by_key: dict[str, list[int]] = {}
    for j, k in enumerate(mlx_keys):
        fork_by_key.setdefault(k, []).append(j)
    for key in dict.fromkeys(g_keys):
        g_idx = [gi for gi, gk in enumerate(g_keys) if gk == key]
        f_idx = fork_by_key.get(key, [])
        g_boxes = [tuple(g_bbox[gi]) for gi in g_idx]
        f_boxes = [tuple(mlx_bbox[j]) for j in f_idx]
        for gi_local, fj_local in fep.greedy_match(g_boxes, f_boxes, iou_thresh=0.3):
            gi, fj = g_idx[gi_local], f_idx[fj_local]
            drift.append(float(np.dot(g_emb[gi], mlx_emb_arr[fj])))
            matched_per_image[key] += 1

    unmatched_images = [k for k, n in matched_per_image.items() if n == 0]
    assert not unmatched_images, (
        f"no fork face matched golden in image(s) {unmatched_images} — detection-set regression"
    )
    assert drift, "no golden/fork face pairs matched by IoU"

    median_cos = float(np.median(drift))
    assert median_cos >= MEDIAN_COS_MIN, (
        f"median alignment-drift cosine {median_cos:.4f} < {MEDIAN_COS_MIN}; per-face={sorted(drift)}"
    )

    # Fork's own top-1 retrieval accuracy (same-image excluded), compared to the
    # golden's. Integer image ids derived from the stable keys.
    key_to_id = {k: i for i, k in enumerate(dict.fromkeys(mlx_keys))}
    mlx_img_ids = [key_to_id[k] for k in mlx_keys]
    mlx_top1 = fep.top1_accuracy(mlx_emb_arr, mlx_labels, mlx_img_ids, mlx_emb_arr, mlx_labels, mlx_img_ids)
    drop = golden_top1 - float(mlx_top1)
    assert drop <= TOP1_DROP_MAX, (
        f"top-1 retrieval dropped {drop:.4f} (golden {golden_top1:.4f} -> mlx {mlx_top1:.4f}); max {TOP1_DROP_MAX}"
    )
