"""Tests for scripts/face_embedding_parity.py.

Cover the pure, model-free pieces of the face-parity harness: IoU box overlap,
the greedy cross-detector matcher (the bit that stops index-0-vs-index-0
mismatches when SCRFD and Vision find different SETS of faces), the drift-cosine
summary, and same-image-excluded top-1 retrieval. The heavyweight detectors
(insightface, Apple Vision) are NOT imported — the script loads by path.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "face_embedding_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("face_embedding_parity", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def test_iou_identical_and_disjoint():
    box = (0.0, 0.0, 10.0, 10.0)
    assert mod.iou(box, box) == 1.0
    assert mod.iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x5 corner: inter=25, union=175.
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    assert math.isclose(mod.iou(a, b), 25.0 / 175.0, rel_tol=1e-9)


def test_greedy_match_pairs_best_iou_one_to_one():
    # Detector A finds 3 boxes; B finds 2. Boxes 0 and 2 align; box 1 (A) and
    # the extra B box go unmatched. Greedy must not double-assign.
    a = [(0, 0, 10, 10), (100, 100, 110, 110), (50, 50, 60, 60)]
    b = [(1, 1, 11, 11), (51, 49, 61, 59)]
    pairs = mod.greedy_match(a, b, iou_thresh=0.3)
    assert sorted(pairs) == [(0, 0), (2, 1)]
    # each index used at most once
    assert len({i for i, _ in pairs}) == len(pairs)
    assert len({j for _, j in pairs}) == len(pairs)


def test_greedy_match_respects_threshold():
    a = [(0, 0, 10, 10)]
    b = [(8, 8, 18, 18)]  # tiny overlap, IoU well below 0.3
    assert mod.greedy_match(a, b, iou_thresh=0.3) == []


def test_describe_pass_fractions():
    stats = mod.describe([1.0, 0.96, 0.92, 0.80])
    assert stats["max"] == 1.0
    assert stats["min"] == 0.80
    assert math.isclose(stats["frac_ge_090"], 0.75)
    assert math.isclose(stats["frac_ge_095"], 0.5)


def test_describe_empty_is_nan():
    stats = mod.describe([])
    assert math.isnan(stats["median"])


def test_top1_excludes_same_image():
    # Two identities, two faces each, on distinct source images. A face must not
    # retrieve itself (same img_id) and the nearest *other-image* face shares its
    # label, so accuracy is perfect.
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.99, 0.14])
    b1 = np.array([0.0, 1.0])
    b2 = np.array([0.14, 0.99])
    emb = np.vstack([a1, a2, b1, b2])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = ["A", "A", "B", "B"]
    img = [0, 1, 2, 3]
    acc = mod.top1_accuracy(emb, labels, img, emb, labels, img)
    assert acc == 1.0


def test_top1_counts_singletons_even_when_wrong():
    # 'A' is a single face but has other-image neighbours (the B faces), so it IS
    # counted — and its nearest other-image face is a B, so it's wrong. Accuracy
    # is 2/3: both B faces match each other, A misses. Confirms a face is never
    # dropped just for being a lone example of its identity.
    emb = np.vstack([[1.0, 0.0], [0.0, 1.0], [0.02, 1.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = ["A", "B", "B"]
    img = [0, 1, 2]
    acc = mod.top1_accuracy(emb, labels, img, emb, labels, img)
    assert math.isclose(acc, 2.0 / 3.0)


def test_top1_uncounted_when_only_candidate_is_same_image():
    # Both faces live on the SAME source image. After excluding same-image
    # candidates neither has any neighbour, so nothing is counted → NaN.
    emb = np.vstack([[1.0, 0.0], [0.0, 1.0]])
    labels = ["A", "B"]
    img = [0, 0]
    acc = mod.top1_accuracy(emb, labels, img, emb, labels, img)
    assert math.isnan(acc)
