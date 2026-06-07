#!/usr/bin/env python3
"""Face-embedding parity harness: fork (Apple Vision landmarks) vs upstream (SCRFD/RetinaFace landmarks).

Parity gate. Decides preserve-vs-reindex for Immich's existing FACE
index (and clusters) after migrating to this fork's Apple-Vision face pipeline.

WHAT ACTUALLY DIFFERS
---------------------
Recognition and alignment are byte-identical to upstream Immich on both sides:

* ArcFace recognition is the SAME ONNX model (insightface ``buffalo_l`` →
  ``w600k_r50.onnx``), called through the SAME ``src.models.face_embed``
  (``insightface`` ``model_zoo`` ``get_feat`` → ``blobFromImages``), and face
  search uses cosine — so weights and metric match.
* The 112×112 alignment is the SAME ``insightface.utils.face_align.norm_crop``.

The ONLY variable is the **5-point landmarks** fed into ``norm_crop``:

* ``upstream`` — insightface ``buffalo_l`` detector ``det_10g.onnx`` (SCRFD-10GF;
  the bead calls it "RetinaFace" — buffalo_l actually ships SCRFD, and that is
  what upstream/Docker Immich runs) emits 5 keypoints directly.
* ``fork``     — ``src.models.face_detect`` reconstructs 5 points from Apple
  Vision face-landmark *contours* (eye centers, nose anchor, mouth corners =
  min/max-x of outerLips). The nose anchor is selectable via ``--nose``:
  ``tip`` (last nose-contour point — the production default that built
  the existing index) or ``center`` (nose-contour centroid, more robust on
  non-frontal poses). ``--nose both`` runs each and compares median drift. See
  ``face_detect.py``.

Different landmarks → different similarity transform → a slightly different
aligned crop → a drifted ArcFace embedding for the *same* physical face. This
harness quantifies that drift and whether it breaks index/cluster compatibility.

To isolate landmarks as the sole variable, BOTH keypoint sets are run through
the production ``src.models.face_embed.get_face_embedding``; the upstream
detector is loaded with ``allowed_modules=['detection']`` so it contributes
keypoints only, never its own recognition pass.

METRICS
-------
1. **Alignment-drift cosine** — for each physical face detected by *both*
   pipelines (matched by bounding-box IoU on the same image), cosine of the
   upstream-aligned vs fork-aligned ArcFace embedding. This is the core
   preserve-vs-reindex number: a face stored in the index was embedded upstream;
   re-detected by the fork it gets the fork embedding — their cosine is how well
   a new query matches the stored vector.

2. **Top-1 identity agreement** (needs labels; LFW provides them) — build a
   gallery of matched faces; for each face find its nearest neighbour (excluding
   the same source image) and check the predicted identity. Reported for:
   - ``upstream→upstream`` (baseline accuracy of the index as-is upstream),
   - ``fork→fork``         (accuracy if the whole index is rebuilt by the fork),
   - ``fork→upstream``     (THE preserve test: fork-detected query faces
     retrieved against the *stored upstream* index — does identity still hold?).

3. **Detection-set agreement** — Vision detects a different SET of faces than
   SCRFD and its confidence is calibrated differently, so the ``minScore=0.7``
   threshold is not equivalent. Reports matched / upstream-only / fork-only
   counts, plus how many fork faces survive the Immich default ``minScore``.

DATA
----
Default: a labelled subset of LFW (``logasja/lfw`` on HuggingFace — one face per
image, multiple images per identity), filtered to identities with enough images
for the top-1 test. Override with real library photos via ``--images DIR``:
a directory of ``identity/*.jpg`` subdirs (labelled top-1) or a flat dir of
images (drift cosine + detection counts only — no identity metric).

Usage (from ml/, venv active):

    .venv/bin/python scripts/face_embedding_parity.py                  # LFW subset
    .venv/bin/python scripts/face_embedding_parity.py --num-ids 60     # bigger sample
    .venv/bin/python scripts/face_embedding_parity.py --images ~/faces # real library
    .venv/bin/python scripts/face_embedding_parity.py --nose both       # tip vs centroid drift
    .venv/bin/python scripts/face_embedding_parity.py --report out.md

Apple Vision is required (the fork pipeline), so this only runs on macOS.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# Make ``src`` importable when run as a standalone script from anywhere.
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

# Upstream Immich face-detection model pack (SCRFD detector + ArcFace recognizer).
UPSTREAM_PACK = "buffalo_l"
# Immich's default face-detection confidence floor (server minScore).
IMMICH_MIN_SCORE = 0.7
# Default HuggingFace LFW dataset (image + integer identity label).
LFW_REPO = "logasja/lfw"
LFW_FILE = "data/train-00000-of-00001.parquet"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Geometry helpers (pure — unit-testable without models)
# --------------------------------------------------------------------------- #
def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes. 0 if they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    boxes_a: list[tuple[float, float, float, float]],
    boxes_b: list[tuple[float, float, float, float]],
    iou_thresh: float,
) -> list[tuple[int, int]]:
    """Greedily pair boxes by descending IoU (each box used at most once).

    Returns a list of ``(idx_a, idx_b)`` pairs with IoU >= ``iou_thresh``.
    Comparing index-0 to index-0 is wrong when the two detectors find different
    SETS of faces (e.g. SCRFD=3, Vision=2 on the same photo) — they may be
    different physical faces — so every comparison goes through this matcher.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, ba in enumerate(boxes_a):
        for j, bb in enumerate(boxes_b):
            ov = iou(ba, bb)
            if ov >= iou_thresh:
                candidates.append((ov, i, j))
    candidates.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _ov, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (defensively L2-normalized)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def describe(values: list[float]) -> dict[str, float]:
    """min / p1 / p5 / median / mean / max plus pass-fractions for a sample."""
    if not values:
        return {k: float("nan") for k in ("min", "p1", "p5", "median", "mean", "max", "frac_ge_090", "frac_ge_095")}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "frac_ge_090": float((arr >= 0.90).mean()),
        "frac_ge_095": float((arr >= 0.95).mean()),
    }


# --------------------------------------------------------------------------- #
# Sample loading
# --------------------------------------------------------------------------- #
class Sample:
    """One source image: raw bytes + an identity label (or None if unlabelled)."""

    __slots__ = ("data", "label", "name")

    def __init__(self, name: str, label: str | None, data: bytes):
        self.name = name
        self.label = label
        self.data = data


def load_image_dir(root: Path) -> list[Sample]:
    """Load images from a directory.

    ``root/identity/*.jpg`` → labelled by subdirectory name. A flat directory of
    images → unlabelled (label None; top-1 identity metric is skipped).
    """
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    samples: list[Sample] = []
    if subdirs:
        for d in subdirs:
            samples.extend(Sample(f"{d.name}/{f.name}", d.name, f.read_bytes()) for f in sorted(d.iterdir()) if f.suffix.lower() in IMAGE_EXTS)
    else:
        samples.extend(Sample(f.name, None, f.read_bytes()) for f in sorted(root.iterdir()) if f.suffix.lower() in IMAGE_EXTS)
    return samples


def load_lfw(num_ids: int, min_per_id: int, max_per_id: int) -> list[Sample]:
    """Download a labelled LFW subset from HuggingFace and return image samples.

    Selects, deterministically (identities sorted by name), the first
    ``num_ids`` identities having at least ``min_per_id`` images, capped at
    ``max_per_id`` images each. Multiple images per identity are required for
    the top-1 retrieval metric.
    """
    import json

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LFW_REPO, LFW_FILE, repo_type="dataset")
    pf = pq.ParquetFile(path)
    names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]

    labels = pq.read_table(path, columns=["label"])["label"].to_pylist()
    by_id: dict[int, list[int]] = defaultdict(list)
    for row, lab in enumerate(labels):
        by_id[lab].append(row)

    eligible = sorted(
        (lab for lab, rows in by_id.items() if len(rows) >= min_per_id),
        key=lambda lab: names[lab],
    )[:num_ids]
    if not eligible:
        raise SystemExit(f"No LFW identities with >= {min_per_id} images found.")

    wanted_rows: list[int] = []
    for lab in eligible:
        wanted_rows.extend(by_id[lab][:max_per_id])
    wanted_rows.sort()

    table = pq.read_table(path, columns=["label", "image"]).take(wanted_rows)
    imgs = table["image"].to_pylist()
    labs = table["label"].to_pylist()
    samples: list[Sample] = []
    for rec, lab in zip(imgs, labs):
        name = rec.get("path") or f"{names[lab]}.jpg"
        samples.append(Sample(name, names[lab], rec["bytes"]))
    return samples


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def make_upstream_detector(det_size: int):
    """insightface ``buffalo_l`` loaded for DETECTION ONLY (SCRFD keypoints)."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=UPSTREAM_PACK, allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def upstream_faces(app, img_bgr: np.ndarray) -> list[dict]:
    """Detect with SCRFD → list of ``{bbox:(x1,y1,x2,y2), kps:[[x,y]*5], score}``."""
    out = []
    for f in app.get(img_bgr):
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        out.append({"bbox": (x1, y1, x2, y2), "kps": f.kps.tolist(), "score": float(f.det_score)})
    return out


def fork_faces(image_bytes: bytes, nose_strategy: str = "tip") -> list[dict]:
    """Detect with Apple Vision → same shape, keeping only faces with landmarks.

    ``nose_strategy`` selects the nose anchor for the reconstructed 5-point
    landmarks (see ``face_detect.NOSE_STRATEGIES``): "tip" reproduces the
    production index (last nose-contour point); "center" is the candidate
    (nose-contour centroid) measured here for alignment drift on hard poses.
    """
    from src.models.face_detect import detect_faces

    faces, _w, _h = detect_faces(image_bytes, nose_strategy=nose_strategy)
    out = []
    for f in faces:
        if "landmarks" not in f:
            continue
        bb = f["boundingBox"]
        out.append(
            {
                "bbox": (float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])),
                "kps": f["landmarks"],
                "score": float(f["score"]),
            }
        )
    return out


def embed(image_bytes: bytes, kps: list[list[float]]) -> np.ndarray:
    """ArcFace embedding for one keypoint set via the production face_embed path."""
    from src.models.face_embed import get_face_embedding

    return get_face_embedding(image_bytes, kps, model_name=UPSTREAM_PACK)


# --------------------------------------------------------------------------- #
# Top-1 identity retrieval
# --------------------------------------------------------------------------- #
def top1_accuracy(
    query_emb: np.ndarray,
    query_lab: list[str],
    query_img: list[int],
    gallery_emb: np.ndarray,
    gallery_lab: list[str],
    gallery_img: list[int],
) -> float:
    """Top-1 identity accuracy: nearest gallery face (excluding same source image).

    ``*_img`` are per-face source-image ids so a face never retrieves itself or
    another crop from the same photo. Embeddings are L2-normalized, so a dot
    product is cosine; we mask same-image candidates to -inf before argmax.
    """
    if len(query_emb) == 0 or len(gallery_emb) == 0:
        return float("nan")
    sims = query_emb @ gallery_emb.T  # (Q, G) cosine
    g_img = np.array(gallery_img)
    correct = 0
    counted = 0
    for qi in range(len(query_emb)):
        row = sims[qi].copy()
        row[g_img == query_img[qi]] = -np.inf  # exclude same source image
        if not np.isfinite(row).any():
            continue
        counted += 1
        if gallery_lab[int(row.argmax())] == query_lab[qi]:
            correct += 1
    return correct / counted if counted else float("nan")


# --------------------------------------------------------------------------- #
# Per-strategy evaluation
# --------------------------------------------------------------------------- #
class StrategyResult(NamedTuple):
    """Metrics for one fork nose-anchor strategy vs the upstream pipeline."""

    drift: list[float]
    n_up: int
    n_fork: int
    n_matched: int
    n_up_only: int
    n_fork_only: int
    n_fork_minscore: int
    up_acc: float  # upstream→upstream top-1 (index as-is)
    fk_acc: float  # fork→fork top-1 (full re-index)
    cross_acc: float  # fork→upstream top-1 (THE preserve test)


def evaluate(samples, app, iou: float, nose_strategy: str, labelled: bool) -> StrategyResult:
    """Run the fork (Apple Vision, ``nose_strategy``) vs upstream (SCRFD) pipeline
    over ``samples`` and return drift + detection-agreement + top-1 metrics.

    Upstream detection/embedding is recomputed per call; for ``--nose both`` that
    repeats the upstream side, which keeps this an offline-eval simplicity win
    (clarity over caching the SCRFD pass)."""
    drift: list[float] = []
    n_up = n_fork = n_matched = n_up_only = n_fork_only = n_fork_minscore = 0
    up_emb: list[np.ndarray] = []
    fk_emb: list[np.ndarray] = []
    lab: list[str] = []
    img_id: list[int] = []

    for idx, s in enumerate(samples):
        arr = np.frombuffer(s.data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [skip] undecodable image: {s.name}")
            continue
        ups = upstream_faces(app, img_bgr)
        fks = fork_faces(s.data, nose_strategy)
        n_up += len(ups)
        n_fork += len(fks)
        n_fork_minscore += sum(1 for f in fks if f["score"] >= IMMICH_MIN_SCORE)

        pairs = greedy_match([u["bbox"] for u in ups], [f["bbox"] for f in fks], iou)
        n_matched += len(pairs)
        n_up_only += len(ups) - len(pairs)
        n_fork_only += len(fks) - len(pairs)

        for ui, fi in pairs:
            try:
                ue = embed(s.data, ups[ui]["kps"])
                fe = embed(s.data, fks[fi]["kps"])
            except Exception as e:  # alignment/inference failure on one face shouldn't kill the run
                print(f"  [warn] embed failed on {s.name}: {e}")
                continue
            drift.append(cosine(ue, fe))
            if s.label is not None:
                up_emb.append(ue)
                fk_emb.append(fe)
                lab.append(s.label)
                img_id.append(idx)

        if (idx + 1) % 25 == 0:
            print(f"  [nose={nose_strategy}] [{idx + 1}/{len(samples)}] matched faces so far: {len(drift)}")

    up_acc = fk_acc = cross_acc = float("nan")
    if labelled and up_emb:
        U = np.vstack(up_emb)
        F = np.vstack(fk_emb)
        up_acc = top1_accuracy(U, lab, img_id, U, lab, img_id)
        fk_acc = top1_accuracy(F, lab, img_id, F, lab, img_id)
        # THE preserve test: fork-detected queries against the stored upstream index.
        cross_acc = top1_accuracy(F, lab, img_id, U, lab, img_id)

    return StrategyResult(
        drift, n_up, n_fork, n_matched, n_up_only, n_fork_only, n_fork_minscore, up_acc, fk_acc, cross_acc
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=None, help="dir of real images (identity/*.jpg subdirs, or flat); else LFW")
    ap.add_argument("--num-ids", type=int, default=40, help="LFW: number of identities to sample")
    ap.add_argument("--min-per-id", type=int, default=5, help="LFW: minimum images per identity")
    ap.add_argument("--max-per-id", type=int, default=10, help="LFW: cap images per identity")
    ap.add_argument("--iou", type=float, default=0.30, help="IoU threshold to call two detections the same face")
    ap.add_argument("--det-size", type=int, default=640, help="SCRFD detector input size")
    ap.add_argument("--min-cos", type=float, default=0.90, help="drift-cosine median required to recommend PRESERVE")
    ap.add_argument("--max-top1-drop", type=float, default=0.02, help="max allowed fork→upstream top-1 drop vs upstream baseline for PRESERVE")
    ap.add_argument("--report", type=Path, default=None, help="write a markdown report here")
    ap.add_argument(
        "--nose",
        choices=["tip", "center", "both"],
        default="tip",
        help="fork nose-anchor strategy: 'tip' (production, last contour point), "
        "'center' (candidate, contour centroid), or 'both' to compare drift (default: tip)",
    )
    args = ap.parse_args()

    # --- load samples ---
    if args.images:
        samples = load_image_dir(args.images)
        source = f"images dir {args.images}"
    else:
        samples = load_lfw(args.num_ids, args.min_per_id, args.max_per_id)
        source = f"LFW ({LFW_REPO}) — {args.num_ids} ids, {args.min_per_id}-{args.max_per_id} imgs/id"
    if not samples:
        raise SystemExit("No images loaded.")
    labelled = any(s.label is not None for s in samples)
    print(f"[setup] {len(samples)} images from {source}; labelled={labelled}; IoU>={args.iou}")

    app = make_upstream_detector(args.det_size)

    from src.models.face_detect import NOSE_STRATEGIES

    strategies = list(NOSE_STRATEGIES) if args.nose == "both" else [args.nose]

    # --- report ---
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit("FACE-EMBEDDING PARITY: fork (Apple Vision landmarks) vs upstream (SCRFD)")
    emit("=" * 78)
    emit(f"Source: {source}")
    emit(f"Recognition: insightface {UPSTREAM_PACK} ArcFace (w600k_r50) — IDENTICAL both sides")
    emit("Alignment: face_align.norm_crop 112x112 — IDENTICAL both sides")
    emit("Sole variable: 5-point landmarks (SCRFD keypoints vs Vision contours)")
    emit(f"Fork nose anchor(s): {', '.join(strategies)}")

    def report_strategy(strategy: str, res: StrategyResult) -> tuple[int, float]:
        """Emit one strategy's blocks; return (rc, median-drift)."""
        d = describe(res.drift)
        emit()
        emit("#" * 78)
        emit(f"# NOSE ANCHOR: {strategy}")
        emit("#" * 78)
        emit("### Detection-set agreement")
        emit(f"  upstream (SCRFD) faces : {res.n_up}")
        emit(f"  fork (Vision) faces    : {res.n_fork}  (>= minScore {IMMICH_MIN_SCORE}: {res.n_fork_minscore})")
        emit(f"  matched (IoU>={args.iou})    : {res.n_matched}")
        emit(f"  upstream-only          : {res.n_up_only}  (SCRFD found, Vision missed)")
        emit(f"  fork-only              : {res.n_fork_only}  (Vision found, SCRFD missed)")
        if res.n_up:
            emit(f"  Vision recall vs SCRFD : {res.n_matched / res.n_up:.3f}")
        emit()
        emit("### Alignment-drift cosine (matched faces, identical ArcFace)")
        emit(f"  n={len(res.drift)}")
        emit(
            f"  min={d['min']:.4f}  p1={d['p1']:.4f}  p5={d['p5']:.4f}  "
            f"median={d['median']:.4f}  mean={d['mean']:.4f}  max={d['max']:.4f}"
        )
        emit(f"  frac >= 0.90: {d['frac_ge_090']:.3f}   frac >= 0.95: {d['frac_ge_095']:.3f}")
        emit()
        emit("### Top-1 identity agreement")
        if labelled and np.isfinite(res.up_acc):
            emit(f"  upstream→upstream (index as-is)      : {res.up_acc:.4f}")
            emit(f"  fork→fork (full re-index by fork)    : {res.fk_acc:.4f}")
            emit(f"  fork→upstream (PRESERVE test)        : {res.cross_acc:.4f}")
            if np.isfinite(res.up_acc) and np.isfinite(res.cross_acc):
                emit(f"  preserve drop (upstream - cross)     : {res.up_acc - res.cross_acc:+.4f}")
        else:
            emit("  (skipped — needs labelled identities; use LFW or identity/*.jpg subdirs)")

        emit()
        emit("-" * 78)
        preserve_ok = bool(np.isfinite(d["median"]) and d["median"] >= args.min_cos)
        top1_ok = True
        if labelled and np.isfinite(res.up_acc) and np.isfinite(res.cross_acc):
            top1_ok = (res.up_acc - res.cross_acc) <= args.max_top1_drop
        if not np.isfinite(d["median"]):
            emit(f"VERDICT [nose={strategy}]: INCONCLUSIVE — no matched faces. Check IoU / detectors / data.")
            rc = 2
        elif preserve_ok and top1_ok:
            emit(f"VERDICT [nose={strategy}]: PRESERVE existing face index & clusters (median drift {d['median']:.4f} >= {args.min_cos}).")
            emit("  Fork-detected faces match stored upstream embeddings closely; new faces")
            emit("  will join the right clusters. A re-scan is NOT required for compatibility.")
            if labelled and np.isfinite(res.up_acc):
                emit(f"  fork→upstream top-1 within {args.max_top1_drop:.0%} of the upstream baseline.")
            rc = 0
        else:
            emit(f"VERDICT [nose={strategy}]: RE-RUN face recognition (median drift {d['median']:.4f}; preserve_ok={preserve_ok}, top1_ok={top1_ok}).")
            emit("  Landmark divergence drifts embeddings enough to risk cluster splits;")
            emit("  re-embed the library with the fork pipeline for a consistent index.")
            rc = 1
        emit("-" * 78)
        return rc, d["median"]

    rc_by_strategy: dict[str, int] = {}
    median_by_strategy: dict[str, float] = {}
    for strategy in strategies:
        res = evaluate(samples, app, args.iou, strategy, labelled)
        rc, median = report_strategy(strategy, res)
        rc_by_strategy[strategy] = rc
        median_by_strategy[strategy] = median

    # --- nose-strategy comparison: does the centroid reduce drift? ---
    if len(strategies) > 1 and all(np.isfinite(median_by_strategy[s]) for s in strategies):
        emit()
        emit("### Nose-anchor comparison (higher median drift cosine = better alignment)")
        for s in strategies:
            emit(f"  nose={s:<6} median drift cosine: {median_by_strategy[s]:.4f}")
        delta = median_by_strategy["center"] - median_by_strategy["tip"]
        better = "center" if delta > 0 else "tip"
        emit(f"  Δ(center - tip) = {delta:+.4f} → '{better}' aligns closer to the upstream index.")
        emit("  ADOPT 'center' only if it improves median drift on a HARD-POSE sample by a")
        emit("  margin worth a re-index; on frontal LFW the two are expected to be ~equal.")

    # rc reflects the production-default strategy ("tip") when present, else the run.
    rc = rc_by_strategy.get("tip", next(iter(rc_by_strategy.values())))

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\n[report] written to {args.report}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
