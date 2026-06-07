#!/usr/bin/env python3
"""Quantization evaluation for the native MLX SigLIP2 backend.

Measures **latency, memory and embedding accuracy** across weight precisions and
picks a default. The baseline is the production fp16 convert; every
other precision is compared against it because fp16 is what the existing 4716-row
smart-search index was (or will be) built with — so "accuracy" here means *cosine
agreement with fp16*, the drift a user's search results would actually see.

What it does, per precision config:

  1. Build the converted dir if missing (``mlx_embeddings.convert``, quantize=...),
     named with the loader's required ``patchNN-NNN`` token + a precision suffix
     (e.g. ``siglip2-so400m-patch16-384-4bit``). The loader regex
     ``patch\\d+-(\\d+)(?:-|$)`` tolerates the suffix (verified during conversion testing).
  2. Load it through the **production path** (``src.models.clip.MLXClip`` with the
     ``ML_SIGLIP2_MLX_PATH`` override) — so we measure exactly what Immich runs,
     not a bespoke loader.
  3. Embed the same images + text queries as the parity harness, timing each
     encode and recording MLX peak memory (``mx.get_peak_memory``) and on-disk
     weight bytes.
  4. Cosine-compare image/text embeddings against the fp16 baseline, plus
     cross-modal top-1 retrieval agreement (the user-facing "do searches still
     rank the same?" question).

The crucial knob is **skip_vision**. ``mlx_embeddings.convert`` defaults to
``skip_vision=True`` (quantize the text tower only). Smart-search *indexing* is
all image encodes, so text-only quantization barely moves indexing memory/latency
and leaves image-embedding accuracy at fp16. Quantizing the vision tower
(``skip_vision=False``) is where the real memory/latency win — and the real
accuracy risk — lives. This harness evaluates both so the default is an informed
choice, not an inherited library default.

Usage (from ml/, venv active):

    .venv/bin/python scripts/quantization_eval.py                  # default matrix
    .venv/bin/python scripts/quantization_eval.py --images ~/Pics  # real photos
    .venv/bin/python scripts/quantization_eval.py --configs fp16 8bit 4bit
    .venv/bin/python scripts/quantization_eval.py --report out.md --keep

By default the converted quant dirs are written under ``models/quant_eval/`` and
removed at the end (``--keep`` to retain them, e.g. to promote one to the cache).
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make ``src`` and sibling scripts importable when run from anywhere.
ML_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(ML_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the parity harness' loaders/metrics so both tools test identical inputs.
import embedding_parity as parity
import mlx.core as mx

IMMICH_MODEL = parity.IMMICH_MODEL  # "ViT-SO400M-16-SigLIP2-384__webli"
EMBED_DIM = parity.EMBED_DIM  # 1152


# --------------------------------------------------------------------------- #
# Precision configs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Precision:
    """A weight-precision variant to convert + evaluate.

    ``key`` is the report label and the converted-dir suffix. ``quantize`` /
    ``bits`` / ``group_size`` / ``skip_vision`` feed ``mlx_embeddings.convert``.
    fp16 is the un-quantized baseline everything else is scored against.
    """

    key: str
    quantize: bool
    bits: int = 16
    group_size: int = 64
    skip_vision: bool = True
    note: str = ""


# Config catalog. The text-only (skip_vision=True) variants are the only ones
# mlx_embeddings 0.1.0 can actually run: its SigLIP vision pooling head
# (MultiheadAttentionPoolingHead) slices `in_proj.weight` directly, which is
# invalid for a QuantizedLinear, so quantizing the vision tower
# (skip_vision=False) converts fine but crashes at image-encode time. The
# full-vision variants are kept in the catalog (opt-in via --configs) so the
# eval can *demonstrate and document* that limitation rather than hide it; the
# harness records the failure instead of crashing.
CONFIGS: dict[str, Precision] = {
    "fp16": Precision("fp16", quantize=False, bits=16, note="baseline (production convert)"),
    "8bit-textonly": Precision("8bit-textonly", quantize=True, bits=8, skip_vision=True, note="affine, text tower only (mlx default; vision stays fp16)"),
    "4bit-textonly": Precision("4bit-textonly", quantize=True, bits=4, skip_vision=True, note="affine, text tower only (mlx default; vision stays fp16)"),
    "8bit": Precision("8bit", quantize=True, bits=8, skip_vision=False, note="affine vision+text — UNSUPPORTED by mlx_embeddings 0.1.0 vision head"),
    "4bit": Precision("4bit", quantize=True, bits=4, skip_vision=False, note="affine vision+text — UNSUPPORTED by mlx_embeddings 0.1.0 vision head"),
}

# What we evaluate unless --configs overrides: the runnable set.
DEFAULT_KEYS = ["fp16", "8bit-textonly", "4bit-textonly"]


@dataclass
class Result:
    key: str
    note: str
    disk_bytes: int
    peak_mem_bytes: int
    img_ms: float  # mean per-image encode latency
    txt_ms: float  # mean per-text encode latency
    img_embeds: np.ndarray | None = field(repr=False, default=None)
    txt_embeds: np.ndarray | None = field(repr=False, default=None)
    # Filled in vs the fp16 baseline (None on the baseline row itself):
    img_cos: dict | None = None
    txt_cos: dict | None = None
    agree: dict | None = None
    error: str | None = None  # set if the config converted but failed to infer


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def ensure_convert(cfg: Precision, source: str, out_root: Path) -> Path:
    """Return a converted dir for ``cfg``, building it if absent.

    fp16 reuses the production cache dir if it's already complete (no need to
    re-convert 2 GB). Quantized variants are written under ``out_root`` with a
    name carrying the ``patchNN-NNN`` token + precision suffix.
    """
    from src.models.clip import siglip2_cache_dir, siglip2_dir_is_complete

    base = siglip2_cache_dir(parity.HF_REPO)  # .../siglip2-so400m-patch16-384

    if not cfg.quantize:
        if siglip2_dir_is_complete(base):
            print(f"[convert] fp16: reusing production cache {base}")
            return base
        out = out_root / base.name
    else:
        out = out_root / f"{base.name}-{cfg.key}"

    if siglip2_dir_is_complete(out):
        print(f"[convert] {cfg.key}: reusing existing {out}")
        # A dir from an interrupted run (or older code) may still carry the
        # skip_vision key that crashes the loader — strip it before reuse.
        _strip_skip_vision_key(out)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    from mlx_embeddings.convert import convert

    print(f"[convert] {cfg.key}: {source} -> {out} (quantize={cfg.quantize} bits={cfg.bits} skip_vision={cfg.skip_vision})")
    convert(
        hf_path=source,
        mlx_path=str(out),
        quantize=cfg.quantize,
        q_bits=cfg.bits,
        q_group_size=cfg.group_size,
        dtype="float16",
        skip_vision=cfg.skip_vision,
    )
    if not siglip2_dir_is_complete(out):
        raise SystemExit(f"[convert] FAILED: {out} incomplete after convert")
    _strip_skip_vision_key(out)
    return out


def _strip_skip_vision_key(out: Path) -> None:
    """Work around an mlx_embeddings 0.1.0 convert/load incompatibility.

    ``convert(quantize=True, skip_vision=...)`` writes ``skip_vision`` into
    ``config.json``'s ``vision_config``, but ``load_model`` then calls
    ``VisionConfig(**vision_config)`` which rejects the extra kwarg with
    ``TypeError: ... unexpected keyword argument 'skip_vision'`` — so the
    library cannot load its own quantized output. The key is redundant for
    loading: the loader's ``class_predicate`` already quantizes exactly the
    modules that carry ``*.scales`` in the weights, so dropping it reproduces
    the intended (vision-skipped or not) layout. Idempotent.
    """
    import json

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    vc = cfg.get("vision_config")
    if isinstance(vc, dict) and "skip_vision" in vc:
        vc.pop("skip_vision")
        cfg_path.write_text(json.dumps(cfg, indent=2))


def dir_weight_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.glob("*.safetensors"))


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def evaluate(
    cfg: Precision,
    convert_dir: Path,
    images: list[tuple[str, bytes]],
    queries: list[str],
) -> Result:
    """Load via the production MLXClip and measure latency/memory/embeddings."""
    from src.models.clip import MLXClip

    # Point the production resolver at this exact dir.
    os.environ["ML_SIGLIP2_MLX_PATH"] = str(convert_dir)

    mx.clear_cache()
    mx.reset_peak_memory()

    clip = MLXClip(IMMICH_MODEL)
    if not getattr(clip, "_use_mlx_embeddings", False):
        return Result(
            key=cfg.key,
            note=cfg.note,
            disk_bytes=dir_weight_bytes(convert_dir),
            peak_mem_bytes=0,
            img_ms=0.0,
            txt_ms=0.0,
            error="backend fell back off the native MLX path (load failed)",
        )

    try:
        # Warm up (first call pays lazy-eval + graph build; don't time it).
        clip.encode_image(images[0][1])
        clip.encode_text(queries[0])

        img_embeds, img_times = [], []
        for _, b in images:
            t0 = time.perf_counter()
            e = clip.encode_image(b)
            img_times.append((time.perf_counter() - t0) * 1000.0)
            img_embeds.append(e)

        txt_embeds, txt_times = [], []
        for q in queries:
            t0 = time.perf_counter()
            e = clip.encode_text(q)
            txt_times.append((time.perf_counter() - t0) * 1000.0)
            txt_embeds.append(e)

        peak = mx.get_peak_memory()
        img = np.stack(img_embeds)
        txt = np.stack(txt_embeds)
        assert img.shape[1] == EMBED_DIM, f"unexpected dim {img.shape}"
    except Exception as e:  # converted OK but inference unsupported (e.g. quantized vision head)
        print(f"[{cfg.key}] inference FAILED: {type(e).__name__}: {e}")
        return Result(
            key=cfg.key,
            note=cfg.note,
            disk_bytes=dir_weight_bytes(convert_dir),
            peak_mem_bytes=0,
            img_ms=0.0,
            txt_ms=0.0,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        clip.unload()
        gc.collect()
        mx.clear_cache()

    return Result(
        key=cfg.key,
        note=cfg.note,
        disk_bytes=dir_weight_bytes(convert_dir),
        peak_mem_bytes=peak,
        img_ms=float(np.mean(img_times)),
        txt_ms=float(np.mean(txt_times)),
        img_embeds=img,
        txt_embeds=txt,
    )


def score_vs_baseline(res: Result, base: Result) -> None:
    """Fill cosine/agreement fields comparing ``res`` to the fp16 baseline."""
    assert res.img_embeds is not None and res.txt_embeds is not None
    assert base.img_embeds is not None and base.txt_embeds is not None
    img_sims = np.array([parity.cosine(res.img_embeds[i], base.img_embeds[i]) for i in range(len(base.img_embeds))])
    txt_sims = np.array([parity.cosine(res.txt_embeds[j], base.txt_embeds[j]) for j in range(len(base.txt_embeds))])
    res.img_cos = parity.stats(img_sims)
    res.txt_cos = parity.stats(txt_sims)
    res.agree = parity.retrieval_agreement(res.img_embeds, res.txt_embeds, base.img_embeds, base.txt_embeds)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def gib(n: int) -> float:
    return n / (1024**3)


def pct_save(new: int, base: int) -> str:
    """A percentage reduction of ``new`` vs ``base``, or "n/a" if unmeasurable.

    ``base`` can be 0 — e.g. ``mx.get_peak_memory()`` returns 0 on some paths —
    so guard the division rather than crash (or report a misleading 100%).
    """
    if base <= 0:
        return "n/a"
    return f"{(1 - new / base) * 100:.0f}%"


def recommend(results: list[Result]) -> tuple[str, list[str]]:
    """Recommend the production default and (if any) a safe opt-in, with reasons.

    Bars a quantized variant must clear to be a usable *opt-in*:
      * image cosine vs fp16 >= 0.99 (the parity-harness index gate), and
      * top-1 retrieval agreement >= 0.999 (search rankings unchanged).

    But the *default* stays fp16 unless a variant also delivers a meaningful win
    on the **image-indexing path** — the dominant smart-search workload. With
    mlx_embeddings 0.1.0 the vision tower cannot be quantized (its SigLIP pooling
    head indexes a raw ``.weight``), so every runnable variant is text-only and
    leaves image latency at fp16. Such a variant is reported as an OPT-IN memory
    saver, not a new default, because changing the shipped default for a
    text-tower-only saving isn't worth the conversion/compat surface.
    """
    IMG_GATE = 0.99
    rationale: list[str] = []
    base = next(r for r in results if r.key == "fp16")

    rationale.extend(f"{r.key}: UNSUPPORTED — {r.error}" for r in results if r.error)

    candidates = sorted(
        (r for r in results if r.key != "fp16" and not r.error),
        key=lambda r: r.disk_bytes,
    )
    usable: list[Result] = []
    for r in candidates:
        assert r.img_cos and r.txt_cos and r.agree  # set for non-error candidates
        ok_img = r.img_cos["min"] >= IMG_GATE
        ok_rank = r.agree["top1_agreement"] >= 0.999
        if ok_img and ok_rank:
            usable.append(r)
        rationale.append(
            f"{r.key}: img cos={r.img_cos['min']:.4f} (>= {IMG_GATE}), "
            f"top1={r.agree['top1_agreement']:.3f} (>= 0.999), "
            f"txt cos={r.txt_cos['min']:.4f}, {gib(r.disk_bytes):.2f} GiB, "
            f"peak {gib(r.peak_mem_bytes):.2f} GiB -> "
            f"{'usable' if (ok_img and ok_rank) else 'rejected'}"
        )
    # The default is always fp16 here: no runnable variant touches the image path
    # (vision quant is unsupported), so none earns a default change on its own.
    if usable:
        opt = usable[0]  # smallest retrieval-preserving variant
        disk_save = pct_save(opt.disk_bytes, base.disk_bytes)
        mem_save = pct_save(opt.peak_mem_bytes, base.peak_mem_bytes)
        rationale.append("=> DEFAULT: fp16 (unchanged) — quantization can't reach the vision tower, so the image-indexing path gets no speed/accuracy benefit.")
        rationale.append(
            f"=> OPT-IN: {opt.key} — near-lossless ({disk_save} smaller disk, "
            f"{mem_save} lower peak memory, retrieval identical) for "
            f"memory-constrained installs. Revisit a real default change when "
            f"mlx_embeddings can quantize the SigLIP vision attention."
        )
    else:
        rationale.append("=> DEFAULT: fp16 — no runnable quantized variant preserved retrieval; the savings don't justify index/search drift.")
    return "fp16", rationale


def build_report(results: list[Result], images, queries) -> list[str]:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    base = next(r for r in results if r.key == "fp16")
    emit()
    emit("=" * 88)
    emit("QUANTIZATION EVALUATION: native MLX SigLIP2 (vs fp16 baseline)")
    emit("=" * 88)
    emit(f"Model: {IMMICH_MODEL}  |  images: {len(images)}  queries: {len(queries)}")
    emit("Accuracy = cosine vs the fp16 production convert (the index baseline).")
    emit()

    # Headline table.
    hdr = f"{'precision':<16} {'disk GiB':>9} {'peak GiB':>9} {'img ms':>8} {'txt ms':>8} {'img mincos':>11} {'img mean':>9} {'txt mincos':>11} {'top1':>6}"
    emit(hdr)
    emit("-" * len(hdr))
    for r in results:
        if r.error:
            emit(f"{r.key:<16} {gib(r.disk_bytes):>9.2f} {'—':>9} {'—':>8} {'—':>8} {'UNSUPPORTED (see notes)':>40}")
        elif r.key == "fp16":
            emit(f"{r.key:<16} {gib(r.disk_bytes):>9.2f} {gib(r.peak_mem_bytes):>9.2f} {r.img_ms:>8.1f} {r.txt_ms:>8.1f} {'—':>11} {'(base)':>9} {'—':>11} {'—':>6}")
        else:
            assert r.img_cos and r.txt_cos and r.agree
            emit(
                f"{r.key:<16} {gib(r.disk_bytes):>9.2f} {gib(r.peak_mem_bytes):>9.2f} "
                f"{r.img_ms:>8.1f} {r.txt_ms:>8.1f} {r.img_cos['min']:>11.4f} "
                f"{r.img_cos['mean']:>9.4f} {r.txt_cos['min']:>11.4f} "
                f"{r.agree['top1_agreement']:>6.3f}"
            )
    emit()
    emit("Notes:")
    for r in results:
        emit(f"  {r.key:<16} {r.note}")
    emit()

    # Per-config detail.
    for r in results:
        if r.key == "fp16":
            continue
        if r.error:
            emit(f"### {r.key} vs fp16")
            emit(f"  UNSUPPORTED: converted to {gib(r.disk_bytes):.2f} GiB but failed at inference:\n    {r.error}")
            emit()
            continue
        assert r.img_cos and r.txt_cos and r.agree
        emit(f"### {r.key} vs fp16")
        emit(f"  IMAGE cosine: min={r.img_cos['min']:.4f} mean={r.img_cos['mean']:.4f} median={r.img_cos['median']:.4f}")
        emit(f"  TEXT  cosine: min={r.txt_cos['min']:.4f} mean={r.txt_cos['mean']:.4f} median={r.txt_cos['median']:.4f}")
        emit(f"  Cross-modal: top-1 retrieval agreement={r.agree['top1_agreement']:.3f} matrix_corr={r.agree['matrix_corr']:.4f}")
        emit(f"  Size: {gib(r.disk_bytes):.2f} GiB on disk ({gib(base.disk_bytes):.2f} fp16), peak {gib(r.peak_mem_bytes):.2f} GiB")
        emit()

    emit("-" * 88)
    pick, rationale = recommend(results)
    for ln in rationale:
        emit("  " + ln)
    emit("-" * 88)
    emit(f"RECOMMENDED DEFAULT PRECISION: {pick}")
    return lines


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=None, help="dir of real images (else download deterministic samples)")
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--queries-file", type=Path, default=None)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_KEYS,
        choices=list(CONFIGS.keys()),
        help="precision variants to evaluate (fp16 always included). Default: the runnable set. The full-vision '8bit'/'4bit' are opt-in and expected to fail at inference on mlx_embeddings 0.1.0.",
    )
    ap.add_argument("--out-root", type=Path, default=ML_ROOT / "models" / "quant_eval", help="where quantized converts are written")
    ap.add_argument("--keep", action="store_true", help="keep converted quant dirs instead of deleting them")
    ap.add_argument("--report", type=Path, default=None, help="write a markdown report here")
    ap.add_argument("--cache-dir", type=Path, default=ML_ROOT / "cache" / "parity_images")
    args = ap.parse_args()

    queries = parity.DEFAULT_QUERIES
    if args.queries_file:
        queries = [ln.strip() for ln in args.queries_file.read_text().splitlines() if ln.strip()]

    images, _used_synthetic = parity.load_images(args.images, args.num_images, args.cache_dir)

    # fp16 is always the baseline and must run first.
    keys = ["fp16"] + [k for k in args.configs if k != "fp16"]
    configs = [CONFIGS[k] for k in keys]
    print(f"[setup] {len(images)} images x {len(queries)} queries; configs={keys}")

    # Source for quantized converts: the fp16 cache if present (avoids a 2 GB
    # re-download), else the HF repo. Both carry the patch token the loader needs.
    from src.models.clip import siglip2_cache_dir, siglip2_dir_is_complete

    base_dir = siglip2_cache_dir(parity.HF_REPO)
    quant_source = str(base_dir) if siglip2_dir_is_complete(base_dir) else parity.HF_REPO
    print(f"[setup] quantization source weights: {quant_source}")

    results: list[Result] = []
    made_dirs: list[Path] = []
    saved_override = os.environ.get("ML_SIGLIP2_MLX_PATH")
    try:
        for cfg in configs:
            print(f"\n[backend] {cfg.key} ...")
            cdir = ensure_convert(cfg, quant_source, args.out_root)
            if cfg.quantize and args.out_root in cdir.parents:
                made_dirs.append(cdir)
            results.append(evaluate(cfg, cdir, images, queries))
    finally:
        if saved_override is None:
            os.environ.pop("ML_SIGLIP2_MLX_PATH", None)
        else:
            os.environ["ML_SIGLIP2_MLX_PATH"] = saved_override

    base = next(r for r in results if r.key == "fp16")
    if base.error:
        raise SystemExit(f"[fatal] fp16 baseline failed: {base.error}")
    for r in results:
        if r.key != "fp16" and not r.error:
            score_vs_baseline(r, base)

    lines = build_report(results, images, queries)

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\n[report] written to {args.report}")

    if made_dirs and not args.keep:
        for d in made_dirs:
            shutil.rmtree(d, ignore_errors=True)
        print(f"[cleanup] removed {len(made_dirs)} quant dirs (use --keep to retain)")
    elif made_dirs:
        print("[keep] retained quant dirs:")
        for d in made_dirs:
            print(f"  {d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
