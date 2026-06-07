#!/usr/bin/env python3
"""CLIP throughput/latency benchmark: the production mlx_clip path vs upstream ONNX.

The companion ``clip_parity.py`` proved mlx_clip is *correct* (cosine 1.0000 vs
the open_clip checkpoint Immich exports to ONNX) for the OpenAI CLIP ports, but
speed was never measured. This harness answers the open question from that work:
is mlx_clip's MLX/Metal path actually "in line with upstream performance", or
materially slower than the standard Immich ML server's ONNX path?

It times WARM single-item encodes — the production serving pattern, where Immich
sends one image (or one text query) at a time — for both backends and reports
median/p90 latency and throughput (items/s):

* ``mlx_clip`` — THE CANDIDATE: the production path (``src.models.clip`` ->
                 mlx_clip backend) running on the Apple-Silicon GPU via Metal.
                 Timed two ways: END-TO-END (the public ``encode_image`` /
                 ``encode_text`` — decode + preprocess + Metal forward + L2) and
                 FORWARD-ONLY (just the Metal forward, preprocessing factored
                 out) to localise where the time goes.
* ``onnx``     — THE BASELINE: the SAME model's upstream ONNX export
                 (``immich-app/<model>`` on the Hub — ``visual/model.onnx`` +
                 ``textual/model.onnx``, the exact files the standard Immich ML
                 server loads) under onnxruntime. Run on every requested
                 execution provider. ``CPUExecutionProvider`` is what Immich
                 actually uses in its Docker image on Apple-Silicon (no CUDA),
                 so it is the honest apples-to-apples baseline;
                 ``CoreMLExecutionProvider`` is reported too as the Mac-native
                 accelerated ONNX point.

Note: Immich's DEFAULT model is SigLIP2 (native mlx-embeddings backend), so the
OpenAI CLIP ports benchmarked here are a SECONDARY path — scope conclusions
accordingly. The "direct-MLX baseline" the bead mentions (a hand-rolled MLX CLIP
reusing immich_preprocess) does not exist; mlx_clip *is* the MLX implementation,
so the meaningful comparison is mlx_clip(Metal) vs ONNX(CPU/CoreML). Building a
faster MLX backend is only worth it if mlx_clip loses badly to ONNX here.

Backends load -> warm up -> time -> free sequentially, bounding peak memory so
this runs on a single Mac.

Usage (from ml/, venv active):

    .venv/bin/python scripts/clip_benchmark.py                          # both default models, download a sample image
    .venv/bin/python scripts/clip_benchmark.py --image ~/Pics/cat.jpg   # a real library photo
    .venv/bin/python scripts/clip_benchmark.py --models ViT-B-16__openai
    .venv/bin/python scripts/clip_benchmark.py --providers CPUExecutionProvider CoreMLExecutionProvider
    .venv/bin/python scripts/clip_benchmark.py --no-onnx                # mlx_clip numbers only (no download)
    .venv/bin/python scripts/clip_benchmark.py --report bench.md
"""

from __future__ import annotations

import argparse
import gc
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make ``src`` importable and reuse the parity harnesses' helpers (scripts/ is
# not a package, so add both dirs to the path).
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from clip_parity import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD, _reference_arch
from embedding_parity import load_images

DEFAULT_MODELS = ["ViT-B-16__openai", "ViT-L-14__openai"]
DEFAULT_QUERY = "a photo of a dog playing in the park"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _bench(fn, warmup: int, iters: int) -> dict[str, float]:
    """Run ``fn`` ``warmup`` times (untimed) then ``iters`` times (timed).

    Returns latency stats in milliseconds plus throughput in items/s. ``fn``
    must fully realise its result (e.g. force lazy MLX eval) so the timing
    captures real compute, not deferred work.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = statistics.median(samples)
    p90 = samples[min(len(samples) - 1, round(0.9 * (len(samples) - 1)))]
    return {
        "median_ms": median,
        "mean_ms": statistics.fmean(samples),
        "p90_ms": p90,
        "min_ms": samples[0],
        "throughput": 1000.0 / median if median > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# mlx_clip backend (production path)
# --------------------------------------------------------------------------- #
def bench_mlxclip(model_name: str, image_bytes: bytes, query: str, warmup: int, iters: int) -> dict:
    """Benchmark the production src.models.clip MLXClip (mlx_clip backend)."""
    import mlx.core as mx

    from src.models.clip import get_clip_model
    from src.models.immich_preprocess import clean_text

    model = get_clip_model(model_name)
    if getattr(model, "_use_mlx_embeddings", False) or getattr(model, "_use_fallback", False):
        raise SystemExit(f"{model_name!r} did not load via the mlx_clip backend (got the native SigLIP2 or open_clip path). clip_benchmark.py covers the mlx_clip OpenAI CLIP path only.")

    out: dict[str, dict] = {}

    # End-to-end (the public serving API): decode + preprocess + Metal forward + L2.
    out["image_e2e"] = _bench(lambda: model.encode_image(image_bytes), warmup, iters)
    out["text_e2e"] = _bench(lambda: model.encode_text(query), warmup, iters)

    # Forward-only: factor out CPU preprocessing/tokenisation so the Metal
    # forward is isolated. Touches the mlx_clip object directly (dev tool); guard
    # so a layout change degrades to end-to-end-only rather than failing the run.
    try:
        m = model._model  # the mlx_clip instance (see MLXClip.encode_image)
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = m.img_processor([pil])
        input_ids = m.tokenizer([clean_text(query, canonicalize=False)])

        def _img_fwd():
            emb = m.model(pixel_values=pixel_values).image_embeds
            mx.eval(emb)

        def _txt_fwd():
            emb = m.model(input_ids=input_ids).text_embeds
            mx.eval(emb)

        out["image_fwd"] = _bench(_img_fwd, warmup, iters)
        out["text_fwd"] = _bench(_txt_fwd, warmup, iters)
    except Exception as e:
        print(f"  [mlx forward-only skipped: {e}]")

    model.unload()
    gc.collect()
    return out


# --------------------------------------------------------------------------- #
# Upstream ONNX backend (the standard Immich ML server's export)
# --------------------------------------------------------------------------- #
def bench_onnx(
    model_name: str,
    image_bytes: bytes,
    query: str,
    providers: list[str],
    warmup: int,
    iters: int,
) -> dict:
    """Benchmark the upstream immich-app ONNX export under each provider.

    Downloads ``visual/model.onnx`` + ``textual/model.onnx`` (cached after the
    first run). Preprocessing mirrors Immich's own (resize-shortest-224 +
    center-crop + CLIP-normalize for images; open_clip BPE tokeniser -> 77 for
    text), so the end-to-end numbers are comparable to mlx_clip's.
    """
    import onnxruntime as ort
    import open_clip
    from huggingface_hub import hf_hub_download

    from src.models.immich_preprocess import clean_text, siglip_image_pixels

    repo = f"immich-app/{model_name}"
    print(f"  [onnx] fetching {repo} (visual/textual model.onnx; cached after first run)...")
    visual_path = hf_hub_download(repo, "visual/model.onnx")
    textual_path = hf_hub_download(repo, "textual/model.onnx")

    arch, _ = _reference_arch(model_name)
    tokenizer = open_clip.get_tokenizer(arch)

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = siglip_image_pixels(pil, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD).astype(np.float32)
    tokens = tokenizer([clean_text(query, canonicalize=False)]).cpu().numpy()

    results: dict[str, dict] = {}
    for provider in providers:
        try:
            so = ort.SessionOptions()
            vsess = ort.InferenceSession(visual_path, sess_options=so, providers=[provider])
            tsess = ort.InferenceSession(textual_path, sess_options=so, providers=[provider])
        except Exception as e:
            print(f"  [onnx provider {provider} unavailable: {e}]")
            continue

        actual = vsess.get_providers()[0]  # onnxruntime falls back silently; report what ran
        results[actual] = _bench_onnx_session(vsess, tsess, image_bytes, query, pixel_values, tokens, tokenizer, warmup, iters)
        del vsess, tsess
        gc.collect()

    return results


def _bench_onnx_session(
    vsess,
    tsess,
    image_bytes: bytes,
    query: str,
    pixel_values: np.ndarray,
    tokens: np.ndarray,
    tokenizer,
    warmup: int,
    iters: int,
) -> dict:
    """Time forward-only and end-to-end image/text encodes for one ONNX session
    pair. A function (not an inline loop body) so the closures bind real
    parameters, not loop variables.
    """
    from src.models.immich_preprocess import clean_text, siglip_image_pixels

    # Match each session's expected input name + integer dtype for the text ids.
    v_in = vsess.get_inputs()[0].name
    t_in = tsess.get_inputs()[0].name
    t_dtype = np.int64 if "int64" in tsess.get_inputs()[0].type else np.int32
    tok_in = tokens.astype(t_dtype)

    out: dict[str, dict] = {}
    out["image_fwd"] = _bench(lambda: vsess.run(None, {v_in: pixel_values}), warmup, iters)
    out["text_fwd"] = _bench(lambda: tsess.run(None, {t_in: tok_in}), warmup, iters)

    # End-to-end: include decode + preprocess (image) / tokenise (text), as
    # Immich's server does, so it lines up with mlx_clip's end-to-end.
    def _img_e2e():
        p = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        px = siglip_image_pixels(p, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD).astype(np.float32)
        vsess.run(None, {v_in: px})

    def _txt_e2e():
        tk = tokenizer([clean_text(query, canonicalize=False)]).cpu().numpy().astype(t_dtype)
        tsess.run(None, {t_in: tk})

    out["image_e2e"] = _bench(_img_e2e, warmup, iters)
    out["text_e2e"] = _bench(_txt_e2e, warmup, iters)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Immich CLIP model names (mlx_clip OpenAI ports)")
    ap.add_argument("--image", type=Path, default=None, help="a single real image to encode (else download one sample)")
    ap.add_argument("--query", default=DEFAULT_QUERY, help="text query to encode")
    ap.add_argument("--warmup", type=int, default=5, help="untimed warm-up iterations")
    ap.add_argument("--iters", type=int, default=30, help="timed iterations per path")
    ap.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider", "CoreMLExecutionProvider"],
        help="onnxruntime execution providers for the upstream baseline",
    )
    ap.add_argument("--no-onnx", action="store_true", help="skip the ONNX baseline (no Hub download)")
    ap.add_argument("--report", type=Path, default=None, help="write a markdown report here")
    ap.add_argument("--cache-dir", type=Path, default=ML_ROOT / "cache" / "parity_images")
    args = ap.parse_args()

    # One sample image, shared across all models/backends.
    if args.image:
        image_bytes = args.image.read_bytes()
        img_label = str(args.image)
    else:
        images, _ = load_images(None, 1, args.cache_dir, allow_synthetic=True)
        image_bytes = images[0][1]
        img_label = f"sample:{images[0][0]}"

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("CLIP THROUGHPUT/LATENCY: mlx_clip (Metal) vs upstream ONNX")
    emit("=" * 78)
    emit(f"image={img_label}  query={args.query!r}  warmup={args.warmup}  iters={args.iters}")
    emit("Warm single-item encode (the production serving pattern). Lower ms / higher items/s is better.")

    for model_name in args.models:
        emit()
        emit("#" * 78)
        emit(f"# {model_name}")
        emit("#" * 78)

        emit("\n[backend] mlx_clip (production src.models.clip, Metal GPU) ...")
        mlx = bench_mlxclip(model_name, image_bytes, args.query, args.warmup, args.iters)

        onnx: dict = {}
        if not args.no_onnx:
            emit(f"\n[backend] upstream ONNX immich-app/{model_name} (providers={args.providers}) ...")
            onnx = bench_onnx(model_name, image_bytes, args.query, args.providers, args.warmup, args.iters)

        # Table: one row per backend/path, columns = image vs text.
        emit()
        header = f"{'backend':<34} {'IMG median ms':>14} {'IMG it/s':>10} {'TXT median ms':>14} {'TXT it/s':>10}"
        emit(header)
        emit("-" * len(header))

        def row(label: str, b: dict) -> None:
            img = b.get("image_e2e") or b.get("image_fwd")
            txt = b.get("text_e2e") or b.get("text_fwd")
            emit(f"{label:<34} {img['median_ms']:>14.2f} {img['throughput']:>10.1f} {txt['median_ms']:>14.2f} {txt['throughput']:>10.1f}")

        row("mlx_clip (end-to-end)", {"image_e2e": mlx["image_e2e"], "text_e2e": mlx["text_e2e"]})
        if "image_fwd" in mlx:
            row("mlx_clip (forward-only)", {"image_e2e": mlx["image_fwd"], "text_e2e": mlx["text_fwd"]})
        for provider, ob in onnx.items():
            row(f"onnx {provider} (end-to-end)", {"image_e2e": ob["image_e2e"], "text_e2e": ob["text_e2e"]})
            row(f"onnx {provider} (forward-only)", {"image_e2e": ob["image_fwd"], "text_e2e": ob["text_fwd"]})

        # Verdict vs the CPU provider (Immich's actual Docker baseline on this hardware).
        cpu = onnx.get("CPUExecutionProvider")
        if cpu:
            emit()
            for path, key in (("image", "image_e2e"), ("text", "text_e2e")):
                speedup = cpu[key]["median_ms"] / mlx[key]["median_ms"]
                verdict = f"mlx_clip {speedup:.2f}x FASTER than ONNX-CPU" if speedup >= 1 else f"mlx_clip {1 / speedup:.2f}x SLOWER than ONNX-CPU"
                emit(f"  {path}: {verdict} (mlx {mlx[key]['median_ms']:.2f}ms vs onnx-cpu {cpu[key]['median_ms']:.2f}ms)")

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\n[report] written to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
