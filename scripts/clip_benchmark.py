#!/usr/bin/env python3
"""CLIP throughput/latency benchmark: the production mlx_clip path vs upstream ONNX.

Parity is already settled — ``clip_parity.py`` shows mlx_clip is cosine-1.0000
faithful to the Immich server for the OpenAI CLIP ports. The open question this
harness answers is SPEED: is mlx_clip's MLX/Metal path actually "in line with
upstream performance", or materially slower than what Immich already ships? A
material slowdown would be the only reason to consider a different backend
(parity gives no reason). Immich's DEFAULT model is SigLIP2 (a separate native
backend), so the OpenAI CLIP ports benchmarked here are a SECONDARY path.

It times warm single-item (batch=1, the serving pattern) image and text encodes
for each backend and reports median/mean/p90 latency and sustained throughput,
both END-TO-END (decode + preprocess + forward + L2 — the real serving cost) and
FORWARD-ONLY (the Metal/ORT compute alone, inputs prepared once) so the gap
between them attributes how much of each path is CPU preprocessing:

* ``mlx_clip``    — THE CANDIDATE: the exact production path,
                    ``src.models.clip.MLXClip.encode_image/encode_text`` (Metal,
                    incl. the metal-lock + swap-retry scaffolding a real request
                    pays).
* ``direct_mlx``  — the SAME converted MLX weights, but driven leanly: Immich's
                    ``immich_preprocess`` (siglip_image_pixels / clean_text) feeding
                    the raw mlx_clip ``nn.Module`` directly, bypassing mlx_clip's
                    own CLIPImageProcessor/tokenizer wrappers and the production
                    lock. Isolates how much of mlx_clip's cost is wrapper overhead
                    vs. the Metal compute itself — i.e. whether a "direct MLX impl
                    reusing immich_preprocess" could be faster.
* ``onnx_cpu``    — THE UPSTREAM BASELINE Immich actually runs on a Mac: the
                    published ``immich-app/<model>`` ONNX export under
                    onnxruntime's CPUExecutionProvider (Docker Immich on Apple
                    Silicon has no CUDA, so it serves CPU ONNX).
* ``onnx_coreml`` — the same ONNX export under CoreMLExecutionProvider — the best
                    onnxruntime can do on Apple Silicon (ANE/GPU), the closest
                    GPU-vs-GPU comparison against mlx_clip's Metal path.

All backends consume identical JPEG bytes and identical query strings, take the
batch-1 single-item path, and L2-normalize the output, so the numbers are
apples-to-apples on ONE Apple-Silicon Mac. Backends load and free sequentially to
bound peak memory; mlx_clip and direct_mlx share one loaded model.

A cosine sanity-check (each backend vs mlx_clip on the first item) guards against
timing a broken/mismatched path — a wrong preprocessing layout or wrong
checkpoint collapses cosine toward 0. Note ``direct_mlx`` is ~1.0000 (same
weights), but the ONNX backends land near ~0.97: that is the upstream ONNX
export's OWN numerical drift from the open_clip/MLX reference (verified — our
siglip preprocessing is cosine-1.0000 to open_clip's own transform), not a
benchmark error. The two backends still run the identical ViT forward over the
identical input, so the LATENCY is comparable regardless.

Usage (from ml/, venv active):

    .venv/bin/python scripts/clip_benchmark.py                       # B-16 + L-14, sample images
    .venv/bin/python scripts/clip_benchmark.py --images ~/Pics       # real library photos
    .venv/bin/python scripts/clip_benchmark.py --models ViT-B-16__openai --repeats 12
    .venv/bin/python scripts/clip_benchmark.py --report bench.md
"""

from __future__ import annotations

import argparse
import gc
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make ``src`` importable and the sibling ``embedding_parity`` helpers reusable
# (scripts/ is not a package, so add both dirs to the path).
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from embedding_parity import DEFAULT_QUERIES, cosine, load_images

# OpenAI CLIP preprocessing constants — image size 224, OpenAI CLIP mean/std.
# These are what Immich's OpenClipVisualEncoder uses for the OpenAI CLIP models.
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Immich CLIP name -> the upstream HF repo holding its ONNX export, plus the
# open_clip arch whose tokenizer matches it (for the ONNX text input). Only the
# mlx_clip-backed OpenAI ports are benchmarked — LAION/SigLIP have no mlx_clip
# backend (see src/models/clip.py MODEL_MAP).
ONNX_REPO = {
    "ViT-B-16__openai": ("immich-app/ViT-B-16__openai", "ViT-B-16-quickgelu"),
    "ViT-L-14__openai": ("immich-app/ViT-L-14__openai", "ViT-L-14-quickgelu"),
}
DEFAULT_MODELS = ["ViT-B-16__openai", "ViT-L-14__openai"]


# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #
def _time_calls(fn, items: list, warmup: int, repeats: int) -> dict:
    """Time ``fn(item)`` over ``repeats`` passes of ``items`` after ``warmup`` passes.

    ``fn`` MUST force the backend's lazy work to complete before returning (e.g.
    materialize the MLX array to numpy / run the ORT session), so each timed call
    captures the full encode cost. Returns latency stats (ms) and throughput.
    """
    # Warmup: triggers lazy MLX compile, Metal allocation, and ORT graph
    # optimization so the timed passes measure the warm steady state.
    for _ in range(warmup):
        for it in items:
            fn(it)

    latencies_ms: list[float] = []
    for _ in range(repeats):
        for it in items:
            t0 = time.perf_counter()
            fn(it)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    median = float(np.median(arr))
    return {
        "n": len(arr),
        "median_ms": median,
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        # Sustained single-stream throughput from the median per-item latency.
        "throughput": 1000.0 / median if median > 0 else float("inf"),
    }


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- #
# Backends — each exposes encode_image(bytes)->vec and encode_text(str)->vec,
# both L2-normalized, and a close() to free memory.
# --------------------------------------------------------------------------- #
class MlxBackends:
    """mlx_clip (production) and direct_mlx, sharing ONE loaded MLX model."""

    def __init__(self, model_name: str):
        import mlx.core as mx

        from src.models.clip import get_clip_model
        from src.models.immich_preprocess import clean_text, siglip_image_pixels

        self._mx = mx
        self._clean_text = clean_text
        self._siglip_image_pixels = siglip_image_pixels
        self._prod = get_clip_model(model_name)
        if getattr(self._prod, "_use_mlx_embeddings", False) or getattr(self._prod, "_use_fallback", False):
            raise SystemExit(f"{model_name!r} did not load via the mlx_clip backend (got the native SigLIP2 or open_clip path). clip_benchmark.py covers the mlx_clip OpenAI CLIP ports only.")
        self._clip = self._prod._model  # the underlying mlx_clip object

    # -- mlx_clip: the exact production encode path (incl. metal-lock) ---------
    def prod_image(self, image_bytes: bytes) -> np.ndarray:
        return self._prod.encode_image(image_bytes)

    def prod_text(self, text: str) -> np.ndarray:
        return self._prod.encode_text(text)

    # -- direct_mlx: Immich preprocess -> raw mlx module, no wrapper/lock -------
    def direct_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        px = self._siglip_image_pixels(image, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD)
        # mlx_clip's conv is NHWC; siglip_image_pixels emits NCHW [1,3,H,W].
        px_nhwc = np.transpose(px, (0, 2, 3, 1))
        out = self._clip.model(pixel_values=self._mx.array(px_nhwc))
        # np.array(...) forces the lazy MLX graph to evaluate — the timed work.
        emb = np.array(out.image_embeds[0])
        return _l2(emb).astype(np.float32)

    def direct_text(self, text: str) -> np.ndarray:
        ids = self._clip.tokenizer([self._clean_text(text, canonicalize=False)])
        out = self._clip.model(input_ids=ids)
        emb = np.array(out.text_embeds[0])
        return _l2(emb).astype(np.float32)

    # -- forward-only: time JUST the Metal module forward, inputs prepared once --
    # mlx_clip and direct_mlx share this exact module forward (they differ only in
    # preprocessing), so one "mlx module" forward number covers both — the gap to
    # each one's end-to-end is that path's preprocessing/wrapper cost.
    def prep_image_fwd(self, image_bytes: bytes):
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._clip.img_processor([pil])  # NHWC mlx array

    def fwd_image(self, pixel_values) -> None:
        self._mx.eval(self._clip.model(pixel_values=pixel_values).image_embeds)

    def prep_text_fwd(self, text: str):
        return self._clip.tokenizer([self._clean_text(text, canonicalize=False)])

    def fwd_text(self, input_ids) -> None:
        self._mx.eval(self._clip.model(input_ids=input_ids).text_embeds)

    def close(self):
        self._prod.unload()
        self._clip = None
        gc.collect()


class OnnxBackend:
    """Upstream ONNX export under one onnxruntime execution provider."""

    def __init__(self, model_name: str, provider: str):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        repo, arch = ONNX_REPO[model_name]
        from src.models.immich_preprocess import siglip_image_pixels

        self._siglip_image_pixels = siglip_image_pixels

        vis = hf_hub_download(repo, "visual/model.onnx")
        txt = hf_hub_download(repo, "textual/model.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._vis = ort.InferenceSession(vis, so, providers=[provider])
        self._txt = ort.InferenceSession(txt, so, providers=[provider])
        # Confirm the requested provider actually loaded (CoreML can silently
        # fall back to CPU, which would mislabel the row).
        self.active_provider = self._vis.get_providers()[0]
        self._vis_in = self._vis.get_inputs()[0].name
        self._txt_in = self._txt.get_inputs()[0].name

        import open_clip

        self._tokenizer = open_clip.get_tokenizer(arch)

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        px = self._siglip_image_pixels(image, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD)
        emb = self._vis.run(None, {self._vis_in: px.astype(np.float32)})[0][0]
        return _l2(emb).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        ids = self._tokenizer([text]).numpy().astype(np.int32)  # [1,77]
        emb = self._txt.run(None, {self._txt_in: ids})[0][0]
        return _l2(emb).astype(np.float32)

    # -- forward-only: time JUST the ORT session run, inputs prepared once -------
    def prep_image_fwd(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._siglip_image_pixels(image, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD).astype(np.float32)

    def fwd_image(self, px: np.ndarray) -> None:
        self._vis.run(None, {self._vis_in: px})

    def prep_text_fwd(self, text: str) -> np.ndarray:
        return self._tokenizer([text]).numpy().astype(np.int32)

    def fwd_text(self, ids: np.ndarray) -> None:
        self._txt.run(None, {self._txt_in: ids})

    def close(self):
        self._vis = None
        self._txt = None
        gc.collect()


# --------------------------------------------------------------------------- #
# Per-model benchmark
# --------------------------------------------------------------------------- #
def benchmark_model(
    model_name: str,
    images: list[tuple[str, bytes]],
    queries: list[str],
    warmup: int,
    repeats: int,
    providers: list[str],
    emit,
) -> dict:
    img_bytes = [b for _, b in images]
    rows: dict[str, dict] = {}
    sanity: dict[str, float] = {}

    # --- MLX backends (one shared load) -------------------------------------
    emit(f"\n[backend] mlx_clip + direct_mlx (shared MLX load) for {model_name} ...")
    mlx = MlxBackends(model_name)
    ref_img = mlx.prod_image(img_bytes[0])
    # Pre-prepare inputs once so the forward-only pass times the Metal module
    # alone (no decode/preprocess); the gap to end-to-end is preprocessing cost.
    mlx_img_pv = [mlx.prep_image_fwd(b) for b in img_bytes]
    mlx_txt_ids = [mlx.prep_text_fwd(q) for q in queries]
    rows["mlx_clip"] = {
        "image": _time_calls(mlx.prod_image, img_bytes, warmup, repeats),
        "text": _time_calls(mlx.prod_text, queries, warmup, repeats),
        # Forward-only on the shared MLX module (covers direct_mlx too).
        "image_fwd": _time_calls(mlx.fwd_image, mlx_img_pv, warmup, repeats),
        "text_fwd": _time_calls(mlx.fwd_text, mlx_txt_ids, warmup, repeats),
    }
    # direct_mlx should match mlx_clip's embedding (same weights) — sanity it.
    sanity["direct_mlx"] = cosine(ref_img, mlx.direct_image(img_bytes[0]))
    rows["direct_mlx"] = {
        "image": _time_calls(mlx.direct_image, img_bytes, warmup, repeats),
        "text": _time_calls(mlx.direct_text, queries, warmup, repeats),
    }
    mlx.close()

    # --- ONNX backends (one session per provider) ---------------------------
    for provider in providers:
        label = "onnx_cpu" if provider == "CPUExecutionProvider" else "onnx_coreml"
        emit(f"[backend] {label} ({provider}) for {model_name} ...")
        try:
            onnx = OnnxBackend(model_name, provider)
        except Exception as e:
            emit(f"  SKIP {label}: failed to init ({e})")
            continue
        if onnx.active_provider != provider:
            emit(f"  NOTE {label}: requested {provider} but session reports {onnx.active_provider} (silent fallback)")
        # Sanity: upstream ONNX vs mlx_clip should be the same embedding (parity).
        sanity[label] = cosine(ref_img, onnx.encode_image(img_bytes[0]))
        onnx_img_px = [onnx.prep_image_fwd(b) for b in img_bytes]
        onnx_txt_ids = [onnx.prep_text_fwd(q) for q in queries]
        rows[label] = {
            "image": _time_calls(onnx.encode_image, img_bytes, warmup, repeats),
            "text": _time_calls(onnx.encode_text, queries, warmup, repeats),
            "image_fwd": _time_calls(onnx.fwd_image, onnx_img_px, warmup, repeats),
            "text_fwd": _time_calls(onnx.fwd_text, onnx_txt_ids, warmup, repeats),
        }
        onnx.close()

    return {"rows": rows, "sanity": sanity}


def _fmt_table(model_name: str, result: dict, emit) -> None:
    rows = result["rows"]
    sanity = result["sanity"]
    emit(f"\n### {model_name}")
    emit("")
    emit("End-to-end (decode/preprocess + forward + L2 — the serving cost):")
    emit("| backend | image med (ms) | image p90 | image img/s | text med (ms) | text p90 | text txt/s |")
    emit("|---|--:|--:|--:|--:|--:|--:|")
    for name, r in rows.items():
        im, tx = r["image"], r["text"]
        emit(f"| {name} | {im['median_ms']:.1f} | {im['p90_ms']:.1f} | {im['throughput']:.1f} | {tx['median_ms']:.2f} | {tx['p90_ms']:.2f} | {tx['throughput']:.1f} |")

    # Forward-only isolates the Metal/ORT compute from CPU preprocessing — the
    # gap to end-to-end is preprocessing/tokenization cost. The MLX module row is
    # shared by mlx_clip and direct_mlx (same module, different preprocessing).
    emit("")
    emit("Forward-only (compute alone, inputs prepared once) and preprocessing overhead:")
    emit("| compute path | image fwd (ms) | image e2e | image preprocess | text fwd (ms) | text e2e | text preprocess |")
    emit("|---|--:|--:|--:|--:|--:|--:|")
    for name, r in rows.items():
        if "image_fwd" not in r:
            continue
        imf, txf = r["image_fwd"], r["text_fwd"]
        im, tx = r["image"], r["text"]
        path = "mlx module (mlx_clip/direct_mlx)" if name == "mlx_clip" else name
        emit(
            f"| {path} | {imf['median_ms']:.1f} | {im['median_ms']:.1f} | {im['median_ms'] - imf['median_ms']:.1f} "
            f"| {txf['median_ms']:.2f} | {tx['median_ms']:.2f} | {tx['median_ms'] - txf['median_ms']:.2f} |"
        )

    # mlx_clip speed relative to each backend (image path) — >1 means mlx_clip
    # is FASTER (lower latency) than that backend.
    if "mlx_clip" in rows:
        mc_img = rows["mlx_clip"]["image"]["median_ms"]
        mc_txt = rows["mlx_clip"]["text"]["median_ms"]
        emit("")
        emit("mlx_clip end-to-end latency vs each backend (>1.0 = mlx_clip faster):")
        for name, r in rows.items():
            if name == "mlx_clip":
                continue
            emit(f"  vs {name}: image {r['image']['median_ms'] / mc_img:.2f}x  text {r['text']['median_ms'] / mc_txt:.2f}x")

    emit("")
    emit("sanity cosine vs mlx_clip (image, item 0; confirms same model is wired up):")
    emit("  ~1.0 = identical weights (direct_mlx); ~0.97 onnx = the upstream ONNX")
    emit("  export's own drift from the open_clip/MLX reference, not a timing issue.")
    for name, c in sanity.items():
        # A genuinely broken path (wrong layout/checkpoint) collapses toward 0;
        # ~0.97 is the known ONNX-export drift, so only flag a real mismatch.
        flag = "" if c >= 0.90 else "  <-- BROKEN: wrong preprocessing/checkpoint, timing meaningless"
        emit(f"  {name}: {c:.4f}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(ONNX_REPO), help="models to benchmark")
    ap.add_argument("--images", type=Path, default=None, help="dir of real images (else download samples)")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--num-queries", type=int, default=8, help="cap on text queries timed")
    ap.add_argument("--warmup", type=int, default=3, help="warmup passes over the item set per backend")
    ap.add_argument("--repeats", type=int, default=8, help="timed passes over the item set per backend")
    ap.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider", "CoreMLExecutionProvider"],
        help="onnxruntime execution providers for the ONNX baseline",
    )
    ap.add_argument("--report", type=Path, default=None, help="write a markdown report here")
    ap.add_argument("--cache-dir", type=Path, default=ML_ROOT / "cache" / "parity_images")
    ap.add_argument("--allow-synthetic", action="store_true", help="fall back to synthetic images if download fails")
    args = ap.parse_args()

    images, used_synthetic = load_images(args.images, args.num_images, args.cache_dir, allow_synthetic=args.allow_synthetic)
    queries = DEFAULT_QUERIES[: args.num_queries]

    import onnxruntime as ort

    avail = ort.get_available_providers()
    providers = [p for p in args.providers if p in avail]
    missing = [p for p in args.providers if p not in avail]

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("CLIP THROUGHPUT/LATENCY BENCHMARK: mlx_clip vs upstream ONNX")
    emit("=" * 78)
    emit(f"Models: {args.models}")
    emit(f"Items: {len(images)} images x {len(queries)} queries  |  warmup={args.warmup} repeats={args.repeats} (batch=1)")
    emit(f"ONNX providers: {providers}" + (f"  (UNAVAILABLE, skipped: {missing})" if missing else ""))
    if used_synthetic:
        emit("NOTE: synthetic images (download failed) — latency is still valid; preprocessing on flat images is representative enough for timing.")

    results: dict[str, dict] = {}
    for model_name in args.models:
        emit("\n" + "-" * 78)
        emit(f"MODEL: {model_name}")
        emit("-" * 78)
        results[model_name] = benchmark_model(model_name, images, queries, args.warmup, args.repeats, providers, emit)

    emit("\n" + "=" * 78)
    emit("SUMMARY")
    emit("=" * 78)
    for model_name in args.models:
        _fmt_table(model_name, results[model_name], emit=emit)

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\n[report] written to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
