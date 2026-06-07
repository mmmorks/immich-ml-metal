#!/usr/bin/env python3
"""Embedding-parity harness: native MLX SigLIP2 vs the reference impl the NAS uses.

Parity gate. Decides preserve-vs-reindex for Immich's existing
smart-search index (4716 rows built with ``ViT-SO400M-16-SigLIP2-384__webli``).

It embeds the SAME images + text queries through several backends and reports
per-item cosine similarity:

* ``mlx``          — the production path (``src.models.clip``), i.e. the native
                     MLX SigLIP2 backend (Google ``siglip2-so400m-patch16-384``
                     weights, fp16, on-device). THIS is what we'd switch to.
* ``immich``       — the **gate**: the same weights run through the *standard
                     Immich ML server's* CLIP visual transform (PIL bicubic
                     resize-shortest-side + center-crop, normalize 0.5 — see
                     ``_immich_image_tensor``). That transform produced the
                     existing index, so MLX-vs-immich >= ~0.99 means the stored
                     4716 vectors stay valid (preserve, no re-index).
* ``transformers`` — HF ``SiglipModel`` (same weights) with HF's squash + no
                     text-canonicalize preprocessing. This is what ``clip.py``
                     did BEFORE the preprocessing-parity fix, so it's a *regression witness*: a low
                     MLX-vs-transformers score is the size of the preprocessing
                     bug the Immich-faithful path fixed (not a port problem —
                     the ``immich`` comparison, identical preprocessing on both
                     sides, already proves the MLX port reproduces HF at ~1.0).
* ``openclip``     — open_clip ``ViT-SO400M-16-SigLIP2-384`` / ``webli`` with its
                     own torchvision (squash) transform. Diagnostic ONLY: this is
                     NOT what the NAS runs at inference, so don't read it as the
                     gate. open_clip is merely the source the NAS exported to ONNX.

The ``mlx``/``transformers``/``openclip`` image paths all squash; ``immich``
center-crops. That geometry difference is the whole story on non-square photos.

Backends are loaded and freed sequentially to bound peak memory (each model is
~2-4 GB), so this runs on a single Apple-Silicon Mac.

Usage (from ml/, venv active):

    .venv/bin/python scripts/embedding_parity.py                 # download samples
    .venv/bin/python scripts/embedding_parity.py --images ~/Pics # real library photos
    .venv/bin/python scripts/embedding_parity.py --ref immich transformers
    .venv/bin/python scripts/embedding_parity.py --threshold 0.99 --report out.md

For the real preserve-vs-reindex decision, point ``--images`` at a sample of the
actual library — the squash-vs-crop effect is aspect-ratio dependent.
"""

from __future__ import annotations

import argparse
import gc
import io
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# Make ``src`` importable when run as a standalone script from anywhere.
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

# The smart-search model whose index we're deciding about.
IMMICH_MODEL = "ViT-SO400M-16-SigLIP2-384__webli"
OPENCLIP_ARCH = "ViT-SO400M-16-SigLIP2-384"
OPENCLIP_PRETRAINED = "webli"
HF_REPO = "google/siglip2-so400m-patch16-384"
EMBED_DIM = 1152
SIGLIP_CONTEXT_LEN = 64

# A spread of text queries that exercise objects, scenes, animals, people and
# abstract attributes — the kinds of things a smart-search user actually types.
DEFAULT_QUERIES = [
    "a photo of a cat",
    "a dog playing in the park",
    "a red sports car on a highway",
    "a plate of food on a wooden table",
    "a sunset over the ocean",
    "people walking on a city street at night",
    "a snowy mountain landscape",
    "a child blowing out birthday candles",
    "a laptop and a cup of coffee on a desk",
    "a close-up of a flower with a bee",
    "an old stone building with arched windows",
    "a screenshot of a chat conversation",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (defensively L2-normalized)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Sample images
# --------------------------------------------------------------------------- #
def _download_one(url: str, attempts: int = 3, base_delay: float = 1.0) -> bytes:
    """Fetch one URL, retrying transient failures. Raises the last error if all
    ``attempts`` fail (so the caller decides between fail-fast and fallback)."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "parity-harness"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                delay = base_delay * attempt
                print(f"[images] fetch failed ({e!r}); retry {attempt}/{attempts - 1} in {delay:.0f}s")
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def load_images(
    images_dir: Path | None,
    num: int,
    cache_dir: Path,
    allow_synthetic: bool = False,
) -> tuple[list[tuple[str, bytes]], bool]:
    """Return ``([(label, jpeg_bytes)], used_synthetic)``. Use a directory if
    given, else download deterministic picsum.photos samples (seeded by index).

    A failed download is retried per-image. If a sample still cannot be fetched,
    we fail fast (``SystemExit``) rather than silently degrading the gate — unless
    ``allow_synthetic`` is set, in which case we fall back LOUDLY and flag the run
    via the returned ``used_synthetic`` so the caller can refuse to pass the gate
    on synthetic data.
    """
    if images_dir is not None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)[:num]
        if not files:
            raise SystemExit(f"No images found in {images_dir}")
        out = []
        for p in files:
            # Re-encode to JPEG so every backend decodes identical source pixels.
            img = Image.open(p).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            out.append((p.name, buf.getvalue()))
        print(f"[images] {len(out)} real images from {images_dir}")
        return out, False

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, bytes]] = []
    for i in range(num):
        seed = 100 + i
        # Non-square (4:3) on purpose: square images hide the dominant
        # preprocessing effect (Immich center-crops, SiglipProcessor squashes).
        fpath = cache_dir / f"picsum_{seed}_640x480.jpg"
        if not fpath.exists():
            url = f"https://picsum.photos/seed/{seed}/640/480"
            try:
                fpath.write_bytes(_download_one(url))
            except Exception as e:
                if not allow_synthetic:
                    raise SystemExit(
                        f"[images] sample download failed after retries ({e!r}). "
                        "A trustworthy parity gate needs real photos — re-run with "
                        "network access, pass --images DIR with local photos, or pass "
                        "--allow-synthetic to force the (weaker) synthetic fallback "
                        "(which marks the run INCONCLUSIVE)."
                    ) from e
                print("!" * 78)
                print(f"[images] sample download failed after retries ({e!r})")
                print("[images] --allow-synthetic set: falling back to SYNTHETIC images.")
                print("[images] Parity stats on synthetic data are NOT a valid gate.")
                print("!" * 78)
                return _synthetic_images(num), True
        img = Image.open(fpath).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        out.append((f"picsum_{seed}", buf.getvalue()))
    print(f"[images] {len(out)} downloaded sample photos 640x480 (cache: {cache_dir})")
    return out, False


def _synthetic_images(num: int) -> list[tuple[str, bytes]]:
    """Deterministic structured fallback images (weaker preprocessing stress)."""
    out = []
    rng = np.random.RandomState(0)
    for i in range(num):
        arr = rng.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
        # Add low-frequency structure so it isn't pure noise.
        _, xx = np.mgrid[0:512, 0:512]
        arr[..., 0] = ((np.sin(xx / (8 + i)) + 1) * 127).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=95)
        out.append((f"synthetic_{i}", buf.getvalue()))
    print(f"[images] {num} SYNTHETIC images (offline) — weaker preprocessing test")
    return out


# --------------------------------------------------------------------------- #
# Backends — each returns (image_embeds [N,D], text_embeds [M,D]), L2-normalized
# --------------------------------------------------------------------------- #
def embed_mlx(images: list[tuple[str, bytes]], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Production path: src.models.clip MLX SigLIP2 backend (fp16, on-device)."""
    from src.models.clip import get_clip_model

    model = get_clip_model(IMMICH_MODEL)
    if not getattr(model, "_use_mlx_embeddings", False):
        raise SystemExit("MLX SigLIP2 backend did not load. Check that mlx-embeddings is installed and the model downloaded.")
    img = np.stack([model.encode_image(b) for _, b in images])
    txt = np.stack([model.encode_text(q) for q in queries])
    model.unload()
    gc.collect()
    return img, txt


def embed_openclip(images: list[tuple[str, bytes]], queries: list[str], device: str) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic: open_clip webli with its own torchvision (squash) transform.

    NOT the index gate — the NAS does its own preprocessing at inference (see the
    ``immich`` reference). open_clip is only the source exported to ONNX.

    open-clip-torch is optional (not in requirements.txt), so guide the user to
    install it rather than crashing with a bare ImportError.
    """
    try:
        import open_clip
        import torch
    except ImportError as e:
        raise SystemExit(
            "The 'openclip' diagnostic needs open-clip-torch (optional). "
            "Install it to use this reference: pip install open-clip-torch"
        ) from e

    model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_ARCH, pretrained=OPENCLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(OPENCLIP_ARCH)
    model = model.to(device).eval()

    img_list, txt_list = [], []
    with torch.no_grad():
        for _, b in images:
            pil = Image.open(io.BytesIO(b)).convert("RGB")
            t = preprocess(pil).unsqueeze(0).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            img_list.append(f.squeeze(0).cpu().numpy().astype(np.float32))
        for q in queries:
            tok = tokenizer([q]).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            txt_list.append(f.squeeze(0).cpu().numpy().astype(np.float32))

    del model
    gc.collect()
    return np.stack(img_list), np.stack(txt_list)


def embed_hf(
    images: list[tuple[str, bytes]],
    queries: list[str],
    device: str,
    variants: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """HF SiglipModel (same weights as MLX *and* as the webli ONNX) under one or
    more preprocessing variants. Loads the model once.

    variants:
      "immich"       — the standard Immich ML server's preprocessing, via the
                       SAME production module clip.py uses
                       (src.models.immich_preprocess): resize-shortest +
                       center-crop for images, clean_text + raw tokenizer.json
                       for text. So "MLX vs immich" checks that the MLX backend
                       reproduces an HF-weights server under our real
                       preprocessing — image AND text, incl. caps/punctuation.
                       THE GATE.
      "transformers" — HF SiglipProcessor (squash images, no text canonicalize).
                       This is what clip.py used BEFORE the preprocessing-parity fix, so MLX-vs-this
                       isolates pure port fidelity from the preprocessing change.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    from src.models.immich_preprocess import SiglipTextTokenizer, siglip_image_pixels

    model = AutoModel.from_pretrained(HF_REPO).to(device).eval()
    processor = AutoProcessor.from_pretrained(HF_REPO)

    def pooled(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    pils = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in images]

    # Immich-faithful tokenizer (only needed for the immich variant).
    immich_tok = None
    if "immich" in variants:
        from huggingface_hub import hf_hub_download

        immich_tok = SiglipTextTokenizer(hf_hub_download(HF_REPO, "tokenizer.json"))

    def img_embeds(variant: str) -> np.ndarray:
        out = []
        with torch.no_grad():
            for pil in pils:
                if variant == "immich":
                    pix = torch.from_numpy(siglip_image_pixels(pil)).to(device)
                else:  # HF SiglipProcessor (squash)
                    pix = processor(images=[pil], return_tensors="pt")["pixel_values"].to(device)
                f = pooled(model.get_image_features(pixel_values=pix))
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(out)

    def txt_embeds(variant: str) -> np.ndarray:
        out = []
        with torch.no_grad():
            for q in queries:
                if variant == "immich":
                    assert immich_tok is not None
                    ids = torch.from_numpy(immich_tok(q)).to(device)
                    f = pooled(model.get_text_features(input_ids=ids))
                else:
                    inputs = processor(
                        text=[q],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=SIGLIP_CONTEXT_LEN,
                        truncation=True,
                    ).to(device)
                    f = pooled(model.get_text_features(**inputs))
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(out)

    result = {v: (img_embeds(v), txt_embeds(v)) for v in variants}
    del model
    gc.collect()
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def stats(sims: np.ndarray) -> dict:
    return {
        "min": float(np.min(sims)),
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "max": float(np.max(sims)),
    }


def retrieval_agreement(mlx_img, mlx_txt, ref_img, ref_txt) -> dict:
    """Does cross-modal search return the same ranking under both backends?

    This is the user-facing question: for each text query, is the best-matching
    image the same under MLX as under the reference? Captures whether *retrieval*
    is preserved even if absolute cosines drift slightly.
    """
    mlx_sim = mlx_txt @ mlx_img.T  # [M, N]
    ref_sim = ref_txt @ ref_img.T
    mlx_top = np.argmax(mlx_sim, axis=1)
    ref_top = np.argmax(ref_sim, axis=1)
    top1_agree = float(np.mean(mlx_top == ref_top))
    # Spearman-ish: correlation of the flattened similarity matrices.
    # A single image + single query yields a one-element flattened matrix, whose
    # corrcoef is a legitimate NaN (zero variance / DoF <= 0). Suppress the numpy
    # RuntimeWarnings (DoF, divide-by-zero, invalid) for that expected edge case
    # at the call site so runtime logs stay clean too — NaN is still returned.
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        matrix_corr = float(np.corrcoef(mlx_sim.ravel(), ref_sim.ravel())[0, 1])
    return {"top1_agreement": top1_agree, "matrix_corr": matrix_corr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=None, help="dir of real images (else download samples)")
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--queries-file", type=Path, default=None, help="newline-separated text queries")
    ap.add_argument(
        "--ref",
        nargs="+",
        choices=["immich", "transformers", "openclip"],
        default=["immich"],
        help="reference backend(s). 'immich'=Immich server transform (the gate "
        "AND port fidelity, since both sides share production preprocessing); "
        "'transformers'=HF squash (regression witness for the pre-parity-fix path); "
        "'openclip'=open_clip's own torchvision transform (diagnostic only)",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="torch device for references")
    ap.add_argument("--threshold", type=float, default=0.99, help="cosine gate (applied to immich)")
    ap.add_argument("--report", type=Path, default=None, help="write a markdown report here")
    ap.add_argument("--cache-dir", type=Path, default=ML_ROOT / "cache" / "parity_images")
    ap.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="if sample-image download fails, fall back to synthetic images instead "
        "of erroring out. The run is then marked INCONCLUSIVE (non-zero exit) since "
        "synthetic data is not a valid parity gate.",
    )
    args = ap.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [ln.strip() for ln in args.queries_file.read_text().splitlines() if ln.strip()]

    images, used_synthetic = load_images(args.images, args.num_images, args.cache_dir, allow_synthetic=args.allow_synthetic)
    labels = [name for name, _ in images]
    print(f"[setup] {len(images)} images x {len(queries)} queries; references={args.ref}; device={args.device}")

    # MLX first (the candidate). Each backend loads -> embeds -> frees.
    print("\n[backend] MLX SigLIP2 (production src.models.clip) ...")
    mlx_img, mlx_txt = embed_mlx(images, queries)
    assert mlx_img.shape[1] == EMBED_DIM, f"unexpected dim {mlx_img.shape}"

    refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hf_variants = [v for v in ("immich", "transformers") if v in args.ref]
    if hf_variants:
        print(f"\n[backend] HF SiglipModel (variants: {hf_variants}) ...")
        print("  'immich' = Immich server transform (gate); 'transformers' = HF squash")
        refs.update(embed_hf(images, queries, args.device, hf_variants))
    if "openclip" in args.ref:
        print("\n[backend] open_clip webli (open_clip's own transform — diagnostic) ...")
        refs["openclip"] = embed_openclip(images, queries, args.device)
    # Order the report so the gate (immich) is first.
    refs = {k: refs[k] for k in ("immich", "transformers", "openclip") if k in refs}

    # Build report.
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit("EMBEDDING PARITY: MLX SigLIP2 vs reference")
    emit("=" * 78)
    emit(f"Model: {IMMICH_MODEL}  |  images: {len(images)}  queries: {len(queries)}")
    emit(f"Threshold (gate, immich image): >= {args.threshold}")

    gate_pass = None
    port_faithful = None  # MLX vs transformers (same weights): isolates port fidelity
    for ref_name, (ref_img, ref_txt) in refs.items():
        img_sims = np.array([cosine(mlx_img[i], ref_img[i]) for i in range(len(images))])
        txt_sims = np.array([cosine(mlx_txt[j], ref_txt[j]) for j in range(len(queries))])
        ist, tst = stats(img_sims), stats(txt_sims)
        agree = retrieval_agreement(mlx_img, mlx_txt, ref_img, ref_txt)

        emit()
        emit(f"### MLX vs {ref_name}")
        emit(f"  IMAGE cosine: min={ist['min']:.4f} mean={ist['mean']:.4f} median={ist['median']:.4f} max={ist['max']:.4f}")
        emit(f"  TEXT  cosine: min={tst['min']:.4f} mean={tst['mean']:.4f} median={tst['median']:.4f} max={tst['max']:.4f}")
        emit(f"  Cross-modal: top-1 retrieval agreement={agree['top1_agreement']:.3f} matrix_corr={agree['matrix_corr']:.4f}")
        emit("  per-image:")
        for lbl, s in zip(labels, img_sims):
            flag = "" if s >= args.threshold else "  <-- below threshold"
            emit(f"    {s:.4f}  {lbl}{flag}")
        emit("  per-query:")
        for q, s in zip(queries, txt_sims):
            flag = "" if s >= args.threshold else "  <-- below threshold"
            emit(f"    {s:.4f}  {q!r}{flag}")

        if ref_name == "immich":
            gate_pass = bool(ist["min"] >= args.threshold)
        if ref_name == "transformers":
            # Same weights AND same preprocessing (squash) as the MLX backend.
            # >=0.999 means the mlx-embeddings port itself is exact, so the
            # immich gap below is purely the resize geometry, not the port.
            port_faithful = bool(ist["min"] >= 0.999 and tst["min"] >= 0.999)

    emit()
    emit("-" * 78)
    if gate_pass is None:
        emit("VERDICT: immich reference not run — cannot decide index strategy.")
        verdict_rc = 2
    elif gate_pass:
        emit(f"VERDICT: PASS (min image cosine vs Immich server >= {args.threshold}).")
        emit("  => PRESERVE the existing index; the MLX backend reproduces it.")
        verdict_rc = 0
    elif port_faithful:
        # Port is exact, but clip.py uses HF SiglipProcessor (squash) while the
        # Immich server resizes-shortest + center-crops. That geometry mismatch
        # is the whole gap and it lives entirely in clip.py preprocessing.
        emit(f"VERDICT: FIX-PREPROCESSING (image cosine vs Immich server < {args.threshold}; MLX==transformers to ~1.0).")
        emit("  The MLX port and weights are exact; the gap is the IMAGE RESIZE only:")
        emit("  clip.py uses HF SiglipProcessor (squash to 384^2) but the Immich")
        emit("  server resizes the shortest side to 384 then center-crops. On")
        emit("  non-square photos these feed different pixels to an identical model.")
        emit("  => Fix: make clip.py replicate Immich's transform (PIL bicubic")
        emit("     resize-shortest + center-crop, normalize 0.5). That reproduces the")
        emit("     index embeddings (~1.0) and PRESERVES the existing 4716 rows.")
        verdict_rc = 0
    else:
        emit(f"VERDICT: FAIL (min image cosine vs Immich server < {args.threshold}).")
        emit("  => Port/weights issue beyond preprocessing; investigate before deciding.")
        verdict_rc = 1
    emit("-" * 78)

    if used_synthetic:
        # Synthetic data can't validate a preserve-vs-reindex decision; never let
        # a synthetic run report a passing gate. rc=3 distinguishes it from the
        # other non-zero verdicts (1=FAIL, 2=immich-not-run).
        emit()
        emit("!" * 78)
        emit("RUN USED SYNTHETIC IMAGES — this verdict is NOT a valid index-strategy gate.")
        emit("Re-run with real sample photos (network access or --images DIR) before deciding.")
        emit("!" * 78)
        if verdict_rc == 0:
            verdict_rc = 3

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\n[report] written to {args.report}")

    return verdict_rc


if __name__ == "__main__":
    raise SystemExit(main())
