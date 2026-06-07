#!/usr/bin/env python3
"""CLIP embedding-parity harness: the production mlx_clip path vs the Immich server.

Parity gate. The OpenAI/LAION CLIP models (e.g. ``ViT-B-32__openai``) run
through mlx_clip's *own* image processor and BPE tokenizer — NOT the
Immich-faithful ``immich_preprocess`` path that SigLIP2 uses (which
``embedding_parity.py`` covers). So whether their embeddings are interchangeable
with the standard Immich ML server's — i.e. whether an existing smart-search
index built with the model stays valid — was plausible but UNVERIFIED. This
harness measures it.

It embeds the SAME images + text queries through each backend and reports
per-item cosine similarity:

* ``mlxclip``  — the production path (``src.models.clip`` -> mlx_clip backend).
                 THE CANDIDATE. Images: mlx_clip ``CLIPImageProcessor``
                 (resize-shortest-224 + center-crop + CLIP-normalize). Text:
                 ``clean_text(canonicalize=False)`` then mlx_clip's CLIP BPE
                 tokenizer.
* ``immich``   — THE GATE: the SAME model's open_clip checkpoint run through
                 Immich's EXACT transform — ``siglip_image_pixels`` with CLIP
                 constants (resize-shortest-224 + center-crop + CLIP-normalize)
                 for images, ``clean_text(canonicalize=False)`` + the CLIP
                 tokenizer for text. The ``arch``/``pretrained`` pair comes from
                 ``resolve_fallback_arch`` — the same open_clip mapping the
                 production fallback uses, and the checkpoint Immich exports to
                 ONNX. So ``mlxclip`` vs ``immich`` >= ~0.99 means the stored
                 vectors stay valid (preserve, no re-index).
* ``openclip`` — diagnostic: open_clip with its OWN torchvision transform +
                 tokenizer. Isolates any gap that is purely preprocessing rather
                 than weights/activation.

Caveat for non-default models: only ``ViT-B-32__openai`` is verified.
``MLXClip._load_model`` calls ``mlx_clip(repo_id)`` WITHOUT ``hf_repo``, so a
model whose local dir is absent converts the DEFAULT
``openai/clip-vit-base-patch32`` — i.e. ``ViT-B-16__openai`` / ``ViT-L-14__openai``
/ the LAION mappings all silently load OpenAI B-32 weights (empirically ~0 cosine
vs the right reference). mlx_clip also hardcodes ``quick_gelu`` (``model.py``),
which is wrong for LAION's standard-GELU ViT-B-32. So those models FAIL this gate;
run them with ``--model`` to see it. The fix: pass the correct
``hf_repo`` so mlx_clip loads the intended OpenAI B-16/L-14 weights, and mark
LAION unsupported (``MODEL_MAP -> None``, which raises) — mlx_clip can't reproduce
its standard-GELU activation and the open_clip fallback was removed.

Backends are loaded and freed sequentially to bound peak memory, so this runs on
a single Apple-Silicon Mac.

Usage (from ml/, venv active):

    .venv/bin/python scripts/clip_parity.py                      # ViT-B-32__openai, download samples
    .venv/bin/python scripts/clip_parity.py --images ~/Pics      # real library photos
    .venv/bin/python scripts/clip_parity.py --model ViT-B-32__laion2b-s34b-b79k  # the GELU-mismatch case
    .venv/bin/python scripts/clip_parity.py --threshold 0.99 --report out.md

For the real preserve-vs-reindex decision, point ``--images`` at a sample of the
actual library — preprocessing effects are aspect-ratio dependent.
"""

from __future__ import annotations

import argparse
import gc
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make ``src`` importable, and the sibling ``embedding_parity`` helpers reusable
# (scripts/ is not a package, so add both dirs to the path).
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the model-agnostic harness helpers from the SigLIP2 parity script rather
# than re-implementing them (they're unit-tested in test_embedding_parity.py).
from embedding_parity import (
    DEFAULT_QUERIES,
    cosine,
    load_images,
    retrieval_agreement,
    stats,
)

# Standard CLIP ViT-B/32 (OpenAI) preprocessing constants: image size 224, OpenAI
# CLIP mean/std. These are what Immich's OpenClipVisualEncoder uses for the
# OpenAI/LAION CLIP models (cf. SigLIP2's 384 / 0.5 in immich_preprocess).
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_MODEL = "ViT-B-32__openai"

# open_clip (arch, pretrained) reference for each Immich CLIP name — the
# checkpoint the standard Immich server exports to ONNX. Mirrors the production
# resolve_fallback_arch that was removed, so the gate keeps a self-contained
# reference (its open_clip use is dev-only, not a production dependency).
_OPENCLIP_REF = {
    # OpenAI checkpoints REQUIRE the quickgelu variant — OpenAI CLIP trained with
    # quick_gelu, and mlx_clip hardcodes it too. Loading the plain (standard-gelu)
    # arch builds a wrong-activation reference: open_clip even warns "QuickGELU
    # mismatch", and the gate then false-FAILs a correct mlx_clip at ~0.985 (the
    # quickgelu-vs-gelu gap) instead of ~1.0. All three OpenAI ports
    # must carry -quickgelu, not just B-32.
    "ViT-B-32__openai": ("ViT-B-32-quickgelu", "openai"),
    "ViT-B-16__openai": ("ViT-B-16-quickgelu", "openai"),
    "ViT-L-14__openai": ("ViT-L-14-quickgelu", "openai"),
    "ViT-B-32__laion2b-s34b-b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-32__laion2b_s34b_b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
}


def _reference_arch(model_name: str) -> tuple[str, str]:
    """open_clip ``(arch, pretrained)`` reference for ``model_name``.

    Curated map first, then an ``arch__pretrained`` split with the OpenAI
    quickgelu suffix rule (OpenAI weights need the quickgelu variant) — the same
    resolution the removed production fallback used.
    """
    if model_name in _OPENCLIP_REF:
        return _OPENCLIP_REF[model_name]
    if "__" in model_name:
        arch, pretrained = model_name.split("__", 1)
        if pretrained == "openai" and "quickgelu" not in arch.lower() and "siglip" not in arch.lower():
            arch += "-quickgelu"
        return arch, pretrained
    raise SystemExit(f"No open_clip reference known for {model_name!r}; add it to _OPENCLIP_REF.")


# --------------------------------------------------------------------------- #
# Backends — each returns (image_embeds [N,D], text_embeds [M,D]), L2-normalized
# --------------------------------------------------------------------------- #
def embed_mlxclip(
    model_name: str,
    images: list[tuple[str, bytes]],
    queries: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Production path: src.models.clip MLXClip via the mlx_clip backend."""
    from src.models.clip import get_clip_model

    model = get_clip_model(model_name)
    if getattr(model, "_use_mlx_embeddings", False) or getattr(model, "_use_fallback", False):
        raise SystemExit(
            f"{model_name!r} did not load via the mlx_clip backend (got the native "
            "SigLIP2 or open_clip path). clip_parity.py covers the mlx_clip "
            "OpenAI/LAION CLIP path only; use embedding_parity.py for SigLIP2."
        )
    img = np.stack([model.encode_image(b) for _, b in images])
    txt = np.stack([model.encode_text(q) for q in queries])
    model.unload()
    gc.collect()
    return img, txt


def embed_openclip_refs(
    arch: str,
    pretrained: str,
    images: list[tuple[str, bytes]],
    queries: list[str],
    device: str,
    variants: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """open_clip ``arch``/``pretrained`` (the checkpoint Immich exports to ONNX)
    under one or both preprocessing variants. Loads the model once.

    variants:
      "immich"   — Immich's EXACT transform: siglip_image_pixels with CLIP
                   constants (resize-shortest-224 + center-crop + CLIP-normalize)
                   for images; clean_text(canonicalize=False) + the CLIP
                   tokenizer for text. So "mlxclip vs immich" checks the mlx_clip
                   backend reproduces the Immich server — image AND text. THE GATE.
      "openclip" — open_clip's OWN torchvision transform + raw tokenizer.
                   Diagnostic: a low score here but a high "immich" score would
                   mean the gap is preprocessing, not weights.
    """
    import open_clip
    import torch

    from src.models.immich_preprocess import clean_text, siglip_image_pixels

    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device).eval()
    pils = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in images]

    def _embed(img_tensor_fn, text_fn) -> tuple[np.ndarray, np.ndarray]:
        imgs, txts = [], []
        with torch.no_grad():
            for pil in pils:
                f = model.encode_image(img_tensor_fn(pil))
                f = f / f.norm(dim=-1, keepdim=True)
                imgs.append(f.squeeze(0).cpu().numpy().astype(np.float32))
            for q in queries:
                tok = tokenizer([text_fn(q)]).to(device)
                f = model.encode_text(tok)
                f = f / f.norm(dim=-1, keepdim=True)
                txts.append(f.squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(imgs), np.stack(txts)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "immich" in variants:
        out["immich"] = _embed(
            lambda pil: torch.from_numpy(siglip_image_pixels(pil, size=CLIP_IMAGE_SIZE, mean=CLIP_MEAN, std=CLIP_STD)).to(device),
            lambda q: clean_text(q, canonicalize=False),
        )
    if "openclip" in variants:
        out["openclip"] = _embed(
            lambda pil: preprocess(pil).unsqueeze(0).to(device),
            lambda q: q,
        )

    del model
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Immich CLIP model name routed through mlx_clip")
    ap.add_argument("--images", type=Path, default=None, help="dir of real images (else download samples)")
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--queries-file", type=Path, default=None, help="newline-separated text queries")
    ap.add_argument(
        "--ref",
        nargs="+",
        choices=["immich", "openclip"],
        default=["immich", "openclip"],
        help="reference variant(s). 'immich'=open_clip weights through Immich's transform (THE GATE); 'openclip'=open_clip's own transform (diagnostic)",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="torch device for the reference")
    ap.add_argument("--threshold", type=float, default=0.99, help="cosine gate (applied to immich image+text)")
    ap.add_argument("--report", type=Path, default=None, help="write a markdown report here")
    ap.add_argument("--cache-dir", type=Path, default=ML_ROOT / "cache" / "parity_images")
    ap.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="if sample-image download fails, fall back to synthetic images (marks the run INCONCLUSIVE)",
    )
    args = ap.parse_args()

    arch, pretrained = _reference_arch(args.model)

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [ln.strip() for ln in args.queries_file.read_text().splitlines() if ln.strip()]

    images, used_synthetic = load_images(args.images, args.num_images, args.cache_dir, allow_synthetic=args.allow_synthetic)
    labels = [name for name, _ in images]
    print(f"[setup] model={args.model} (mlx_clip) vs open_clip {arch}/{pretrained}; {len(images)} images x {len(queries)} queries; refs={args.ref}; device={args.device}")

    # mlx_clip first (the candidate). Each backend loads -> embeds -> frees.
    print("\n[backend] mlx_clip (production src.models.clip) ...")
    mlx_img, mlx_txt = embed_mlxclip(args.model, images, queries)
    dim = mlx_img.shape[1]

    print(f"\n[backend] open_clip {arch}/{pretrained} (variants: {args.ref}) ...")
    print("  'immich' = open_clip weights through Immich's transform (gate); 'openclip' = open_clip's own transform")
    refs = embed_openclip_refs(arch, pretrained, images, queries, args.device, args.ref)
    # Order the report so the gate (immich) is first.
    refs = {k: refs[k] for k in ("immich", "openclip") if k in refs}

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit("CLIP EMBEDDING PARITY: mlx_clip vs reference")
    emit("=" * 78)
    emit(f"Model: {args.model}  ->  open_clip {arch}/{pretrained}  |  dim={dim}  images={len(images)}  queries={len(queries)}")
    emit(f"Threshold (gate, immich image+text): >= {args.threshold}")

    gate_pass = None
    for ref_name, (ref_img, ref_txt) in refs.items():
        img_sims = np.array([cosine(mlx_img[i], ref_img[i]) for i in range(len(images))])
        txt_sims = np.array([cosine(mlx_txt[j], ref_txt[j]) for j in range(len(queries))])
        ist, tst = stats(img_sims), stats(txt_sims)
        agree = retrieval_agreement(mlx_img, mlx_txt, ref_img, ref_txt)

        emit()
        emit(f"### mlx_clip vs {ref_name}")
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
            gate_pass = bool(ist["min"] >= args.threshold and tst["min"] >= args.threshold)

    emit()
    emit("-" * 78)
    if gate_pass is None:
        emit("VERDICT: immich reference not run — cannot decide index strategy.")
        verdict_rc = 2
    elif gate_pass:
        emit(f"VERDICT: PASS (min image & text cosine vs Immich server >= {args.threshold}).")
        emit("  => mlx_clip embeddings are interchangeable with the Immich server for this")
        emit("     model; PRESERVE the existing index (no re-index).")
        verdict_rc = 0
    else:
        emit(f"VERDICT: FAIL (min image or text cosine vs Immich server < {args.threshold}).")
        emit("  => mlx_clip embeddings are NOT interchangeable with the Immich server for")
        emit("     this model — do NOT use it as a drop-in (indexing through it would")
        emit("     poison the smart-search index). A near-ZERO cosine indicates different")
        emit("     WEIGHTS, not just preprocessing: MLXClip._load_model calls mlx_clip()")
        emit("     without hf_repo, so non-default models convert the default")
        emit("     openai/clip-vit-base-patch32 (and mlx_clip hardcodes quick_gelu, wrong")
        emit("     for LAION). Fix: pass the correct hf_repo so mlx_clip loads")
        emit("     the intended weights (OpenAI B-16/L-14), or mark LAION unsupported")
        emit("     (MODEL_MAP -> None; raises). See the README CLIP mapping notes.")
        verdict_rc = 1
    emit("-" * 78)

    if used_synthetic:
        # Synthetic data can't validate a preserve-vs-reindex decision; never let a
        # synthetic run report a passing gate. rc=3 distinguishes it.
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
