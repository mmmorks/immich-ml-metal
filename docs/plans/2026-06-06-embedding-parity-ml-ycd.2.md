# Embedding-parity harness — MLX SigLIP2 vs the Immich server (ml-ycd.2)

**Date:** 2026-06-06
**Harness:** `ml/scripts/embedding_parity.py`
**Raw run output:** `ml/docs/plans/2026-06-06-embedding-parity-raw.txt`
**Gates:** the preserve-vs-reindex decision for Immich's existing smart-search
index (4716 rows, model `ViT-SO400M-16-SigLIP2-384__webli`) — blocks ml-ycd.9
(execute index strategy) and ml-ycd.6 (quantization eval). Hard input to ml-ycd.4
(preprocessing parity).

## UPDATE 2026-06-06 — RESOLVED by ml-ycd.4

The preprocessing fix is implemented. `clip.py` now preprocesses SigLIP2 images
and text exactly like the Immich server via the shared `src/models/immich_preprocess.py`
(resize-shortest + center-crop; `clean_text` canonicalize + raw `tokenizer.json`).
Re-running the harness (`--ref immich`, the production preprocessing on both sides):

| Comparison (640×480 photos) | IMAGE cosine (min / mean) | TEXT cosine (min / median) |
|---|---|---|
| **MLX vs Immich server** | **0.9999 / 1.0000** | **1.0000 / 1.0000** |

Verdict flips to **PASS / PRESERVE** — the MLX backend now produces embeddings
interchangeable with the standard Immich server. The `transformers` (HF squash)
reference now reads ~0.83, which is just the *regression witness* showing the size
of the bug that was fixed. Remaining: ml-410 confirms against real stored vectors.
The analysis below documents the original finding that drove the fix.

---

## TL;DR — verdict (original ml-ycd.2 finding): **FIX-PREPROCESSING (then PRESERVE)**

The MLX SigLIP2 **model is bit-for-bit faithful** to the reference (MLX vs HF
`transformers` = **1.0000** on every item). But the MLX backend as merged in
ml-ycd.3 does **not** reproduce the embeddings already in the index, because it
preprocesses images differently from the standard Immich ML server:

| Comparison (12 photos × 12 queries, 640×480) | IMAGE cosine (min / mean / median) | TEXT cosine (median) |
|---|---|---|
| **MLX vs Immich server** (PIL bicubic resize-shortest + center-crop) — **the gate** | **0.795 / 0.831 / 0.835** | 1.000 |
| MLX vs HF `transformers` (squash) — port fidelity | 0.9999 / 1.0000 / 1.0000 | 1.000 |
| MLX vs open_clip `webli` (its own torchvision squash) — diagnostic | 0.966 / 0.984 / 0.985 | 1.000 |

So weights and the port are exact; **the entire gap is the image resize geometry**,
which lives entirely in `clip.py` and is fixable without touching weights or
re-embedding the 4716 rows.

## Root cause: squash vs resize-shortest+center-crop

The standard Immich ML server's CLIP visual transform
(`immich-app/immich` → `machine-learning/immich_ml/models/transforms.py` +
`models/clip/visual.py`, `OpenClipVisualEncoder.transform`) is, *unconditionally*:

```
resize_pil(img, 384)   # PIL BICUBIC, scale SHORTEST side to 384, keep aspect
crop_pil(img, 384)     # center-crop to 384×384
to_numpy / 255
normalize(mean=0.5, std=0.5)
```

`clip.py`'s SigLIP2 path uses HF `SiglipProcessor`, which **squashes** the whole
image to 384×384 (no crop, aspect distorted). open_clip's webli transform also
squashes (torchvision, antialiased). So:

* **Resampler** is the same family (PIL bicubic both sides) — *not* the problem.
* **Geometry** differs: Immich crops the center, MLX/SiglipProcessor distort the
  whole frame. On a 640×480 (4:3) photo Immich keeps the center 384×384 of a
  512×384 resize (drops ~25% width, no distortion) while SiglipProcessor warps
  640→384 horizontally and 480→384 vertically. Different pixels → different
  embedding → cosine ≈ **0.83**.

This is content/aspect dependent: it nearly vanishes on square inputs (an earlier
run with 512×512 images showed ~0.97–0.99) and is worst on wide/tall photos. Real
libraries are mostly non-square, so the production impact is the 0.83 regime.

Note this is an Immich *quirk*: the SigLIP2 model was trained with `resize_mode=
'squash'`, but Immich's `OpenClipVisualEncoder` always resize-shortest+crops
regardless. Since that is what built the index, **matching Immich (crop) — not the
model's training transform (squash) — is what preserves the index.**

## Why this is FIX-PREPROCESSING and not RE-INDEX

The model forward is identical: MLX reproduces HF `SiglipModel` to 1.0 for *any*
given pixels (the `transformers` column). Feeding Immich-preprocessed pixels into
the MLX model therefore reproduces the Immich-server embedding to ~1.0 (same
weights, same pixels). So:

* **Do NOT re-index.** The 4716 stored vectors stay valid.
* **ml-ycd.4:** replace the `SiglipProcessor` image path in
  `_encode_image_siglip2` with Immich's exact transform — PIL bicubic
  resize-shortest-side-to-384, center-crop 384, `/255`, normalize 0.5 — then feed
  the resulting `pixel_values` to `get_image_features`. Text is already at parity
  (cosine 1.0), so the tokenizer path is fine as-is.
* After the fix, re-run this harness; expect MLX-vs-Immich image cosine ≈ 1.0,
  flipping the verdict to PASS/PRESERVE.

## What the references isolate (why three of them)

* **immich** — HF weights (identical to the webli ONNX the NAS ran) + Immich's
  exact server transform. Reproduces the index embeddings → the real gate.
* **transformers** — same weights + HF squash (== `clip.py`'s current
  preprocessing). MLX≈1.0 here proves the mlx-embeddings port is exact, so the
  immich gap is preprocessing, not the port.
* **openclip** — open_clip's own torchvision transform. Diagnostic only; it is
  *not* what the NAS runs (the NAS uses its own Python preprocessing, above), so
  don't read its number as the index gate.

## Recommended verification (optional, ground truth)

Pull a handful of real stored embeddings from the live index plus their originals,
embed those files with the *fixed* MLX backend, and confirm cosine ≈ 1.0 against
the stored vectors. This closes the loop against ground truth rather than a
reconstructed reference. Filed as a follow-up bead.

## Reproduce

```bash
cd ml
# venv needs mlx-embeddings + torch/open-clip-torch/transformers (all in requirements.txt):
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/embedding_parity.py                       # downloaded 640×480 samples
.venv/bin/python scripts/embedding_parity.py --images ~/library    # real photos (preferred)
.venv/bin/python scripts/embedding_parity.py --ref immich transformers openclip --report out.txt
```

Exit: 0 = PASS/FIX-PREPROCESSING (preserve), 1 = genuine fail, 2 = inconclusive.

## Caveats

- Immich-server transform was replicated from `immich-app/immich` **main**; if the
  NAS runs an older release, confirm its `OpenClipVisualEncoder.transform` still
  does resize-shortest + center-crop (it has for a long time).
- Sample images are downloaded (picsum, deterministic by seed, 640×480). For the
  real gate, point `--images` at a representative sample of the actual library —
  the effect is aspect-ratio dependent.
- mlx-embeddings 0.1.0 transitively pulls `mlx-vlm`/`mlx-lm`; under the production
  pin `mlx<0.31.2` pip backtracks them to `mlx-vlm==0.4.4` / `mlx-lm==0.31.2`
  (newer versions hard-require `mlx>=0.31.2`). Install with the pin present.
