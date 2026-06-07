# Fallback silent-harm audit

Audit of every fallback path in `ml/src/` for the hazard class that motivated this
audit: a "best-effort" fallback that **silently produces WRONG or INCOMPATIBLE output
instead of failing loudly** — the way the removed open_clip CLIP fallback served
squash-preprocessed embeddings (~0.83 cosine vs the index) and poisoned smart search
with only a log line as signal.

Classification:

- **(a) SAFE** — fallback output is equivalent/interchangeable with the normal path.
- **(b) LOUD** — degradation is surfaced (raises, `/health` degraded, or a flagged
  field) so a human/caller notices.
- **(c) SILENTLY HARMFUL** — produces wrong/incompatible output or a silent
  false-negative; must be made loud or removed. A follow-up bead is filed for each.

Guiding principle: **degrade-to-wrong is worse than fail-loud when the damage is
invisible until later.**

## (c) Silently harmful — follow-up beads filed

| Site | What it does | Why harmful | Failure mode |
|------|--------------|-------------|------|
| `clip.py:325-327` `_load_model` else-branch | Unknown (unmapped) CLIP model name → WARNING + serve `ViT-B-32__openai` | Wrong-model, index-incompatible vectors served for any unmapped name. Inconsistent with the already-hardened sibling branches, which **raise** for a None-backend model (`clip.py:317-324`) and for a SigLIP2 load failure. | **Wrong-model vectors** |
| `face_embed.py:313-331` (with `face_detect.py:226-234`) | Face missing `landmarks` → silent bbox-crop + resize instead of landmark `norm_crop` | bbox crop skips pose normalization → drifted ArcFace embedding mixed into the **same** index that the face-embedding parity gate verified parity for *only with* landmark alignment. Landmark-miss is DEBUG-only, fallback itself is silent. | **Silent bbox-crop fallback** |
| `ocr.py:89-91, 115-117, 168-170`; `main.py:341-343` | Hard failure (decode error, Vision error, inference exception) → return empty result | Does **not** poison with wrong vectors (stores nothing), but Immich can't tell "decode failed" from "no text/faces", marks the asset processed, and never retries → permanent silent false-negative. | **Silent empty result** |

## Explicitly adjudicated as NOT harmful (the two the bead called out + corrections)

- **SigLIP2 bf16 last-resort load** (`clip.py:245-247`, `ensure_siglip2_source`): the
  cache→download→convert→`hf` chain ends by loading the bf16 HF repo directly. **(b)
  LOUD-enough / borderline (a).** This is the *same weights and same fixed-res-384
  preprocessing* as the fp16 path, only a different reduced precision — embeddings are
  near-identical (≈0.999 cosine), nowhere near the open_clip 0.83 squash divergence, so
  bf16 is effectively index-interchangeable. Each step logs a WARNING. The only residual
  gap is that the served dtype/source isn't surfaced in the response or `/health`;
  low-priority hardening, not a poisoning hazard. **Verdict: keep.**
- **Batch → per-face inference fallback** (`face_embed.py:359-373`): **(a) SAFE.**
  ArcFace `get_feat` preprocesses each image independently (`cv2.dnn.blobFromImages`),
  so per-face output is numerically identical to the batched path; logged WARNING.
- **Batch-dim mismatch → per-face** (`face_embed.py:383-401`): **(a) / protective.** This
  is a *correctness guard* that detects a fixed-batch ONNX export returning too few rows
  and re-runs per-face — it prevents a silent mis-mapping rather than causing one.
- **`_l2_normalize` zero-norm guard** (`clip.py:28-36`): **(a) SAFE by design.** Returns
  the (detectably-degenerate) zero vector instead of dividing by zero into an all-NaN
  vector that *would* silently poison the index.
- **Missing-`modelName` defaults** (`main.py` visual/face/ocr task config): **(b).** When
  Immich omits `modelName` they fall back to `settings.clip_model` / `settings.face_model`
  — the deployment's configured model, i.e. the same model the index was built with, so
  the default is consistent, not wrong. In practice Immich always sends `modelName`.
- **`/health` degraded** (`main.py:378-437`): **(b) good.** Surfaces `status: "degraded"`
  on component failure; only the exception *string* is hidden in non-debug mode.
- **`config.py` log-level coercion** (`config.py:20-33, 102-104`): **(b) good.** Invalid
  level is coerced to INFO now and a WARNING is logged later (fallback-now/warn-later).
- **`embedding_parity.py` real→synthetic images**: **(b) good — the pattern to copy.**
  Sets `used_synthetic=True`, which flips the gate's exit code to 3 (inconclusive) and
  prints a banner. Degradation changes the outcome a human sees.

## Lower-severity, documented-only (no bead)

- `main.py` `vm_stat` parse failure → `9999` MB "plenty of memory" (`main.py:~80-95`):
  operational only (idle-unload memory floor wouldn't trigger under real pressure); not an
  index hazard.
- `main.py` unknown task type silently ignored (`main.py:~666-672`): version-skew only;
  we implement all current Immich task types.
- `clip.py` env-var defaults (`ML_SIGLIP2_HF_REPO`, `ML_MODEL_CACHE_DIR`,
  `ML_SIGLIP2_AUTO_CONVERT`): intentional configuration defaults, logged where they cause
  large writes.

## Method

`grep -rE 'fallback|except|getattr\('` over `src/**.py` (clip.py 37, face_embed.py 19,
main.py 17, ocr.py/face_detect.py/config.py the rest; `immich_preprocess.py`, `gpu_lock.py`,
`utils/__init__.py` have none), then each site read and classified against the code.
