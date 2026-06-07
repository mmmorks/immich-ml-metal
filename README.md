# immich-ml-metal

> **⚠️ Important Disclaimer**: I (the repository owner) am not a software developer. This project was architected, designed, and primarily authored by Claude based on my requirements and feedback. While I've tested it in my home environment, please treat this as an **experimental community project** rather than production-ready software. I'm sharing this in hopes it's useful for the Mac community. If it's not helpful for you, please don't feel pressured to use it.

A Metal/ANE-optimized drop-in replacement for [Immich's](https://immich.app/) machine learning service, designed specifically for Apple Silicon Macs. This allows Mac users to run Immich's ML workloads natively on their hardware.

## What This Does

Immich's standard ML container runs well on NVIDIA, Intel, and AMD GPUs. Recently, the community has had trouble running it natively on Apple's ML framework (particularly after OCR was implemented). This project is a drop-in replacement for Immich-ML that uses the same ML API but uses as many native Apple ML frameworks as possible:

- **CLIP Embeddings**: MLX-accelerated for image/text search — a native MLX SigLIP2 backend for Immich's default `ViT-SO400M-16-SigLIP2-384__webli`, plus a vendored OpenAI-CLIP backend for the `*__openai` models
- **Face Detection**: Apple Vision framework (runs on Neural Engine)
- **Face Recognition**: InsightFace ArcFace with CoreML acceleration
- **OCR**: Apple Vision framework text recognition

## Performance: Why This is Fast

Apple Silicon has three independent compute units — GPU (Metal), Neural Engine (ANE), and CPU. This service runs ML tasks across all three concurrently:

| Task | Compute Unit | Framework |
|------|-------------|-----------|
| CLIP embedding | GPU (Metal) | MLX / mlx-embeddings (SigLIP2) + vendored OpenAI-CLIP |
| Face detection | ANE | Apple Vision |
| Face embedding | CPU / CoreML | InsightFace ONNX |
| OCR | ANE | Apple Vision |

Within a single `/predict` request, CLIP, face recognition, and OCR run simultaneously via `asyncio.gather`. Face embeddings are batched into a single ONNX inference call regardless of how many faces are in the photo.

**Benchmarks (M4, 24GB):**

| Photo | Faces | Latency |
|-------|-------|---------|
| 3.5MB portrait | 0 | 134ms |
| 5.5MB landscape | 0 | 149ms |
| 26MB group photo | 3 | 602ms |
| 23MB group photo | 5 | 589ms |

Per-task timing is logged on every request:
```
INFO:   clip: 25ms
INFO:   faces: 3 detected
INFO:   faces: 135ms
INFO:   ocr: 47ms
INFO: predict: 3 task(s) [clip+facial-recognition+ocr] completed in 135ms
```

### CLIP backend: vendored MLX vs upstream ONNX

The OpenAI CLIP ports (`ViT-B-16__openai`, `ViT-L-14__openai`) run on MLX/Metal
via the vendored OpenAI-CLIP backend (`src/models/clip_mlx.py`). To confirm that
path is at least as fast as what stock Immich ships — not just numerically
faithful — `scripts/clip_benchmark.py` times the
warm, batch-1 encode against the same upstream `immich-app/<model>` ONNX export
under both onnxruntime providers available on Apple Silicon (CPU, which is what
Docker Immich actually serves here since there's no CUDA; and CoreML). Warm
median latency, single-stream throughput in parentheses (M5 Pro, 24 GB):

| Model | Path | Image | Text |
|-------|------|------:|-----:|
| ViT-B-16 | **vendored (Metal)** | **10.0 ms** (100/s) | **2.3 ms** (439/s) |
| ViT-B-16 | upstream ONNX, CPU | 27.9 ms (36/s) | 8.0 ms (126/s) |
| ViT-B-16 | upstream ONNX, CoreML | 39.4 ms (25/s) | 23.0 ms (44/s) |
| ViT-L-14 | **vendored (Metal)** | **32.0 ms** (31/s) | **3.6 ms** (279/s) |
| ViT-L-14 | upstream ONNX, CPU | 147.0 ms (7/s) | 15.2 ms (66/s) |
| ViT-L-14 | upstream ONNX, CoreML | 216.9 ms (5/s) | 62.6 ms (16/s) |

The vendored MLX backend is **2.8–6.8× faster on images and 3.5–17× faster on text** than the
upstream ONNX baseline, and the gap widens with model size. (CoreML is *slower*
than plain CPU for these CLIP graphs — onnxruntime offloads only part of the
graph and pays for the partition.) Forward-only timing (compute alone, inputs
prepared once) tracks end-to-end within a couple ms, so preprocessing is not the
differentiator — the Metal forward itself is faster. A leaner hand-rolled MLX
path that reuses `immich_preprocess` and drives the raw module directly
(`direct_mlx` in the benchmark) lands within ~3% of the production wrapper, so it
adds no meaningful overhead and there's no performance case for a different CLIP
backend. Reproduce with `.venv/bin/python scripts/clip_benchmark.py` (needs
`pip install open-clip-torch` for the ONNX text tokenizer; the upstream ONNX
exports download once, ~0.6 GB B-16 / ~1.7 GB L-14).

## Project Status

** A(I)lpha Quality - Use at Your Own Risk**

- [x] CLIP implementation (native MLX SigLIP2 backend + vendored OpenAI-CLIP)
- [x] Face detection (Vision framework)
- [x] Face embeddings (InsightFace + CoreML)
- [x] OCR (Vision framework)
- [x] Basic integration testing with real Immich instance
- [ ] Community testing and validation

**Known Limitations:**
- Only tested in my specific home setup (MacBook Pro M1, macOS 26.1, Immich v2.4.1)
- Not all Immich ML features may be fully compatible
- Memory usage not extensively optimized
- No load testing performed
- Could literally only work on my machine(tm)

## Requirements

- **macOS Tahoe+** (Might work on earlier OSs, but you'd have to test it yourself)
- **Apple Silicon Mac** (M1/M2/M3/M4 - Intel Macs not supported)
- **Python 3.11** (Not working on 3.13)
- **Immich server** already running (this replaces just the ML service)

## Installation

```bash
# Clone the repository
git clone https://github.com/sebastianfredette/immich-ml-metal.git
cd immich-ml-metal

# Create and activate virtual environment
# Please ensure python 3.11 is used — 3.13 doesn't yet have all required wheels
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# First run will download models
python -m src.main
```

## Configuration

Configure via environment variables or edit `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port number (must match Immich config) |
| `ML_MODELS_DIR` | `./models` | Model storage directory |
| `ML_CLIP_MODEL` | `ViT-SO400M-16-SigLIP2-384__webli` | CLIP model name (fallback when a request omits one; SigLIP2 default needs no torch) |
| `ML_FACE_MODEL` | `buffalo_l` | Face recognition model (buffalo_s/m/l) |
| `ML_FACE_MIN_SCORE` | `0.7` | Face detection confidence threshold |
| `ML_OCR_LANGUAGE_CORRECTION` | `true` | Language correction for OCR (disable for codes/serials) |
| `ML_USE_COREML` | `true` | Enable CoreML acceleration |
| `ML_USE_ANE` | `true` | Enable Apple Neural Engine |
| `ML_MAX_CONCURRENT_REQUESTS` | `4` | Max queued requests before backpressure |
| `MODEL_UNLOAD_STRATEGY` | `pressure` | `pressure`: unload when RAM is low + idle. `timeout`: unload after idle timeout. `never`: keep loaded. |
| `MODEL_IDLE_TIMEOUT` | `120` | Seconds before unloading idle models (only used with `timeout` strategy) |
| `MODEL_MEMORY_FLOOR_MB` | `500` | Available RAM threshold that triggers model unloading (only used with `pressure` strategy) |
| `ML_LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| `ML_LOG_REQUESTS` | `true` | Log individual requests (disable for high volume) |
| `ML_DEBUG_MODE` | `false` | Expose error details (keep false when network-exposed) |

### Model Choices

**CLIP Model Mapping** (resolved in `src/models/clip.py`):

- **SigLIP2 SO400M -> native MLX** (via [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings))
  - `ViT-SO400M-16-SigLIP2-384__webli` -> `google/siglip2-so400m-patch16-384`
  - This is Immich's current default smart-search model. See
    [Native SigLIP2 backend](#native-siglip2-backend) below.

- OpenAI & LAION CLIP models -> MLX (via the **vendored CLIP backend**,
  `src/models/clip_mlx.py`), converting the HF checkpoint on first use and running it
  with **standard `gelu`** to match Immich's ONNX export (see
  [CLIP parity](#clip-parity-vendored-openai-clip-path)). Its resize-shortest +
  center-crop image processor matches the Immich index, so these are verified
  parity-faithful.
  - `ViT-B-32__openai` -> `openai/clip-vit-base-patch32`
  - `ViT-B-16__openai`-> `openai/clip-vit-base-patch16`
  - `ViT-L-14__openai`-> `openai/clip-vit-large-patch14`
  - `ViT-B-32__laion2b-s34b-b79k` / `ViT-B-32__laion2b_s34b_b79k` ->
    `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
  - For the **OpenAI** ports, standard `gelu` is an override of the checkpoint's
    native `quick_gelu`; **LAION** was trained with standard `gelu` natively, so the
    same setting is correct for both.
  - **Needs `pip install torch` for the first-use conversion only.** torch is an
    optional, convert-only dependency (not in `requirements.txt`) — it reads the
    source PyTorch checkpoint pickle. Serving the cached MLX weights afterward, and
    the default SigLIP2 path, need no torch. Requesting one of these ports on a fresh,
    torch-free install raises an actionable error pointing you here.

- Other SigLIP models -> **unsupported** (no MLX backend; raises a clear error)
  - `ViT-B-16-SigLIP__webli`
  - `ViT-B-16-SigLIP2__webli`

- Unknown/unmapped model name: **raises** a clear error (the `default` mapping is
  reachable only via an explicit `default` request, for internal/test use)

The ViT-B-16 SigLIP variants have no parity-faithful MLX backend, so requesting one
raises a clear error rather than silently serving non-parity embeddings (see
[Parity-or-fail](#parity-or-fail)). **An earlier wrong-weights bug
made every OpenAI port except `ViT-B-32__openai` silently load OpenAI B-32 weights**:
the third-party loader took its first arg as a local `model_dir`, and passing the
repo id there (instead of as `hf_repo`) made an absent dir fall back to that loader's
default (OpenAI B-32) regardless of the requested name. The vendored backend takes
`hf_repo` explicitly, and `_assert_mlx_clip_checkpoint` verifies the loaded vision
tower's arch at load time so this can't silently recur.

#### CLIP parity (vendored OpenAI-CLIP path)

Unlike SigLIP2, the OpenAI CLIP models run through a vendored CLIP backend
(`src/models/clip_mlx.py`) with its own image processor and BPE tokenizer rather
than the Immich-faithful `immich_preprocess` path, so their index-compatibility
is checked by a dedicated gate, `scripts/clip_parity.py` — the production backend
vs the same open_clip checkpoint Immich exports to ONNX, run through Immich's
*exact* transform.

All four supported ports are **verified drop-ins, no re-index** — image **and** text
cosine `1.0000` (12 photos × 12 queries), top-1 retrieval agreement `1.000`, vs the
Immich-transform reference:

- **`ViT-B-32__openai`** — `1.0000` / `1.0000`.
- **`ViT-B-16__openai`** — `1.0000` / `1.0000`.
- **`ViT-L-14__openai`** — `1.0000` / `1.0000`.
- **`ViT-B-32__laion2b-s34b-b79k`** — `1.0000` / `1.0000`.

**Activation — the subtle part.** OpenAI CLIP's *native* activation is `quick_gelu`,
but Immich's shipped ONNX export for these ports runs **standard `gelu`**. (Verified:
the `immich-app/<model>` ONNX export matches a standard-gelu open_clip checkpoint to
cosine `1.0000` on identical pixels/tokens, and a quick_gelu reference at only
~`0.96–0.97` — the gap scales with depth: L-14 > B-16.) Since the goal is parity with
the *index Immich actually built*, the vendored backend deliberately runs **standard
gelu**, not the checkpoint's native quick_gelu — and the gate's open_clip reference
uses the plain (standard-gelu) arch to match. A `quick_gelu` backend (what the
previous third-party `mlx_clip` package hardcoded) drifts ~`0.97` from a stock-Immich
OpenAI-CLIP index — a query/index mismatch the *old* gate missed because it compared
against a quick_gelu reference. The vendored backend made the activation configurable;
`src/models/clip.py` forces `gelu` for these ports.

Text uses `clean_text(canonicalize=False)` then the CLIP BPE tokenizer (OpenAI BPE is
case/punctuation-bearing, unlike SigLIP); images use resize-shortest-224 + center-crop
+ CLIP-normalize — reproducing the standard Immich server. **LAION** (`ViT-B-32` /
`laion2b_s34b_b79k`) runs through the same vendored backend: its HF repo ships a
transformers-format checkpoint the convert path reads, and its native activation is
already standard `gelu`, so no override is needed — verified `1.0000` / `1.0000`
against the open_clip `laion2b_s34b_b79k` reference. Re-check any model with
`.venv/bin/python scripts/clip_parity.py --model <name>`.

#### CLIP speed (vendored OpenAI-CLIP path)

Parity proves the vendored backend is *correct*; `scripts/clip_benchmark.py`
proves it is also *faster* than the upstream ONNX path it replaced — the vendored
MLX path (Metal) beats the upstream ONNX-CPU baseline (what Immich's Docker image
runs on Apple Silicon) by 2.8–6.8× on images and 3.5–17× on text for the OpenAI
ports. Numbers and
methodology are in [Performance: Why This is Fast](#performance-why-this-is-fast).

### Native SigLIP2 backend

`ViT-SO400M-16-SigLIP2-384__webli` runs natively on the Metal GPU through
mlx-embeddings (no PyTorch in the hot path). This is the recommended smart-search
model on Apple Silicon.

What makes it a **drop-in** for Immich's standard ML server — embeddings are
interchangeable with the index Immich already built, so **no re-index is needed**:

- **Bit-faithful weights.** The MLX port matches HF `transformers` to cosine
  `1.0000` on every test item.
- **Immich-faithful preprocessing.** Images use resize-shortest-side-to-384 +
  center-crop + normalize 0.5 (not HF `SiglipProcessor`, which squashes to
  384×384 and diverges on non-square photos). Text uses Immich's `clean_text`
  canonicalization + the raw `tokenizer.json` padded/truncated to 64. Both live
  in `src/models/immich_preprocess.py`.
- **Verified parity.** Against the standard Immich server on real photos:
  image cosine `0.9999 / 1.0000` (min/mean), text cosine `1.0000`. Output is a
  1152-dim, L2-normalized vector (manually normalized — `get_image_features` /
  `get_text_features` return un-normalized pooled output).
- **Verified end-to-end** against a live Immich smart-search workload: 1152-dim,
  `L2 == 1.0`, warm latency ~50 ms text / ~110 ms image.

**Weight loading & caching.** The accelerator prefers a local fp16 cache
(~2.2 GB vs 4.3 GB bf16) and materializes one the first time none exists, so no
setup is required and later loads are smaller/faster. Resolution order is:

1. `ML_SIGLIP2_MLX_PATH` — explicit pre-converted dir (highest precedence).
2. The local cache dir (`models/siglip2-so400m-patch16-384`, or
   `$ML_MODEL_CACHE_DIR/...`) if a complete convert exists there.
3. **Pre-converted HF download** — snapshot a published fp16 repo
   (`ML_SIGLIP2_HF_REPO`, default `mlx-community/siglip2-so400m-patch16-384`) into
   the local cache dir, then load it (`source=cache`). Fast — no local convert.
4. **On-demand convert** — if the download is disabled/unavailable, a one-time
   fp16 convert runs into the local cache dir, then loads from it (`source=cache`).
5. The HF repo bf16 safetensors — fallback if both of the above are off/fail.

Steps 3–4 each run once per machine, then every later load is `source=cache`.
Control them with:

- `ML_SIGLIP2_HF_REPO` — pre-converted fp16 repo to snapshot (default
  `mlx-community/siglip2-so400m-patch16-384`); set empty to skip the download step.
- `ML_SIGLIP2_AUTO_CONVERT=0` — skip the local convert (~2.2 GB write); load HF
  bf16 directly instead.

**Pinned weights & integrity.** Model weights stay HuggingFace/InsightFace-hosted,
but the sources we ship are pinned for reproducibility, so an upstream re-publish
or a corrupted download can't silently shift embeddings (the pins live in
`src/models/weight_pins.py`):

- The default SigLIP2 download (step 3) is fetched at a fixed HF **revision**, and
  the `model.safetensors` + `tokenizer.json` are verified against recorded
  sha256s. A user-supplied `ML_SIGLIP2_HF_REPO` override is unvetted, so it is
  fetched as-is (no pin, no checksum).
- The ArcFace recognition model in the `buffalo_l` pack is verified against its
  recorded sha256; a stale cached copy that mismatches is re-downloaded once.
- In every case a **checksum mismatch is a hard failure** — never a silent
  fallback to different weights.

To intentionally adopt new weights, recompute the digests and update
`weight_pins.py`: a HF LFS file's `oid` (from
`https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=true`) *is* its
sha256, and `shasum -a 256 <file>` covers anything local (e.g. the unzipped
`buffalo_l` pack).

You can also pre-convert explicitly (e.g. ahead of first traffic, or to
`--verify`), optionally publishing the result so other machines hit step 3:

```bash
.venv/bin/python scripts/convert_siglip2_mlx.py --verify
.venv/bin/python scripts/convert_siglip2_mlx.py \
    --upload-repo mmmorks/siglip2-so400m-patch16-384   # publish for step 3
```

Any local weights directory name **must** contain a `patchNN-NNN` token (e.g.
`patch16-384`), because the mlx-embeddings loader regex-parses the patch size
from the path; the convert script, the download step, and the on-demand convert
all enforce/preserve this.

**Quantization — why the default stays fp16.** We evaluated 8-bit and 4-bit
weight quantization for speed/memory vs accuracy
(`scripts/quantization_eval.py`, 12 photos × 12 queries, cosine vs the fp16
convert). The decision: **keep fp16; offer 8-bit text-only as an opt-in for
memory-constrained installs.** The numbers:

| precision | disk | peak mem | image cos | text cos | top-1 retrieval | verdict |
|---|---|---|---|---|---|---|
| fp16 (default) | 2.12 GiB | 2.63 GiB | — | — | — | baseline |
| 8-bit text-only | 1.62 GiB | 2.14 GiB | 1.0000 | 0.9999 | 1.000 | near-lossless opt-in |
| 4-bit text-only | 1.35 GiB | 1.87 GiB | 1.0000 | 0.9447 | **0.750** | rejected (search rankings drift) |
| 8/4-bit incl. vision | — | — | — | — | — | **unsupported** by mlx-embeddings 0.1.0 |

Two findings drive this:

- **The vision tower can't be quantized** on mlx-embeddings 0.1.0: its SigLIP
  `MultiheadAttentionPoolingHead` indexes a raw `in_proj.weight`, which is
  invalid for a `QuantizedLinear`, so a vision-quantized model converts but
  crashes at image-encode time. Every *runnable* variant is therefore
  text-tower-only — image embeddings stay bit-identical to fp16 (`1.0000`), and
  image-encode latency is unchanged. Since smart-search indexing is image-bound,
  quantization buys no speed there.
- **8-bit text-only is safe; 4-bit is not.** 8-bit leaves text-query embeddings
  and retrieval rankings effectively unchanged (~19% lower peak memory, ~24%
  smaller on disk). 4-bit drops text cosine to `0.9447` and flips 1 in 4 query
  top-results — unacceptable for search quality.

To use the 8-bit text-only weights (e.g. on a memory-tight machine), convert
with quantization and point the accelerator at the result:

```bash
.venv/bin/python scripts/quantization_eval.py --configs 8bit-textonly --keep
export ML_SIGLIP2_MLX_PATH=$PWD/models/quant_eval/siglip2-so400m-patch16-384-8bit-textonly
```

Revisit a true default change once mlx-embeddings can quantize the SigLIP vision
attention (that's where the real memory/latency win for an image workload lives).

### Parity-or-fail

A CLIP model is served only by a backend whose preprocessing matches the upstream
Immich ML server. There is **no degrade-to-a-different-backend safety net**,
because a wrong-but-working backend poisons the smart-search index invisibly —
worse than failing, since the damage only surfaces later as degraded search.

The hazard is preprocessing. Upstream Immich's `OpenClipVisualEncoder.transform`
hardcodes resize-shortest-side + center-crop for *every* CLIP model. A generic
SigLIP transform, by contrast, **squashes** inputs to a square, which lands at
only ~0.83 cosine vs the existing index for `ViT-SO400M-16-SigLIP2-384__webli`.

So:

- The native MLX SigLIP2 backend is upstream-faithful (it reuses Immich's exact
  transform; see [Native SigLIP2 backend](#native-siglip2-backend)). A load
  failure **raises** rather than falling back, so a partial cache / version drift
  can't poison the index; `/health` then reports degraded.
- OpenAI and LAION CLIP models are served by the vendored CLIP backend
  (`src/models/clip_mlx.py`), converting the HF checkpoint and running it with
  **standard `gelu`** + center-crop preprocessing to match Immich's ONNX export
  (parity-faithful).
- A model with no parity-faithful backend — the `ViT-B-16-SigLIP*` variants —
  **raises a clear "no parity-faithful MLX backend" error** instead of silently
  serving non-parity vectors.

`scripts/embedding_parity.py` has an optional `openclip` diagnostic backend (a
regression witness for the squash-vs-crop divergence); `pip install open-clip-torch`
to run it.

#### Automated parity gate (committed golden references)

The manual `*_parity.py` scripts need a human to supply `--images`, so a parity
regression is invisible to automation. A committed-golden gate closes that gap:

- **Fixtures** (`tests/fixtures/`): ~8 public-domain CLIP photos, a fixed query
  list, and a tiny LFW face subset — all committed, so the gate is hermetic.
- **Golden references** (`tests/fixtures/golden/*.npz`): embeddings frozen from
  the literal upstream ONNX models Immich ships (the `immich-app/*` exports and
  insightface `buffalo_l`), generated once via
  `.venv/bin/python scripts/gen_parity_golden.py` and committed with a `.json`
  manifest (ONNX repo + resolved commit SHA, onnxruntime version, dim, date).
  Regenerating the `openai_clip` golden needs `pip install open-clip-torch`.
- **Gated tests** assert the MLX production backends stay faithful to those
  goldens: `tests/test_clip_golden_parity.py` (SigLIP2 + OpenAI-CLIP image/text
  cosine ≥ 0.99) and `tests/test_face_golden_parity.py` (Apple-Vision fork vs
  SCRFD/ArcFace golden: median alignment-drift cosine ≥ 0.90, top-1 retrieval
  drop ≤ 0.02).

The gate auto-detects availability: it **skips** when the native deps/weights or
golden artifacts are absent (e.g. a non-Apple-Silicon CI lane), so a plain
`.venv/bin/python -m pytest` stays green everywhere. Set `ML_RUN_PARITY=1` to
turn a missing prerequisite into a **hard failure** instead — so a CI lane that
should run the gate can't silently skip it:

```bash
ML_RUN_PARITY=1 .venv/bin/python -m pytest tests/test_clip_golden_parity.py tests/test_face_golden_parity.py
```

The `*_parity_harness.py` tests cover only the harness math (cosine, IoU,
retrieval); model-output parity lives in the golden gate above.

**Face Models**:
- `buffalo_s`
- `buffalo_m`
- `buffalo_l` - **default**

### Face-embedding parity (preserve vs re-scan)

Face **recognition** is the same upstream model and metric: InsightFace
`buffalo_l` ArcFace (`w600k_r50`), bit-identical 112×112 `norm_crop` alignment
(`face_embed._norm_crop` reimplements insightface's 5-point similarity warp on
skimage's current `SimilarityTransform.from_estimate` instead of the deprecated
in-place `estimate()`; verified equal to 0 ULP in `tests/test_face_align_parity.py`),
and cosine search. The one thing this fork changes is **detection +
5-point landmarks** — upstream Immich uses the `buffalo_l` SCRFD detector
(`det_10g.onnx`; the "RetinaFace" in old notes), which emits keypoints directly,
while this fork reconstructs the 5 points from Apple Vision face-landmark
contours (`src/models/face_detect.py`). Different landmarks → a slightly
different aligned crop → a drifted embedding for the *same* face, so face-index
compatibility had to be measured, not assumed.

`scripts/face_embedding_parity.py` runs both detectors over a labelled face set,
matches each physical face by bounding-box IoU, and routes **both** keypoint sets
through the production `src/models/face_embed.py` so landmarks are the only
variable. On a 40-identity / 287-matched-face LFW sample:

| metric | result |
|---|---|
| alignment-drift cosine (upstream-kps vs Vision-kps, identical ArcFace) | median **0.980**, p5 0.959, p1 0.941, min 0.858 |
| faces with drift cosine ≥ 0.90 / ≥ 0.95 | **99.3% / 97.9%** |
| top-1 identity: upstream→upstream / fork→fork / **fork→upstream (preserve test)** | 0.951 / 0.951 / **0.951 (zero drop)** |
| detection coverage: Vision recall vs SCRFD | **0.854** (Vision missed 49 of 336; found 1 extra) |

**Recommendation: PRESERVE the existing face index and clusters — a re-scan is
not required for embedding compatibility.** A face stored in the index (embedded
via upstream landmarks) and re-detected by this fork lands at cosine ≈0.98 of its
stored vector, and fork-detected query faces retrieve the correct identity from a
stored-upstream index with **no measurable accuracy drop**. New faces join the
right clusters.

Two caveats, both about *detection coverage* rather than embedding drift:

- Apple Vision detects a **different set** of faces than SCRFD (~85% recall on
  this frontal set; profiles, small, or occluded faces in a real library will
  diverge more). Going forward some faces Docker would have indexed may not be
  re-detected (and vice versa) — this changes which faces exist, not whether the
  embeddings of detected faces are compatible.
- Vision confidence is **not** calibrated like SCRFD's `det_score`, so
  `ML_FACE_MIN_SCORE=0.7` is not the same operating point as Docker's threshold
  (on clean LFW faces every Vision detection scored ≥0.7).

LFW is clean, frontal, near-best-case data. To validate against your own library,
re-run on real photos (one `identity/*.jpg` subdir per person for the top-1
metric):

```bash
.venv/bin/python scripts/face_embedding_parity.py --images ~/face-samples --report face_parity.md
```

## Connecting to Immich

In your Immich `docker-compose.yml` or `.env`:

```yaml
# Point to your Mac's IP address (not localhost unless Immich is also native)
MACHINE_LEARNING_URL=http://192.168.1.100:3003
```

## Actually Running the Service

```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 3003 --workers 1
```

### Running as a Service (macOS launchd):

Create `~/Library/LaunchAgents/com.immich.ml.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.immich.ml</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/.venv/bin/python</string>
        <string>-m</string>
        <string>src.main</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/immich-ml-metal</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/immich-ml.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/immich-ml-error.log</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.immich.ml.plist
launchctl start com.immich.ml
```

## Verification

Test the service is working:

```bash
# Health check
curl http://localhost:3003/ping
# Should return: pong

# Service info
curl http://localhost:3003/
# Should return: {"message":"Immich ML"}

# Detailed health (checks all components)
curl http://localhost:3003/health
# Should return: {"status":"healthy","checks":{...}}

# Test CLIP text encoding
curl -X POST http://localhost:3003/predict \
  -F 'entries={"clip":{"textual":{"modelName":"ViT-B-32__openai"}}}' \
  -F 'text=a photo of a cat'
```

In Immich, you should see the ML service connect in the admin logs.

## Contributing

Given this is primarily an AI-assisted project, contributions are **very welcome**, especially from actual developers who can:

- Review and improve the code quality
- Add proper tests
- Verify Immich compatibility  
- Optimize performance
- Add missing features
- Improve documentation

## Support

This is a hobby project with no guarantees. I barely know how to use Git.

## Final Notes

This project exists because I wanted to run Immich ML on my Mac with passable hardware acceleration. **It works for me**, but may not work for everyone. Use at your own risk, and please contribute improvements if you can!

---

**tl;dr**: AI-written Immich ML service for Apple Silicon. Alpha quality. Use with caution. Please help.