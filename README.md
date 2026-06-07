# immich-ml-metal

> **⚠️ Important Disclaimer**: I (the repository owner) am not a software developer. This project was architected, designed, and primarily authored by Claude based on my requirements and feedback. While I've tested it in my home environment, please treat this as an **experimental community project** rather than production-ready software. I'm sharing this in hopes it's useful for the Mac community. If it's not helpful for you, please don't feel pressured to use it.

A Metal/ANE-optimized drop-in replacement for [Immich's](https://immich.app/) machine learning service, designed specifically for Apple Silicon Macs. This allows Mac users to run Immich's ML workloads natively on their hardware.

## What This Does

Immich's standard ML container runs well on NVIDIA, Intel, and AMD GPUs. Recently, the community has had trouble running it natively on Apple's ML framework (particularly after OCR was implemented). This project is a drop-in replacement for Immich-ML that uses the same ML API but uses as many native Apple ML frameworks as possible:

- **CLIP Embeddings**: MLX-accelerated for image/text search — including a native MLX SigLIP2 backend (Immich's default `ViT-SO400M-16-SigLIP2-384__webli`), with an open_clip/MPS fallback for other models
- **Face Detection**: Apple Vision framework (runs on Neural Engine)
- **Face Recognition**: InsightFace ArcFace with CoreML acceleration
- **OCR**: Apple Vision framework text recognition

## Performance: Why This is Fast

Apple Silicon has three independent compute units — GPU (Metal), Neural Engine (ANE), and CPU. This service runs ML tasks across all three concurrently:

| Task | Compute Unit | Framework |
|------|-------------|-----------|
| CLIP embedding | GPU (Metal) | MLX / mlx-embeddings (SigLIP2) / open_clip MPS |
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

## Project Status

** A(I)lpha Quality - Use at Your Own Risk**

- [x] CLIP implementation (MLX, native MLX SigLIP2 backend, open_clip fallback)
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

# Pin mlx-clip to a specific commit (for stability)
# Get the current commit hash:
git ls-remote https://github.com/harperreed/mlx_clip.git HEAD
# Edit requirements.txt and replace the tail end of the mlx_clip.git address with the new hash

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
| `ML_CLIP_MODEL` | `ViT-B-32__openai` | CLIP model name |
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

- OpenAI CLIP models -> MLX (via [mlx_clip](https://github.com/harperreed/mlx_clip))
  - `ViT-B-32__openai` -> `mlx-community/clip-vit-base-patch32`
  - `ViT-B-16__openai`-> `mlx-community/clip-vit-base-patch16`
  - `ViT-L-14__openai`-> `mlx-community/clip-vit-large-patch14`

- LAION CLIP models -> MLX
  - `ViT-B-32__laion2b-s34b-b79k`-> `mlx-community/clip-vit-base-patch32-laion2b`
  - `ViT-B-32__laion2b_s34b_b79k`-> `mlx-community/clip-vit-base-patch32-laion2b`

- Other SigLIP models -> open_clip fallback (MPS)
  - `ViT-B-16-SigLIP__webli`
  - `ViT-B-16-SigLIP2__webli`

- Default fallback: `mlx-community/clip-vit-base-patch32`

Any model name not mapped above is resolved to an open_clip `(arch, pretrained)`
pair by splitting on `__`, and any MLX load failure falls back to open_clip — see
[The open_clip fallback](#the-open_clip-fallback).

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

**Weight loading & caching.** Out of the box, weights load directly from the HF
bf16 safetensors on first use (cached under `~/.cache/huggingface`) — no setup
required. For a smaller, faster-loading install you can pre-convert once to fp16
(~2.2 GB vs 4.3 GB bf16):

```bash
.venv/bin/python scripts/convert_siglip2_mlx.py --verify
```

This writes to the repo's gitignored `models/` dir, and the accelerator
**auto-loads from there** with no further configuration. Resolution order is:

1. `ML_SIGLIP2_MLX_PATH` — explicit pre-converted dir (highest precedence).
2. The local cache dir (`models/siglip2-so400m-patch16-384`, or
   `$ML_MODEL_CACHE_DIR/...`) if a complete convert exists there.
3. The HF repo bf16 safetensors (default fallback).

Any local weights directory name **must** contain a `patchNN-NNN` token (e.g.
`patch16-384`), because the mlx-embeddings loader regex-parses the patch size
from the path; the convert script enforces this.

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

### The open_clip fallback

The open_clip + PyTorch/MPS path is **kept** as a best-effort safety net. It
covers:

- CLIP/SigLIP architectures with no native MLX mapping (e.g.
  `ViT-B-16-SigLIP__webli`, or any `arch__pretrained` name).
- Recovery when an MLX/mlx-embeddings load fails (partial cache, version
  mismatch) — the service degrades to open_clip instead of failing the request.

It is intentionally a fallback, not the primary path: it pulls in `torch` +
`open-clip-torch`, and for `ViT-SO400M-16-SigLIP2-384__webli` it is **not**
parity-guaranteed (open_clip's own preprocessing differs from the Immich-faithful
path above), so a fallback embedding for that model may not match the existing
index. Prefer the native backend for the default model. Fully dropping open_clip
would slim the install but remove support for the non-MLX models and the
recovery path, so it is retained for now.

**Face Models**:
- `buffalo_s`
- `buffalo_m`
- `buffalo_l` - **default**

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