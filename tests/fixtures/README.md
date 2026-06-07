# Parity test fixtures

Committed inputs and golden references for the weights-gated parity gate
(`tests/test_clip_golden_parity.py`, `tests/test_face_golden_parity.py`).

## queries.txt
Fixed CLIP text queries. Mirrors `scripts/embedding_parity.py::DEFAULT_QUERIES`.

## clip/
Public-domain photos for CLIP image parity. Provenance per file:

| file | source URL | license |
|------|------------|---------|
| earth_blue_marble.jpg | https://images-assets.nasa.gov/image/PIA18033/PIA18033~orig.jpg | NASA — public domain |
| apollo17_moon.jpg | https://images-assets.nasa.gov/image/as17-148-22727/as17-148-22727~orig.jpg | NASA — public domain |
| nebula_hubble.jpg | https://images-assets.nasa.gov/image/GSFC_20171208_Archive_e000075/GSFC_20171208_Archive_e000075~orig.jpg | NASA — public domain |
| astronaut_spacewalk.jpg | https://images-assets.nasa.gov/image/iss040e090540/iss040e090540~orig.jpg | NASA — public domain |
| mars_curiosity.jpg | https://images-assets.nasa.gov/image/PIA16239/PIA16239~orig.jpg | NASA — public domain |
| nasa_pia12348.jpg | https://images-assets.nasa.gov/image/PIA12348/PIA12348~orig.jpg | NASA — public domain |
| nasa_pia17011.jpg | https://images-assets.nasa.gov/image/PIA17011/PIA17011~orig.jpg | NASA — public domain |
| nasa_pia03883.jpg | https://images-assets.nasa.gov/image/PIA03883/PIA03883~orig.jpg | NASA — public domain |

## faces/
LFW subset for face parity (identity subdirectories, ≥2 images each).
Source: `logasja/lfw` (HuggingFace dataset). LFW is a research-use face
verification benchmark; only a minimal subset is committed here. Committed
identities (3 × 2 images): `Aaron_Peirsol`, `Aaron_Sorkin`,
`Abdel_Nasser_Assidi`. Selected deterministically by
`scripts/gen_parity_golden.py --targets face` (`load_lfw`).

## golden/
Reference embeddings frozen from the upstream ONNX models Immich ships
(`immich-app/*` repos, insightface `buffalo_l`). Regenerate with
`scripts/gen_parity_golden.py`. Each `.json` manifest records the ONNX repo,
resolved commit SHA, onnxruntime version, dim, and generation date.

Regenerating the `openai_clip` golden needs the open_clip tokenizer
(`pip install open-clip-torch`); the gate tests themselves do not require it.
