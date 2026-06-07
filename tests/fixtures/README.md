# Parity test fixtures

Committed inputs and golden references for the weights-gated parity gate
(`tests/test_clip_golden_parity.py`, `tests/test_face_golden_parity.py`).

## queries.txt
Fixed CLIP text queries. Mirrors `scripts/embedding_parity.py::DEFAULT_QUERIES`.

## clip/
Public-domain photos for CLIP image parity. Provenance per file:

| file | source URL | license |
|------|------------|---------|
| (filled by scripts/fetch_pd_clip_fixtures.py) | | |

## faces/
LFW subset for face parity (identity subdirectories, ≥2 images each).
Source: `logasja/lfw` (HuggingFace dataset). LFW is a research-use face
verification benchmark; only a minimal subset is committed here.

## golden/
Reference embeddings frozen from the upstream ONNX models Immich ships
(`immich-app/*` repos, insightface `buffalo_l`). Regenerate with
`scripts/gen_parity_golden.py`. Each `.json` manifest records the ONNX repo,
resolved commit SHA, onnxruntime version, dim, and generation date.
