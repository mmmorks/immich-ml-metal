"""Pinned weight revisions + sha256 digests, and verification helpers.

Model weights are fetched from upstream hosts: SigLIP2 fp16 MLX weights from a
HuggingFace repo, the ArcFace recognition model from the InsightFace release. To
keep embeddings reproducible — so an upstream re-publish or a corrupted download
can't silently shift what we index — each source is pinned to a fixed
revision/release here and the downloaded bytes are verified against the recorded
sha256. A mismatch is a hard error (:class:`ChecksumMismatch`), never a silent
fallback to different weights.

Refreshing a pin (when we *intentionally* adopt new weights): see the README
"Weight loading & caching" section for how to recompute these digests.
"""

import hashlib
from pathlib import Path

# SigLIP2 fp16 MLX weights — the default pre-converted download source
# (``ML_SIGLIP2_HF_REPO``). Pinned to a revision so re-converting/re-uploading
# the same logical model upstream can't change our embeddings, and the two
# embedding-bearing LFS files are verified by sha256 (their HF LFS oids).
SIGLIP2_PINNED_REPO = "mlx-community/siglip2-so400m-patch16-384"
SIGLIP2_PINNED_REVISION = "47bb082a6eeb9789923d547a845ca2c55ae39245"
SIGLIP2_PINNED_SHA256 = {
    "model.safetensors": "5eb53435c5cec8bb41bd74aca4c4eca03cc698c4533048b4a3f03ed73bd0378f",
    "tokenizer.json": "caefd63119539a63be2d55ef3e05023fbb793948c4bda5bc0c366b42a382f903",
}

# InsightFace ArcFace recognition model inside each ``buffalo_*`` pack
# (pack name -> {recognition-onnx filename: sha256}). The pack is pulled from the
# pinned ``v0.7`` InsightFace release; we verify the one file we actually load
# for embeddings. Packs without an entry here are loaded without verification.
ARCFACE_PINNED_SHA256 = {
    "buffalo_l": {"w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"},
}


class ChecksumMismatch(RuntimeError):
    """A downloaded weight file's sha256 does not match its pinned digest."""


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Stream ``path`` and return its hex sha256 (chunked, so multi-GB weights
    don't load into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Raise :class:`ChecksumMismatch` unless ``path``'s sha256 equals
    ``expected`` (hex comparison is case-insensitive)."""
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ChecksumMismatch(f"{path}: sha256 {actual} != pinned {expected}")


def verify_dir(dir_path: Path, digests: dict[str, str]) -> None:
    """Verify every ``{filename: sha256}`` under ``dir_path``. A missing file or a
    digest mismatch raises :class:`ChecksumMismatch`."""
    base = Path(dir_path)
    for name, expected in digests.items():
        f = base / name
        if not f.exists():
            raise ChecksumMismatch(f"{f}: pinned file missing, cannot verify checksum")
        verify_sha256(f, expected)
