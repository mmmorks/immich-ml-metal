#!/usr/bin/env python3
"""Fetch and digest-verify the pinned model weights the parity gate consumes.

Reads ci/parity_weights.lock.json, downloads each source at its PINNED revision
into the same on-disk locations the production backends load from, then verifies
a sha256 over the file the gate actually uses. A mismatch is a hard failure, so
an upstream re-publish cannot silently shift embeddings underneath the gate.

This only stages the *download* sources. The OpenAI-CLIP port still needs a
one-time torch convert on a cold cache; that happens lazily inside the gated
test (or, on a warm CI cache, the converted weights are restored and no convert
— and no torch — is needed). SigLIP2 fp16 and the InsightFace pack are served
directly with no convert.

Usage: python ci/fetch_parity_weights.py [--check-only]
  --check-only: verify already-present weights against the lock without
                downloading (used by the unit test / for a fast local check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(__file__).resolve().parent / "parity_weights.lock.json"

# Importing the service modules keeps the on-disk paths in lockstep with what the
# backends actually load — no duplicated path logic that could drift.
sys.path.insert(0, str(ML_ROOT))


class DigestMismatch(RuntimeError):
    """A downloaded weight file's sha256 does not match the pinned digest."""


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_digest(path: Path, expected: str) -> None:
    """Raise DigestMismatch unless ``path`` hashes to ``expected`` (lowercase hex)."""
    if not path.is_file():
        raise DigestMismatch(f"missing weight file: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise DigestMismatch(f"{path}: sha256 {actual} != pinned {expected.lower()}")


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text())["sources"]


def _stage_siglip2(spec: dict, check_only: bool) -> Path:
    from src.models.clip import siglip2_cache_dir

    out = siglip2_cache_dir(spec["repo"])
    if not check_only:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=str(out))
    return out / spec["verify_file"]


def _stage_openai_clip(spec: dict, check_only: bool) -> Path:
    # Stage the source pickle at the pinned revision into the HF cache; the gated
    # test converts it to MLX lazily (torch, convert-only) on a cold cache.
    if not check_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            allow_patterns=["*.bin", "*.json", "*.txt"],
        )
    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(
        repo_id=spec["repo"],
        filename=spec["verify_file"],
        revision=spec["revision"],
        local_files_only=check_only,
    )
    return Path(resolved)


def _stage_buffalo(spec: dict, check_only: bool) -> Path:
    insightface_root = Path.home() / ".insightface"
    model_dir = insightface_root / "models" / spec["pack"]
    if not check_only:
        from insightface.utils.storage import download as download_model_pack

        from src.models.face_embed import _ensure_recognition_model_pack

        _ensure_recognition_model_pack(spec["pack"], download_model_pack)
    return model_dir / spec["verify_file"]


_STAGERS = {
    "siglip2": _stage_siglip2,
    "openai_clip": _stage_openai_clip,
    "buffalo_l": _stage_buffalo,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="verify already-present weights without downloading",
    )
    args = ap.parse_args(argv)

    sources = load_lock()
    failures: list[str] = []
    for name, spec in sources.items():
        stage = _STAGERS[name]
        try:
            target = stage(spec, args.check_only)
            verify_digest(target, spec["sha256"])
            print(f"OK   {name}: {target} matches pinned sha256")
        except Exception as e:  # collect all so one bad pin doesn't hide others
            failures.append(f"{name}: {e}")
            print(f"FAIL {name}: {e}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} weight pin(s) failed verification", file=sys.stderr)
        return 1
    print(f"\nAll {len(sources)} weight pins verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
