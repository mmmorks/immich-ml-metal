"""Pinned weight revisions + sha256 verification helpers.

Model weights come from upstream hosts (HuggingFace for SigLIP2, the InsightFace
release for ArcFace). We pin each source to a fixed revision/release and verify
the downloaded bytes against a recorded sha256 so an upstream re-publish can't
silently shift our embeddings. These tests pin the pure verification logic — no
weights, no network.
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.models.weight_pins import (
    ARCFACE_PINNED_SHA256,
    SIGLIP2_PINNED_REPO,
    SIGLIP2_PINNED_REVISION,
    SIGLIP2_PINNED_SHA256,
    ChecksumMismatch,
    sha256_file,
    verify_dir,
    verify_sha256,
)


def _write(path, data: bytes):
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# --- sha256_file -------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "blob"
    expected = _write(f, b"the quick brown fox" * 1000)
    assert sha256_file(f) == expected


def test_sha256_file_streams_large_input(tmp_path):
    """Chunked read must produce the same digest as a one-shot hash (guards a
    buggy chunk loop that drops or double-counts a block)."""
    f = tmp_path / "big"
    expected = _write(f, b"\xa5" * (3 * (1 << 20) + 7))  # > a few chunks, non-aligned
    assert sha256_file(f) == expected


# --- verify_sha256 -----------------------------------------------------------


def test_verify_sha256_passes_on_match(tmp_path):
    f = tmp_path / "blob"
    digest = _write(f, b"hello")
    verify_sha256(f, digest)  # must not raise


def test_verify_sha256_is_case_insensitive(tmp_path):
    f = tmp_path / "blob"
    digest = _write(f, b"hello")
    verify_sha256(f, digest.upper())  # hex case must not matter


def test_verify_sha256_raises_on_mismatch(tmp_path):
    f = tmp_path / "blob"
    _write(f, b"hello")
    with pytest.raises(ChecksumMismatch):
        verify_sha256(f, "0" * 64)


# --- verify_dir --------------------------------------------------------------


def test_verify_dir_passes_when_all_match(tmp_path):
    digests = {
        "a.bin": _write(tmp_path / "a.bin", b"aaa"),
        "b.bin": _write(tmp_path / "b.bin", b"bbb"),
    }
    verify_dir(tmp_path, digests)  # must not raise


def test_verify_dir_raises_on_missing_file(tmp_path):
    _write(tmp_path / "a.bin", b"aaa")
    with pytest.raises(ChecksumMismatch):
        verify_dir(tmp_path, {"a.bin": _write(tmp_path / "a.bin", b"aaa"), "missing.bin": "0" * 64})


def test_verify_dir_raises_on_mismatch(tmp_path):
    _write(tmp_path / "a.bin", b"aaa")
    with pytest.raises(ChecksumMismatch):
        verify_dir(tmp_path, {"a.bin": "0" * 64})


# --- pinned manifest ---------------------------------------------------------


def test_siglip2_pin_is_a_full_revision_and_lfs_digests():
    """The SigLIP2 pin must name a concrete 40-hex git revision and 64-hex sha256
    digests for the two embedding-bearing LFS files (a floating 'main' or a
    truncated digest would let upstream drift through)."""
    assert len(SIGLIP2_PINNED_REVISION) == 40
    assert set(SIGLIP2_PINNED_SHA256) == {"model.safetensors", "tokenizer.json"}
    for digest in SIGLIP2_PINNED_SHA256.values():
        assert len(digest) == 64
    assert SIGLIP2_PINNED_REPO == "mlx-community/siglip2-so400m-patch16-384"


def test_arcface_pin_covers_buffalo_l_recognition_model():
    assert "w600k_r50.onnx" in ARCFACE_PINNED_SHA256["buffalo_l"]
    assert len(ARCFACE_PINNED_SHA256["buffalo_l"]["w600k_r50.onnx"]) == 64


# --- agreement with the CI parity-gate lock ----------------------------------
#
# The CI weights-gated parity gate pins the same upstream weights independently
# in ci/parity_weights.lock.json (it pre-fetches + verifies them without
# importing src). These are two pin-stores for two layers — the runtime/install
# load path here, and the CI gate there — so this guard fails the build if a pin
# is rolled in one place but not the other.

_LOCK = json.loads((Path(__file__).resolve().parents[1] / "ci" / "parity_weights.lock.json").read_text())["sources"]


def test_siglip2_pin_agrees_with_ci_lock():
    lock = _LOCK["siglip2"]
    assert SIGLIP2_PINNED_REPO == lock["repo"]
    assert SIGLIP2_PINNED_REVISION == lock["revision"]
    # The gate verifies model.safetensors; the runtime pin must agree on it.
    assert SIGLIP2_PINNED_SHA256[lock["verify_file"]] == lock["sha256"]


def test_arcface_pin_agrees_with_ci_lock():
    lock = _LOCK["buffalo_l"]
    assert ARCFACE_PINNED_SHA256["buffalo_l"][lock["verify_file"]] == lock["sha256"]
