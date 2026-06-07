"""Guards for the pinned-weights lock + its digest-verify logic.

The lock (ci/parity_weights.lock.json) is what makes the weights-gated parity
gate reproducible: it pins each upstream source by revision + sha256. These
tests prove the verify step actually rejects a digest mismatch (so an upstream
shift fails the build, not just skips), and — when the real weights happen to be
cached locally — that the committed digests still match the bytes on disk.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ML_ROOT / "ci" / "parity_weights.lock.json"
SCRIPT_PATH = ML_ROOT / "ci" / "fetch_parity_weights.py"


def _load_script():
    # ci/ is not a package; load the module by path like the other script tests.
    spec = importlib.util.spec_from_file_location("fetch_parity_weights", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_script()


def test_lock_is_well_formed():
    data = json.loads(LOCK_PATH.read_text())
    sources = data["sources"]
    assert sources, "lock has no sources"
    for name, spec in sources.items():
        assert spec["kind"] in {"hf_snapshot", "insightface_pack"}, name
        assert spec["verify_file"], name
        digest = spec["sha256"]
        assert len(digest) == 64 and int(digest, 16) >= 0, f"{name}: not a sha256"
        if spec["kind"] == "hf_snapshot":
            # A pinned revision is what stops an upstream shift; require a full sha.
            assert len(spec["revision"]) == 40, f"{name}: revision is not a 40-char commit sha"


def test_verify_digest_accepts_matching_file(tmp_path):
    f = tmp_path / "w.bin"
    payload = b"weights-bytes"
    f.write_bytes(payload)
    mod.verify_digest(f, hashlib.sha256(payload).hexdigest())  # must not raise


def test_verify_digest_rejects_drifted_file(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights-bytes")
    wrong = hashlib.sha256(b"different-bytes").hexdigest()
    with pytest.raises(mod.DigestMismatch):
        mod.verify_digest(f, wrong)


def test_verify_digest_rejects_missing_file(tmp_path):
    with pytest.raises(mod.DigestMismatch):
        mod.verify_digest(tmp_path / "absent.bin", "0" * 64)


def test_lock_agrees_with_service_weight_pins():
    """The CI lock and the service's runtime pins must not silently diverge.

    The service (src/models/weight_pins.py) pins SigLIP2 + ArcFace by revision +
    sha256 for its own runtime verification; the CI lock pins the same sources so
    the parity job can pre-stage them. They are two representations of the same
    bytes — if someone rolls one pin without the other, the gate could run
    against different weights than production. Assert they match so that can't
    happen quietly. (The OpenAI-CLIP port lives only in the CI lock: the service
    converts it unpinned, which the lock + the gate's refs/main pin cover.)
    """
    from src.models import weight_pins

    sources = json.loads(LOCK_PATH.read_text())["sources"]

    sig = sources["siglip2"]
    assert sig["repo"] == weight_pins.SIGLIP2_PINNED_REPO
    assert sig["revision"] == weight_pins.SIGLIP2_PINNED_REVISION
    assert sig["sha256"] == weight_pins.SIGLIP2_PINNED_SHA256[sig["verify_file"]]

    buf = sources["buffalo_l"]
    assert buf["sha256"] == weight_pins.ARCFACE_PINNED_SHA256[buf["pack"]][buf["verify_file"]]


@pytest.mark.parametrize("name", list(json.loads(LOCK_PATH.read_text())["sources"]))
def test_pinned_digest_matches_local_weights_if_present(name):
    """If the real weight is already cached locally, its bytes must match the pin.

    Skips when the weight isn't present (e.g. a clean checkout) so this stays a
    fast, network-free guard; CI's fetch step covers the download path.
    """
    spec = json.loads(LOCK_PATH.read_text())["sources"][name]
    try:
        target = mod._STAGERS[name](spec, check_only=True)
    except Exception as e:  # treat "can't locate locally" as not-present
        pytest.skip(f"{name}: weight not locatable offline ({e})")
    if not target.is_file():
        pytest.skip(f"{name}: weight not cached locally ({target})")
    mod.verify_digest(target, spec["sha256"])  # must not raise
