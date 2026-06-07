"""Harness math for scripts/embedding_parity.py (NOT model parity).

Model-output parity is gated by tests/test_clip_golden_parity.py against
committed ONNX golden references; this file covers only the helper math
(download retry, cosine, stats, retrieval_agreement).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "embedding_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("embedding_parity", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _raise(exc):
    def _f(*_a, **_k):
        raise exc

    return _f


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Keep retry backoff instant."""
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_download_one_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(_req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient")
        return _FakeResp(b"jpegbytes")

    monkeypatch.setattr(mod.urllib.request, "urlopen", flaky)
    assert mod._download_one("http://x", attempts=3, base_delay=0) == b"jpegbytes"
    assert calls["n"] == 3


def test_download_one_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(mod.urllib.request, "urlopen", _raise(OSError("down")))
    with pytest.raises(OSError, match="down"):
        mod._download_one("http://x", attempts=2, base_delay=0)


def test_load_images_fails_fast_without_allow_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_download_one", _raise(OSError("net")))
    with pytest.raises(SystemExit) as ei:
        mod.load_images(None, 3, tmp_path, allow_synthetic=False)
    # Loud, actionable message; default behaviour is to fail, not silently degrade.
    assert "synthetic" in str(ei.value).lower()


def test_load_images_loud_synthetic_fallback_when_allowed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mod, "_download_one", _raise(OSError("net")))
    images, used_synthetic = mod.load_images(None, 4, tmp_path, allow_synthetic=True)
    assert used_synthetic is True
    assert len(images) == 4
    assert all(lbl.startswith("synthetic_") for lbl, _ in images)
    out = capsys.readouterr().out.lower()
    assert "synthetic" in out  # the fallback must be loud


def test_load_images_dir_returns_reals_no_synthetic(tmp_path):
    for i in range(2):
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(tmp_path / f"img{i}.png")
    images, used_synthetic = mod.load_images(tmp_path, 5, tmp_path / "cache")
    assert used_synthetic is False
    assert len(images) == 2  # capped at what's available


# --- cosine ------------------------------------------------------------------


def test_cosine_identical_is_one():
    v = np.array([0.3, 0.4, 0.5])
    assert mod.cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert mod.cosine(a, b) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    a = np.array([1.0, 2.0, 3.0])
    assert mod.cosine(a, -a) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.zeros(3), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.zeros(3)),
        (np.zeros(3), np.zeros(3)),
    ],
)
def test_cosine_zero_vector_is_nan_not_division_error(a, b):
    """A genuinely zero embedding must yield nan (guarded), never a 0/0 warning
    or a crash — the harness uses this to flag degenerate samples, not poison the
    aggregate cosine."""
    assert math.isnan(mod.cosine(a, b))


# --- stats -------------------------------------------------------------------


def test_stats_computes_min_mean_median_max():
    sims = np.array([0.2, 0.4, 0.6, 0.8])
    s = mod.stats(sims)
    assert s == {
        "min": pytest.approx(0.2),
        "mean": pytest.approx(0.5),
        "median": pytest.approx(0.5),
        "max": pytest.approx(0.8),
    }


def test_stats_single_value_collapses():
    s = mod.stats(np.array([0.7]))
    assert s["min"] == s["mean"] == s["median"] == s["max"] == pytest.approx(0.7)


# --- retrieval_agreement -----------------------------------------------------


def test_retrieval_agreement_perfect_when_backends_match():
    """Identical sim matrices under both backends -> full top-1 agreement and a
    matrix correlation of 1.0."""
    img = np.eye(3)  # 3 orthonormal image embeds
    txt = np.eye(3)  # query i best-matches image i under both backends
    agree = mod.retrieval_agreement(img, txt, img, txt)
    assert agree["top1_agreement"] == pytest.approx(1.0)
    assert agree["matrix_corr"] == pytest.approx(1.0)


def test_retrieval_agreement_counts_top1_disagreement():
    """One query whose best image flips between backends drops top-1 agreement
    to 0.5 (1 of 2 queries agree)."""
    img = np.eye(2)
    mlx_txt = np.array([[1.0, 0.0], [0.0, 1.0]])  # q0->img0, q1->img1
    ref_txt = np.array([[1.0, 0.0], [1.0, 0.0]])  # q0->img0, q1->img0 (flipped)
    agree = mod.retrieval_agreement(img, mlx_txt, img, ref_txt)
    assert agree["top1_agreement"] == pytest.approx(0.5)


def test_retrieval_agreement_1x1_corr_is_nan_not_crash():
    """A single image + single query gives a 1-element flattened sim matrix;
    np.corrcoef of one point is nan. The function must return that nan rather than
    raising, and top-1 agreement is trivially 1.0 (the only image is the best)."""
    img = np.array([[1.0, 0.0]])  # one image embed
    txt = np.array([[0.0, 1.0]])  # one query embed
    agree = mod.retrieval_agreement(img, txt, img, txt)
    assert agree["top1_agreement"] == pytest.approx(1.0)
    assert math.isnan(agree["matrix_corr"])
