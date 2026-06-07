"""Tests for scripts/embedding_parity.py — ml-7fa.

A single failed sample-image download used to silently switch the WHOLE run to
synthetic images (discarding already-fetched reals) with a zero exit, so the
preserve-vs-reindex gate could pass on weak synthetic data. These cover:
  * per-image download retry,
  * fail-fast (SystemExit) when a download ultimately fails and synthetic isn't
    explicitly allowed,
  * loud, opt-in synthetic fallback that flags the run via ``used_synthetic``.

The script is loaded by path (scripts/ is not a package), so these run without a
heavyweight backend import.
"""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "embedding_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("embedding_parity", _SCRIPT)
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
