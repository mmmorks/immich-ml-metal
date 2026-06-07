"""Local-cache resolution for converted SigLIP2 MLX weights.

The accelerator should prefer locally-converted fp16 weights over downloading +
loading the HF bf16 safetensors on every install. These tests pin the pure
resolution logic (no weights, no network):

  * siglip2_cache_dir derives a dir name that still carries the 'patchNN-NNN'
    token the mlx-embeddings loader regex requires, and honors the
    ML_MODEL_CACHE_DIR override.
  * siglip2_dir_is_complete only accepts a dir holding config + tokenizer +
    safetensors, so a half-written/aborted convert is ignored.
  * resolve_siglip2_source prefers the explicit ML_SIGLIP2_MLX_PATH override,
    then a complete cache dir, then falls back to the HF repo id.
"""

import re
from pathlib import Path

import pytest

from src.models.clip import (
    ensure_siglip2_source,
    resolve_siglip2_source,
    siglip2_cache_dir,
    siglip2_dir_is_complete,
)

REPO = "google/siglip2-so400m-patch16-384"


def _populate(path):
    """Write the minimal file set a complete converted dir must contain."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"\x00")


# --- siglip2_cache_dir -------------------------------------------------------


def test_cache_dir_keeps_patch_token():
    """The loader regex parses patch size from the path, so the cache dir name
    must retain the 'patchNN-NNN' token (here from the repo basename)."""
    d = siglip2_cache_dir(REPO)
    assert re.search(r"patch\d+-\d+", d.name), d


def test_cache_dir_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    d = siglip2_cache_dir(REPO)
    assert d == tmp_path / "siglip2-so400m-patch16-384"


def test_cache_dir_default_is_repo_models_dir(monkeypatch):
    """Without the override the cache lives under the repo's gitignored models/."""
    monkeypatch.delenv("ML_MODEL_CACHE_DIR", raising=False)
    d = siglip2_cache_dir(REPO)
    assert d.parent.name == "models"


# --- siglip2_dir_is_complete -------------------------------------------------


def test_complete_dir_accepted(tmp_path):
    _populate(tmp_path / "m")
    assert siglip2_dir_is_complete(tmp_path / "m") is True


def test_missing_dir_rejected(tmp_path):
    assert siglip2_dir_is_complete(tmp_path / "nope") is False


def test_dir_without_weights_rejected(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")  # no *.safetensors -> incomplete
    assert siglip2_dir_is_complete(d) is False


def test_dir_without_tokenizer_rejected(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")  # no tokenizer.json
    assert siglip2_dir_is_complete(d) is False


def test_sharded_weights_accepted(tmp_path):
    """The completeness check globs ``*.safetensors`` precisely because a large
    convert is written as ``model-00001-of-0000N.safetensors`` shards rather than
    a single ``model.safetensors`` (see the docstring). A dir holding only shards
    — no monolithic file — must still count as complete."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
    (d / "model-00002-of-00002.safetensors").write_bytes(b"\x00")
    assert not (d / "model.safetensors").exists()  # only shards, no monolith
    assert siglip2_dir_is_complete(d) is True


# --- resolve_siglip2_source --------------------------------------------------


def test_resolve_prefers_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_SIGLIP2_MLX_PATH", "/some/override/patch16-384")
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    _populate(tmp_path / "siglip2-so400m-patch16-384")  # cache exists but override wins
    path, source = resolve_siglip2_source(REPO)
    assert source == "override"
    assert path == "/some/override/patch16-384"


def test_resolve_uses_complete_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    cache = tmp_path / "siglip2-so400m-patch16-384"
    _populate(cache)
    path, source = resolve_siglip2_source(REPO)
    assert source == "cache"
    assert path == str(cache)


def test_resolve_falls_back_to_hf_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))  # empty -> no cache
    path, source = resolve_siglip2_source(REPO)
    assert source == "hf"
    assert path == REPO


def test_resolve_ignores_incomplete_cache(monkeypatch, tmp_path):
    """A half-written convert (no safetensors) must not be picked up."""
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    partial = tmp_path / "siglip2-so400m-patch16-384"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    path, source = resolve_siglip2_source(REPO)
    assert source == "hf"
    assert path == REPO


# --- ensure_siglip2_source (on-demand convert) -------------------------------


def _patch_convert(monkeypatch, conv):
    """Patch mlx_embeddings.convert.convert (the submodule attr the impl imports).

    ``mlx_embeddings.__init__`` re-exports the convert function, shadowing the
    submodule on the package object. ``import mlx_embeddings.convert`` (and a
    dotted setattr target) therefore resolve to the function, not the module —
    CPython's import-as does a getattr first. importlib.import_module returns the
    real submodule from sys.modules, which is what ``from mlx_embeddings.convert
    import convert`` reads at call time, so patching it there takes effect.
    """
    import importlib

    conv_mod = importlib.import_module("mlx_embeddings.convert")
    monkeypatch.setattr(conv_mod, "convert", conv)


def _fake_convert(*, boom=False, complete=True):
    """Build a stub mlx_embeddings convert() that records calls and, on success,
    writes a (complete or partial) converted dir at mlx_path."""
    calls = []

    def _convert(hf_path, mlx_path, dtype="float16", upload_repo=None):
        calls.append({"hf_path": hf_path, "mlx_path": mlx_path, "dtype": dtype})
        if boom:
            raise RuntimeError("convert exploded")
        p = Path(mlx_path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}")
        (p / "model.safetensors").write_bytes(b"\x00")
        if complete:
            (p / "tokenizer.json").write_text("{}")

    return _convert, calls


@pytest.fixture
def hf_miss(monkeypatch, tmp_path):
    """No override, empty cache root, HF-repo download disabled, auto-convert on.

    Isolates the local-convert branch: ML_SIGLIP2_HF_REPO is blanked so the
    pre-converted download step is skipped and resolution goes straight to convert.
    """
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.delenv("ML_SIGLIP2_AUTO_CONVERT", raising=False)
    monkeypatch.setenv("ML_SIGLIP2_HF_REPO", "")  # skip the download step
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_ensure_returns_override_without_converting(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_SIGLIP2_MLX_PATH", "/some/override/patch16-384")
    conv, calls = _fake_convert()
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("override", "/some/override/patch16-384")
    assert calls == [], "must not convert when an override is set"


def test_ensure_returns_complete_cache_without_converting(hf_miss, monkeypatch):
    cache = hf_miss / "siglip2-so400m-patch16-384"
    _populate(cache)
    conv, calls = _fake_convert()
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("cache", str(cache))
    assert calls == [], "must not re-convert when a complete cache exists"


def test_ensure_converts_on_hf_miss(hf_miss, monkeypatch):
    cache = hf_miss / "siglip2-so400m-patch16-384"
    conv, calls = _fake_convert()
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("cache", str(cache))
    assert len(calls) == 1
    assert calls[0]["hf_path"] == REPO
    assert calls[0]["mlx_path"] == str(cache)
    assert calls[0]["dtype"] == "float16"


def test_ensure_respects_disable_env(hf_miss, monkeypatch):
    monkeypatch.setenv("ML_SIGLIP2_AUTO_CONVERT", "0")
    conv, calls = _fake_convert()
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("hf", REPO)
    assert calls == [], "must not convert when auto-convert is disabled"


def test_ensure_falls_back_to_hf_on_convert_error(hf_miss, monkeypatch):
    conv, calls = _fake_convert(boom=True)
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("hf", REPO)
    assert len(calls) == 1, "convert was attempted, then we fell back"


def test_ensure_falls_back_to_hf_on_incomplete_convert(hf_miss, monkeypatch):
    conv, calls = _fake_convert(complete=False)  # no tokenizer.json written
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("hf", REPO)


# --- ensure_siglip2_source: pre-converted HF-repo download -------------------


def _fake_snapshot(*, boom=False, complete=True):
    """Build a stub huggingface_hub.snapshot_download that records calls and, on
    success, writes a (complete or partial) converted dir into local_dir."""
    calls = []

    def _snap(repo_id, local_dir, **kwargs):
        calls.append({"repo_id": repo_id, "local_dir": local_dir})
        if boom:
            raise RuntimeError("download exploded")
        p = Path(local_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}")
        (p / "model.safetensors").write_bytes(b"\x00")
        if complete:
            (p / "tokenizer.json").write_text("{}")
        return str(p)

    return _snap, calls


@pytest.fixture
def hf_repo_set(monkeypatch, tmp_path):
    """No override/cache; HF-repo download enabled with an explicit repo id."""
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.delenv("ML_SIGLIP2_AUTO_CONVERT", raising=False)
    monkeypatch.setenv("ML_SIGLIP2_HF_REPO", "acme/siglip2-so400m-patch16-384")
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_ensure_downloads_pre_converted_before_converting(hf_repo_set, monkeypatch):
    cache = hf_repo_set / "siglip2-so400m-patch16-384"
    snap, dl_calls = _fake_snapshot()
    conv, conv_calls = _fake_convert()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    _patch_convert(monkeypatch, conv)

    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("cache", str(cache))
    assert dl_calls == [{"repo_id": "acme/siglip2-so400m-patch16-384", "local_dir": str(cache)}]
    assert conv_calls == [], "download succeeded, so no local convert"


def test_ensure_default_hf_repo_is_ours(monkeypatch, tmp_path):
    """With nothing set, the download step targets the canonical mlx-community repo."""
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.delenv("ML_SIGLIP2_HF_REPO", raising=False)
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    snap, dl_calls = _fake_snapshot()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    path, source = ensure_siglip2_source(REPO)
    assert source == "cache"
    assert dl_calls[0]["repo_id"] == "mlx-community/siglip2-so400m-patch16-384"


def test_ensure_skips_download_when_repo_blank(monkeypatch, tmp_path):
    """ML_SIGLIP2_HF_REPO='' disables the download step -> straight to convert."""
    monkeypatch.delenv("ML_SIGLIP2_MLX_PATH", raising=False)
    monkeypatch.setenv("ML_SIGLIP2_HF_REPO", "")
    monkeypatch.setenv("ML_MODEL_CACHE_DIR", str(tmp_path))
    snap, dl_calls = _fake_snapshot()
    conv, conv_calls = _fake_convert()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert source == "cache"
    assert dl_calls == [], "blank repo must skip the download step"
    assert len(conv_calls) == 1, "fell through to local convert"


def test_ensure_converts_when_download_fails(hf_repo_set, monkeypatch):
    snap, dl_calls = _fake_snapshot(boom=True)
    conv, conv_calls = _fake_convert()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert source == "cache"
    assert len(dl_calls) == 1, "download was attempted"
    assert len(conv_calls) == 1, "then fell through to local convert"


def test_ensure_converts_when_download_incomplete(hf_repo_set, monkeypatch):
    snap, dl_calls = _fake_snapshot(complete=False)  # no tokenizer.json
    conv, conv_calls = _fake_convert()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    _patch_convert(monkeypatch, conv)
    path, source = ensure_siglip2_source(REPO)
    assert source == "cache"
    assert len(conv_calls) == 1, "incomplete download -> local convert"


def test_ensure_hf_bf16_when_download_fails_and_convert_disabled(hf_repo_set, monkeypatch):
    monkeypatch.setenv("ML_SIGLIP2_AUTO_CONVERT", "0")
    snap, dl_calls = _fake_snapshot(boom=True)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    path, source = ensure_siglip2_source(REPO)
    assert (source, path) == ("hf", REPO)
    assert len(dl_calls) == 1
