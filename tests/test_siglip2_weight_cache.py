"""Local-cache resolution for converted SigLIP2 MLX weights — ml-ycd.7.

The accelerator should prefer locally-converted fp16 weights over downloading +
loading the HF bf16 safetensors on every install. These tests pin the pure
resolution logic (no weights, no network):

  * siglip2_cache_dir derives a dir name that still carries the 'patchNN-NNN'
    token the mlx-embeddings loader regex requires (ml-ycd.1), and honors the
    ML_MODEL_CACHE_DIR override.
  * siglip2_dir_is_complete only accepts a dir holding config + tokenizer +
    safetensors, so a half-written/aborted convert is ignored.
  * resolve_siglip2_source prefers the explicit ML_SIGLIP2_MLX_PATH override,
    then a complete cache dir, then falls back to the HF repo id.
"""
import re

import pytest

from src.models.clip import (
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
