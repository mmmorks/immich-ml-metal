"""Tests for scripts/convert_siglip2_mlx.py — ml-98c.

Covers the loader-regex guard. The script validates its --mlx-path dir name; this
adds the same guard for --upload-repo so we never publish a repo whose name lacks
the 'patchNN-NNN' token (which would crash mlx-embeddings' load() on consumers).

The script is loaded by path (scripts/ is not a package) so these run without a
heavyweight convert.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_siglip2_mlx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_siglip2_mlx", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def test_accepts_valid_upload_repo():
    # Should not raise.
    mod._require_patch_token("mmmorks/siglip2-so400m-patch16-384", "--upload-repo")


def test_accepts_bare_dir_name():
    mod._require_patch_token("siglip2-so400m-patch16-384", "--mlx-path dir name")


def test_allows_precision_suffix():
    """The loader regex (patch\\d+-(\\d+)(?:-|$)) tolerates a trailing -suffix,
    so the guard must accept conventional quant/dtype suffixes too."""
    mod._require_patch_token(
        "mlx-community/siglip2-so400m-patch16-384-4bit", "--upload-repo"
    )


def test_rejects_missing_token():
    with pytest.raises(ValueError, match="patch"):
        mod._require_patch_token("mmmorks/my-weights", "--upload-repo")


def test_error_message_names_the_flag():
    with pytest.raises(ValueError, match="--upload-repo"):
        mod._require_patch_token("mmmorks/my-weights", "--upload-repo")
