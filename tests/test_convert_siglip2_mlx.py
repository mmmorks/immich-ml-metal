"""Tests for scripts/convert_siglip2_mlx.py.

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
    assert spec is not None and spec.loader is not None
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
    mod._require_patch_token("mlx-community/siglip2-so400m-patch16-384-4bit", "--upload-repo")


@pytest.mark.parametrize(
    "suffix",
    ["-4bit", "-8bit", "-bf16", "-fp16"],
)
def test_allows_publish_naming_suffixes(suffix):
    """Every mlx-community publish variant is dash-introduced, so the
    loader's ``(?:-|$)`` anchor matches and the guard must accept it."""
    mod._require_patch_token(f"mlx-community/siglip2-so400m-patch16-384{suffix}", "--upload-repo")


@pytest.mark.parametrize(
    "name",
    ["patch16-384bar", "patch16-384x", "patch16-384fp16", "x-patch16-384x"],
)
def test_rejects_trailing_junk_after_token(name):
    """Names where the patch digits are followed by non-dash junk PASS a guard
    without the ``(?:-|$)`` anchor but CRASH the loader (group(1) -> AttributeError
    in mlx_embeddings utils.py); the fixed guard must fail-fast on them."""
    with pytest.raises(ValueError, match="patch"):
        mod._require_patch_token(f"mmmorks/{name}", "--upload-repo")


def test_rejects_missing_token():
    with pytest.raises(ValueError, match="patch"):
        mod._require_patch_token("mmmorks/my-weights", "--upload-repo")


def test_error_message_names_the_flag():
    with pytest.raises(ValueError, match="--upload-repo"):
        mod._require_patch_token("mmmorks/my-weights", "--upload-repo")


# --- --upload-repo on an already-complete cache ---------------------------


def test_upload_existing_reads_config_and_uploads(tmp_path, monkeypatch):
    """_upload_existing must reconstruct upload_to_hub's `config` arg from the
    dir's config.json and forward the dir path + repo + hf_path unchanged."""
    import json

    import mlx_embeddings.utils as utils

    (tmp_path / "config.json").write_text(json.dumps({"vision_config": {}}))
    calls = []
    monkeypatch.setattr(utils, "upload_to_hub", lambda *a, **k: calls.append((a, k)))

    mod._upload_existing(tmp_path, "mmmorks/siglip2-so400m-patch16-384", "google/x")

    assert len(calls) == 1
    (path, repo, hf_path, config), _ = calls[0]
    assert path == str(tmp_path)
    assert repo == "mmmorks/siglip2-so400m-patch16-384"
    assert hf_path == "google/x"
    assert config == {"vision_config": {}}


def test_main_uploads_when_cache_complete(tmp_path, monkeypatch):
    """The regression: with a complete cache and no --force, passing
    --upload-repo must still publish (not silently return 0 with only a note)."""
    out = tmp_path / "siglip2-so400m-patch16-384"
    out.mkdir()

    monkeypatch.setattr(mod, "siglip2_dir_is_complete", lambda _p: True)
    uploaded = []
    monkeypatch.setattr(mod, "_upload_existing", lambda *a: uploaded.append(a))
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_siglip2_mlx.py",
            "--mlx-path",
            str(out),
            "--upload-repo",
            "mmmorks/siglip2-so400m-patch16-384",
        ],
    )

    assert mod.main() == 0
    assert len(uploaded) == 1
    assert uploaded[0][0] == out
    assert uploaded[0][1] == "mmmorks/siglip2-so400m-patch16-384"
