"""Backend tests for MLXClip.

Covers the contract of the native MLX SigLIP2 backend (mlx-embeddings) and the
model-name routing/caching in get_clip_model, without loading real weights:

  * embedding shape == 1152 and L2-normalized (image-only and text-only paths)
  * the un-normalized pooled output from get_*_features is normalized in-place
    while its direction is preserved
  * name mapping invariants (MLX_EMBEDDINGS_MAP regex requirement, SigLIP2 names
    routed to the native backend rather than mlx_clip)
  * get_clip_model normalizes "::" -> "__", caches the loaded instance, and
    unloads the old model when a different name is requested

Concurrency/metal-lock behavior for the same backend lives in
test_clip_concurrency.py; this file deliberately does not duplicate it.

A single opt-in integration test exercises the running ML service end-to-end;
it is skipped unless ML_SERVICE_URL points at a reachable instance.
"""

import io
import os
import re
import threading

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import src.models.clip as clip_module
from src.models.clip import (
    MLX_EMBEDDINGS_MAP,
    MODEL_MAP,
    MLXClip,
    get_clip_model,
)

SIGLIP2_NAME = "ViT-SO400M-16-SigLIP2-384__webli"
SIGLIP2_DIM = 1152


def _red_jpeg() -> bytes:
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- Fakes for the native SigLIP2 backend (no real weights) -------------------


class _FakeSiglip2Processor:
    """Stand-in for the SiglipProcessor returned by mlx_embeddings.load()."""

    def __call__(self, images=None, text=None, **kwargs):
        if images is not None:
            return {"pixel_values": "pixels"}
        return {"input_ids": "ids"}


class _FakeSiglip2Model:
    """Returns a fixed UN-normalized 1152-d pooled output.

    get_image_features / get_text_features yield an object whose [0] converts to
    the raw vector via np.array() — mirroring how clip.py forces lazy Metal eval.
    Using a non-unit, non-uniform vector lets a test assert both that the result
    is L2-normalized and that its direction is preserved.
    """

    def __init__(self, raw):
        self._raw = np.asarray(raw, dtype=np.float32)

    def _features(self):
        raw = self._raw

        class _Feats:
            def __getitem__(self, idx):
                return raw

        return _Feats()

    def get_image_features(self, pixel_values=None):
        return self._features()

    def get_text_features(self, input_ids=None):
        return self._features()


def _bare_siglip2(model, processor=None):
    """Build a SigLIP2-backed MLXClip without loading real weights.

    The SigLIP2 encode paths preprocess via
    src.models.immich_preprocess (siglip_image_pixels + a SiglipTextTokenizer)
    rather than the SiglipProcessor, so inject a callable tokenizer returning
    (1, ctx) int32 ids; the image path needs no processor.
    """
    clip = object.__new__(MLXClip)
    clip.model_name = SIGLIP2_NAME
    clip._model = model
    clip._processor = processor or _FakeSiglip2Processor()
    clip._tokenizer = None
    clip._siglip_tokenizer = lambda text: np.zeros((1, 64), dtype=np.int32)
    clip._loaded = True
    clip._inference_lock = threading.Lock()
    clip._use_mlx_embeddings = True
    return clip


# --- Embedding shape + L2 normalization --------------------------------------


def test_siglip2_image_embedding_shape_and_normalized():
    raw = np.arange(1, SIGLIP2_DIM + 1, dtype=np.float32)  # non-unit, non-uniform
    clip = _bare_siglip2(_FakeSiglip2Model(raw))

    emb = clip.encode_image(_red_jpeg())

    assert emb.shape == (SIGLIP2_DIM,)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    # Direction preserved: normalized == raw / ||raw||
    assert np.allclose(emb, raw / np.linalg.norm(raw), atol=1e-6)


def test_siglip2_text_embedding_shape_and_normalized():
    raw = np.linspace(-3.0, 5.0, SIGLIP2_DIM, dtype=np.float32)
    clip = _bare_siglip2(_FakeSiglip2Model(raw))

    emb = clip.encode_text("a photo of a cat")

    assert emb.shape == (SIGLIP2_DIM,)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(emb, raw / np.linalg.norm(raw), atol=1e-6)


def test_siglip2_image_and_text_paths_independent():
    """Image and text routes both produce a valid unit embedding from one model."""
    clip = _bare_siglip2(_FakeSiglip2Model(np.ones(SIGLIP2_DIM, dtype=np.float32)))

    img_emb = clip.encode_image(_red_jpeg())
    txt_emb = clip.encode_text("hello")

    for emb in (img_emb, txt_emb):
        assert emb.shape == (SIGLIP2_DIM,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)


# --- Zero-embedding guard -----------------------------------------------------
#
# A genuinely zero pooled output (degenerate input / fp16 underflow) divided by
# its zero L2 norm yields an all-NaN vector that silently poisons the smart-
# search index or query. The encode paths must leave a zero vector untouched
# rather than producing NaN.


def test_siglip2_image_zero_embedding_does_not_nan():
    clip = _bare_siglip2(_FakeSiglip2Model(np.zeros(SIGLIP2_DIM, dtype=np.float32)))

    emb = clip.encode_image(_red_jpeg())

    assert emb.shape == (SIGLIP2_DIM,)
    assert not np.isnan(emb).any(), "zero embedding must not normalize to NaN"
    assert np.all(emb == 0.0), "a zero raw output should stay zero, not become NaN"


def test_siglip2_text_zero_embedding_does_not_nan():
    clip = _bare_siglip2(_FakeSiglip2Model(np.zeros(SIGLIP2_DIM, dtype=np.float32)))

    emb = clip.encode_text("a photo of a cat")

    assert emb.shape == (SIGLIP2_DIM,)
    assert not np.isnan(emb).any(), "zero embedding must not normalize to NaN"
    assert np.all(emb == 0.0), "a zero raw output should stay zero, not become NaN"


# --- mlx_clip path: OpenAI/LAION CLIP via mlx_clip ----------------------------
#
# The non-SigLIP path uses the mlx_clip backend for OpenAI/LAION CLIP models.
# Two things this guards:
#   * TEXT must be whitespace-canonicalized (Immich clean_text(canonicalize=False))
#     BEFORE tokenization so query embeddings line up with the Immich server's.
#     canonicalize=False (whitespace only, no lowercase/punctuation strip) because
#     OpenAI/LAION BPE is case- and punctuation-bearing, unlike SigLIP.
#   * The text path must return a normalized float32 ndarray. mlx_clip's
#     high-level text_encoder() returns a Python list (.tolist()), which then
#     breaks _l2_normalize/.flatten(); the path uses the low-level
#     model(input_ids=...) instead, mirroring the image path.


class _FakeMlxClipModel:
    """Stand-in for mlx_clip's loaded model.

    Records the text seen at BOTH the tokenizer (the low-level path the fixed
    code uses) and text_encoder (the legacy list-returning high-level call), so a
    test can assert whitespace canonicalization regardless of which is invoked.
    ``model(input_ids=...)`` returns a fixed raw mx.array as ``text_embeds[0]``,
    matching how clip.py forces lazy Metal eval via np.array().
    """

    def __init__(self, raw):
        self._raw = mx.array(np.asarray(raw, dtype=np.float32))
        self.seen: list[str] = []

    # Legacy high-level API (pre-fix code calls this) — returns a Python list.
    def text_encoder(self, text):
        self.seen.append(text)
        return np.asarray(self._raw).tolist()

    # Low-level APIs used by the image path and the fixed text path.
    def tokenizer(self, texts):
        self.seen.append(texts[0] if isinstance(texts, list) else texts)
        return [[1, 2, 3]]  # dummy ids; the fake model ignores them

    def img_processor(self, images):
        return "pixels"  # ignored by the fake model

    def model(self, input_ids=None, pixel_values=None):
        import types

        return types.SimpleNamespace(text_embeds=[self._raw], image_embeds=[self._raw])


def _bare_mlx_clip(model):
    """Build an mlx_clip-backed MLXClip without real weights.

    Leaves _use_mlx_embeddings unset so encode_text/encode_image take the default
    mlx_clip path (the open_clip fallback was removed).
    """
    clip = object.__new__(MLXClip)
    clip.model_name = "ViT-B-32__openai"
    clip._model = model
    clip._loaded = True
    clip._inference_lock = threading.Lock()
    return clip


def test_mlx_clip_text_embedding_shape_and_normalized():
    raw = np.arange(1, 9, dtype=np.float32)  # non-unit, non-uniform
    clip = _bare_mlx_clip(_FakeMlxClipModel(raw))

    emb = clip.encode_text("a photo of a cat")

    assert emb.shape == (8,)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(emb, raw / np.linalg.norm(raw), atol=1e-6)


def test_mlx_clip_text_path_canonicalizes_whitespace():
    model = _FakeMlxClipModel(np.arange(1, 9, dtype=np.float32))
    clip = _bare_mlx_clip(model)

    clip.encode_text("  a   photo\tof\na cat \n")

    assert model.seen == ["a photo of a cat"], f"text must be whitespace-canonicalized before tokenization, got {model.seen!r}"


def test_mlx_clip_text_zero_embedding_does_not_nan():
    clip = _bare_mlx_clip(_FakeMlxClipModel(np.zeros(8, dtype=np.float32)))

    emb = clip.encode_text("a photo of a cat")

    assert not np.isnan(emb).any(), "zero embedding must not normalize to NaN"
    assert np.all(emb == 0.0), "a zero raw output should stay zero, not become NaN"


def test_mlx_clip_image_embedding_shape_and_normalized():
    """Characterization of the already-working mlx_clip IMAGE path the parity
    gate relies on (the gate's mlxclip backend calls encode_image)."""
    raw = np.arange(1, 9, dtype=np.float32)
    clip = _bare_mlx_clip(_FakeMlxClipModel(raw))

    emb = clip.encode_image(_red_jpeg())

    assert emb.shape == (8,)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(emb, raw / np.linalg.norm(raw), atol=1e-6)


# --- Name mapping invariants -------------------------------------------------


def test_mlx_embeddings_map_repos_have_patch_token():
    """The mlx-embeddings loader regex-parses patch/image size from the repo id
    (config.json omits patch_size), so every mapped repo MUST contain a
    'patchNN-NNN' token or load() crashes."""
    pat = re.compile(r"patch\d+-\d+")
    assert MLX_EMBEDDINGS_MAP, "expected at least one native SigLIP2 mapping"
    for name, repo in MLX_EMBEDDINGS_MAP.items():
        assert pat.search(repo), f"{name} -> {repo!r} lacks a patchNN-NNN token"


# --- Parity-or-fail load contract (no fallback backend) ----------------------
#
# A model is served only by a backend whose preprocessing matches Immich's. A
# native SigLIP2 load failure propagates (nothing serves index-incompatible
# squash embeddings), an unsupported model raises a clear error, and an unknown
# name raises rather than degrading to the mlx_clip default.

UNSUPPORTED_NAME = "ViT-B-16-SigLIP2__webli"  # mapped to None in MODEL_MAP


def _bare_for_load(name=SIGLIP2_NAME):
    """A bare MLXClip with just enough state to call _load_model()."""
    clip = object.__new__(MLXClip)
    clip.model_name = name
    return clip


def test_native_siglip2_load_failure_propagates(monkeypatch):
    """A native backend load failure raises — no fallback silently serves
    index-incompatible embeddings."""

    def boom(self):
        raise RuntimeError("native backend exploded")

    monkeypatch.setattr(MLXClip, "_load_siglip2_mlx", boom)

    clip = _bare_for_load()
    with pytest.raises(RuntimeError, match="native backend exploded"):
        clip._load_model()


def test_unsupported_model_raises_clear_error():
    """A model with no MLX backend (None in MODEL_MAP) must raise a clear
    'no MLX backend' error, not silently serve the wrong/default model."""
    clip = _bare_for_load(UNSUPPORTED_NAME)
    with pytest.raises(RuntimeError) as ei:
        clip._load_model()
    msg = str(ei.value)
    assert UNSUPPORTED_NAME in msg, "error must name the unsupported model"
    assert "MLX backend" in msg
    assert "open_clip" in msg, "error must explain why it is unsupported now"
    # And it must not appear in the documented supported-list helper.
    assert UNSUPPORTED_NAME not in clip_module._supported_model_names()


def test_no_fallback_machinery():
    """No fallback backend machinery exists (no dead attrs/helpers)."""
    for attr in ("_load_fallback", "_encode_image_fallback", "_encode_text_fallback"):
        assert not hasattr(MLXClip, attr), f"{attr} must not exist"
    for name in (
        "OPENCLIP_MAP",
        "resolve_fallback_arch",
        "_allow_openclip_fallback",
        "_l2_normalize_torch",
    ):
        assert not hasattr(clip_module, name), f"{name} must not exist"


def _capture_mlx_clip(monkeypatch):
    """Patch mlx_clip.mlx_clip and return a dict capturing its call args.

    mlx_clip(model_dir, hf_repo): we assert on hf_repo (the checkpoint actually
    converted), since the model_dir is just a local cache path.
    """
    import mlx_clip as mlx_clip_module

    seen = {}

    def fake_mlx_clip(model_dir, hf_repo=None):
        seen["model_dir"] = model_dir
        seen["hf_repo"] = hf_repo
        return object()

    monkeypatch.setattr(mlx_clip_module, "mlx_clip", fake_mlx_clip)
    return seen


def test_unknown_model_raises_clear_error(monkeypatch):
    """An unmapped name must raise a clear 'no MLX backend' error, not
    silently degrade to the mlx_clip default (ViT-B-32) — a wrong, index-incompatible
    vector. Parity with the None-backend branch. mlx_clip must never be invoked."""
    seen = _capture_mlx_clip(monkeypatch)

    unknown = "Totally-Unknown-Model"
    clip = _bare_for_load(unknown)
    with pytest.raises(RuntimeError) as ei:
        clip._load_model()
    msg = str(ei.value)
    assert unknown in msg, "error must name the unknown model"
    assert "no MLX backend" in msg
    assert not seen, "must not have attempted the default mlx_clip load"


def test_explicit_default_still_loads(monkeypatch):
    """MODEL_MAP['default'] stays reachable for internal/test use via an explicit
    'default' request — only *unmapped* names raise. The default
    checkpoint reaches mlx_clip as hf_repo, not as the model_dir."""
    seen = _capture_mlx_clip(monkeypatch)

    clip = _bare_for_load("default")
    clip._load_model()

    assert seen["hf_repo"] == clip_module.MODEL_MAP["default"]
    assert seen["model_dir"] != clip_module.MODEL_MAP["default"]
    assert clip._loaded is True


# --- Wrong-weights regression guard ------------------------------------------
#
# mlx_clip(model_dir) treats model_dir as a LOCAL dir and, if absent, converts
# its DEFAULT hf_repo (openai/clip-vit-base-patch32). The old code passed the
# repo id positionally as model_dir with no hf_repo, so B-16/L-14/LAION all
# silently served OpenAI B-32 weights. These tests pin the corrected routing.

# OpenAI models mlx_clip can serve faithfully -> the CORRECT hf_repo is converted.
WRONG_WEIGHTS_OPENAI = {
    "ViT-B-16__openai": "openai/clip-vit-base-patch16",
    "ViT-L-14__openai": "openai/clip-vit-large-patch14",
}
# LAION models mlx_clip CANNOT serve (it hardcodes quick_gelu) -> must fail loud.
LAION_NAMES = ("ViT-B-32__laion2b-s34b-b79k", "ViT-B-32__laion2b_s34b_b79k")


@pytest.mark.parametrize("name,expected_repo", WRONG_WEIGHTS_OPENAI.items())
def test_openai_variants_convert_correct_weights(monkeypatch, name, expected_repo):
    """OpenAI B-16/L-14 must convert their OWN checkpoint, never the default B-32.

    Guards the wrong-weights bug: the requested hf_repo must be the model-specific
    OpenAI repo, and it must reach mlx_clip as hf_repo (not as the model_dir, the
    arg that silently fell back to the default checkpoint)."""
    seen = _capture_mlx_clip(monkeypatch)

    clip = _bare_for_load(name)
    clip._load_model()

    assert seen["hf_repo"] == expected_repo, f"{name} must convert {expected_repo}"
    assert seen["hf_repo"] != "openai/clip-vit-base-patch32" or name == "ViT-B-32__openai"
    # The repo id must NOT be passed as the model_dir (the original bug shape).
    assert seen["model_dir"] != expected_repo
    assert clip._loaded is True


@pytest.mark.parametrize("name", LAION_NAMES)
def test_laion_variants_fail_loud(monkeypatch, name):
    """LAION names must raise (no parity-faithful backend) rather than silently
    serve OpenAI B-32 — mlx_clip must never be invoked for them."""
    called = _capture_mlx_clip(monkeypatch)

    clip = _bare_for_load(name)
    with pytest.raises(RuntimeError) as ei:
        clip._load_model()

    msg = str(ei.value)
    assert name in msg
    assert "LAION" in msg or "gelu" in msg.lower(), "error should explain the LAION/gelu reason"
    assert called == {}, "mlx_clip must not be called for a LAION model"
    assert name not in clip_module._supported_model_names()


# --- Load-time checkpoint guard ----------------------------------------------
#
# The tests above pin the hf_repo we *request*. This block guards the other
# half: that the checkpoint mlx_clip *actually loaded* matches it. mlx_clip is a
# small third-party port (harperreed) whose ctor footgun — an absent cache dir
# converts the DEFAULT openai/clip-vit-base-patch32 regardless of hf_repo
# — could be reintroduced by a version bump even though we now pass
# hf_repo correctly. _load_model verifies the loaded vision tower against the
# arch the repo id encodes (patch size + base/large width) and fails loud on
# mismatch so wrong weights can't silently poison the smart-search index.

import types


def _fake_loaded_clip(patch_size, hidden_size=768):
    """An mlx_clip-shaped stub exposing model.config.vision_config.{patch_size,hidden_size}."""
    vision = types.SimpleNamespace(patch_size=patch_size, hidden_size=hidden_size)
    config = types.SimpleNamespace(vision_config=vision)
    return types.SimpleNamespace(model=types.SimpleNamespace(config=config))


def _patch_mlx_clip_returning(monkeypatch, model):
    """Patch mlx_clip.mlx_clip to return `model`; return a dict capturing args."""
    import mlx_clip as mlx_clip_module

    seen = {}

    def fake_mlx_clip(model_dir, hf_repo=None):
        seen["model_dir"] = model_dir
        seen["hf_repo"] = hf_repo
        return model

    monkeypatch.setattr(mlx_clip_module, "mlx_clip", fake_mlx_clip)
    return seen


# (requested name, hf_repo, the patch size that checkpoint MUST report)
_CORRECT_ARCH = {
    "ViT-B-32__openai": ("openai/clip-vit-base-patch32", 32, 768),
    "ViT-B-16__openai": ("openai/clip-vit-base-patch16", 16, 768),
    "ViT-L-14__openai": ("openai/clip-vit-large-patch14", 14, 1024),
}


@pytest.mark.parametrize("name,repo,patch,hidden", [(n, r, p, h) for n, (r, p, h) in _CORRECT_ARCH.items()])
def test_loaded_checkpoint_matching_arch_loads(monkeypatch, name, repo, patch, hidden):
    """When mlx_clip returns the RIGHT arch the load completes normally."""
    _patch_mlx_clip_returning(monkeypatch, _fake_loaded_clip(patch, hidden))
    clip = _bare_for_load(name)
    clip._load_model()
    assert clip._loaded is True


def test_loaded_checkpoint_wrong_patch_size_raises(monkeypatch):
    """B-16 requested but mlx_clip hands back B-32 (patch32) weights — exactly the
    silent-wrong-weights footgun. The load must fail loud, naming the
    mismatch, not quietly serve index-incompatible embeddings."""
    # patch16 requested; loaded model reports patch32 (the default-checkpoint bug).
    _patch_mlx_clip_returning(monkeypatch, _fake_loaded_clip(patch_size=32, hidden_size=768))
    clip = _bare_for_load("ViT-B-16__openai")
    with pytest.raises(RuntimeError) as ei:
        clip._load_model()
    msg = str(ei.value)
    assert "ViT-B-16__openai" in msg
    assert "patch_size" in msg
    assert "16" in msg and "32" in msg, "error must show requested vs loaded patch size"
    assert getattr(clip, "_loaded", False) is not True, "a mismatched load must not be marked loaded"


def test_loaded_checkpoint_wrong_width_raises(monkeypatch):
    """L-14 requested but the loaded vision width is the base 768 (not large 1024):
    a different wrong checkpoint that shares no patch size — caught via hidden_size."""
    # Correct patch (14) but base width — e.g. a partial/mixed cache.
    _patch_mlx_clip_returning(monkeypatch, _fake_loaded_clip(patch_size=14, hidden_size=768))
    clip = _bare_for_load("ViT-L-14__openai")
    with pytest.raises(RuntimeError) as ei:
        clip._load_model()
    msg = str(ei.value)
    assert "ViT-L-14__openai" in msg
    assert "hidden_size" in msg
    assert getattr(clip, "_loaded", False) is not True


def test_loaded_checkpoint_guard_skips_when_unintrospectable(monkeypatch, caplog):
    """If a future mlx_clip restructures so the vision config can't be found, the
    guard must NOT break a working load — it warns (so the guard gets maintained)
    rather than crashing on a model it can't inspect."""
    _patch_mlx_clip_returning(monkeypatch, object())  # no .model.config.vision_config
    clip = _bare_for_load("ViT-B-16__openai")
    import logging

    with caplog.at_level(logging.WARNING):
        clip._load_model()
    assert clip._loaded is True
    assert any("guard" in r.message.lower() for r in caplog.records), "should warn it could not verify"


# --- SigLIP2 tokenizer source resolution -------------------------------------


def test_tokenizer_json_from_local_dir(tmp_path):
    """A local cache/override dir supplies its own copied-in tokenizer.json."""
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    assert clip_module._resolve_siglip2_tokenizer_json(str(tmp_path)) == str(tok)


def test_tokenizer_json_from_override_repo_not_default(monkeypatch):
    """A non-dir override (custom/quantized HF repo-id) must fetch tokenizer.json
    from THAT repo, not the default SigLIP2 repo — else weights and tokenizer
    mismatch and query embeddings silently diverge from the index."""
    calls = []

    def fake_download(repo_id, filename, *args, **kwargs):
        calls.append((repo_id, filename))
        return f"/fake/{repo_id}/{filename}"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    override_repo = "someuser/siglip2-so400m-patch16-384-custom"
    result = clip_module._resolve_siglip2_tokenizer_json(override_repo)

    assert calls == [(override_repo, "tokenizer.json")], f"tokenizer must come from the override repo, got {calls!r}"
    assert result == f"/fake/{override_repo}/tokenizer.json"


def test_siglip2_routes_to_native_backend_not_mlx_clip():
    """SigLIP2 names handled natively must be None in MODEL_MAP so they never
    route to the mlx_clip path; they're dispatched via MLX_EMBEDDINGS_MAP."""
    for name in MLX_EMBEDDINGS_MAP:
        assert MODEL_MAP.get(name) is None, f"{name} must be None in MODEL_MAP to avoid the mlx_clip path"


def test_model_map_default_present():
    """A 'default' fallback repo must exist for unknown model names."""
    assert MODEL_MAP.get("default")


# --- get_clip_model routing / caching ----------------------------------------


class _StubClip:
    """Records the name it was constructed with; tracks unload()."""

    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.unloaded = False
        _StubClip.instances.append(self)

    def unload(self):
        self.unloaded = True


@pytest.fixture
def stub_clip(monkeypatch):
    """Replace MLXClip with a no-load stub and reset the module-global cache."""
    _StubClip.instances = []
    monkeypatch.setattr(clip_module, "MLXClip", _StubClip)
    monkeypatch.setattr(clip_module, "_current_model", None)
    monkeypatch.setattr(clip_module, "_current_model_name", None)
    yield _StubClip


def test_get_clip_model_normalizes_double_colon(stub_clip):
    model = get_clip_model("ViT-B-32::openai")
    assert model.model_name == "ViT-B-32__openai"


def test_get_clip_model_caches_same_name(stub_clip):
    first = get_clip_model(SIGLIP2_NAME)
    second = get_clip_model(SIGLIP2_NAME)
    assert first is second, "same name should return the cached instance"
    assert len(stub_clip.instances) == 1, "no reload expected for same name"
    # stub_clip patches get_clip_model to return _StubClip (has .unloaded)
    assert not first.unloaded  # pyright: ignore[reportAttributeAccessIssue]


def test_get_clip_model_switches_and_unloads(stub_clip):
    first = get_clip_model("ViT-B-32__openai")
    second = get_clip_model(SIGLIP2_NAME)
    assert first is not second
    assert first.unloaded, "old model must be unloaded on switch"  # pyright: ignore[reportAttributeAccessIssue]
    assert second.model_name == SIGLIP2_NAME
    assert len(stub_clip.instances) == 2


def test_get_loaded_clip_model_name_tracks_current(stub_clip):
    from src.models.clip import get_loaded_clip_model_name

    assert get_loaded_clip_model_name() is None, "no model loaded yet"
    get_clip_model(SIGLIP2_NAME)
    assert get_loaded_clip_model_name() == SIGLIP2_NAME


# --- Optional end-to-end integration against a running ML service -------------
#
# Opt in by pointing ML_SERVICE_URL at a reachable instance, e.g.:
#   ML_SERVICE_URL=http://localhost:3003 .venv/bin/python -m pytest \
#       tests/test_clip_backend.py -k integration
# Skipped (not failed) when unset or unreachable so the unit suite stays
# hermetic and CI-friendly.

_SERVICE_URL = os.getenv("ML_SERVICE_URL", "").rstrip("/")


def _service_reachable(url: str) -> bool:
    if not url:
        return False
    try:
        import requests

        return requests.get(f"{url}/ping", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _service_reachable(_SERVICE_URL),
    reason="ML_SERVICE_URL not set or service unreachable",
)
def test_predict_clip_text_integration():
    """Text smartSearch against the live service returns a 1152-d unit vector."""
    import json

    import requests

    entries = json.dumps({"clip": {"textual": {"modelName": SIGLIP2_NAME}}})
    resp = requests.post(
        f"{_SERVICE_URL}/predict",
        data={"entries": entries, "text": "a photo of a cat"},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "clip" in body, body
    emb = np.asarray(json.loads(body["clip"]), dtype=np.float32)
    assert emb.shape == (SIGLIP2_DIM,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-3)
