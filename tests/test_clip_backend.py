"""Backend tests for MLXClip — ml-ycd.10.

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

    Post-ml-ycd.4 the SigLIP2 encode paths preprocess via
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


# --- Zero-embedding guard (ml-9g1) -------------------------------------------
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


# --- Zero-embedding guard: open_clip fallback paths (ml-3bt) ------------------
#
# The non-default open_clip fallback (_encode_image_fallback /
# _encode_text_fallback) had the identical hazard as the SigLIP2 paths above,
# but via the torch tensor norm API rather than np.linalg.norm. A zero pooled
# output normalized by its zero norm produces an all-NaN embedding that poisons
# the smart-search index/query. _l2_normalize_torch must leave it untouched.


def _bare_fallback(model):
    """Build an open_clip-fallback MLXClip without loading real weights.

    Mirrors the attributes _load_fallback sets (_use_fallback + a torch
    processor/tokenizer/device). The fake processor/tokenizer just return throw-
    away tensors — the fake model ignores its input and returns a fixed (1, D)
    pooled output, exercising only the normalization in the run() closures.
    """
    import torch

    clip = object.__new__(MLXClip)
    clip.model_name = "ViT-B-16-SigLIP2__webli"
    clip._model = model
    clip._processor = lambda image: torch.zeros(3, 2, 2)
    clip._tokenizer = lambda texts: torch.zeros(1, 4, dtype=torch.int64)
    clip._device = torch.device("cpu")
    clip._loaded = True
    clip._inference_lock = threading.Lock()
    clip._use_fallback = True
    return clip


class _FakeTorchModel:
    """open_clip stand-in: encode_image/encode_text return a fixed (1, D) tensor."""

    def __init__(self, raw):
        import torch

        self._raw = torch.tensor(np.asarray(raw, dtype=np.float32)).unsqueeze(0)

    def encode_image(self, image_tensor):
        return self._raw

    def encode_text(self, tokens):
        return self._raw


def test_fallback_image_embedding_normalized():
    pytest.importorskip("torch")
    raw = np.arange(1, 9, dtype=np.float32)  # non-unit, non-uniform
    clip = _bare_fallback(_FakeTorchModel(raw))

    emb = clip._encode_image_fallback(Image.new("RGB", (8, 8)))

    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(emb, raw / np.linalg.norm(raw), atol=1e-6)


def test_fallback_image_zero_embedding_does_not_nan():
    pytest.importorskip("torch")
    clip = _bare_fallback(_FakeTorchModel(np.zeros(8, dtype=np.float32)))

    emb = clip._encode_image_fallback(Image.new("RGB", (8, 8)))

    assert not np.isnan(emb).any(), "zero embedding must not normalize to NaN"
    assert np.all(emb == 0.0), "a zero raw output should stay zero, not become NaN"


def test_fallback_text_zero_embedding_does_not_nan():
    pytest.importorskip("torch")
    clip = _bare_fallback(_FakeTorchModel(np.zeros(8, dtype=np.float32)))

    emb = clip._encode_text_fallback("a photo of a cat")

    assert not np.isnan(emb).any(), "zero embedding must not normalize to NaN"
    assert np.all(emb == 0.0), "a zero raw output should stay zero, not become NaN"


# --- Name mapping invariants -------------------------------------------------


def test_mlx_embeddings_map_repos_have_patch_token():
    """The mlx-embeddings loader regex-parses patch/image size from the repo id
    (config.json omits patch_size), so every mapped repo MUST contain a
    'patchNN-NNN' token or load() crashes. See ml-ycd.1 spike."""
    pat = re.compile(r"patch\d+-\d+")
    assert MLX_EMBEDDINGS_MAP, "expected at least one native SigLIP2 mapping"
    for name, repo in MLX_EMBEDDINGS_MAP.items():
        assert pat.search(repo), f"{name} -> {repo!r} lacks a patchNN-NNN token"


# --- Native SigLIP2 load-failure guard (ml-7j8.11) ---------------------------
#
# The open_clip fallback serves SigLIP-*squash* embeddings (~0.83 cosine vs the
# existing index), so silently using it when the parity-verified native backend
# fails to load would poison the smart-search index with only a log line as
# signal. For the index-critical native model the load must fail LOUDLY by
# default; an explicit opt-in env var trades correctness for availability.


def _bare_for_load(name=SIGLIP2_NAME):
    """A bare MLXClip with just enough state to call _load_model()."""
    clip = object.__new__(MLXClip)
    clip.model_name = name
    return clip


def test_native_siglip2_load_failure_raises_by_default(monkeypatch):
    monkeypatch.delenv("ML_SIGLIP2_ALLOW_OPENCLIP_FALLBACK", raising=False)

    def boom(self):
        raise RuntimeError("native backend exploded")

    fell_back = []
    monkeypatch.setattr(MLXClip, "_load_siglip2_mlx", boom)
    monkeypatch.setattr(MLXClip, "_load_fallback", lambda self: fell_back.append(True))

    clip = _bare_for_load()
    with pytest.raises(RuntimeError) as ei:
        clip._load_model()

    assert not fell_back, "must NOT silently fall back to open_clip for the index-critical model"
    msg = str(ei.value)
    assert SIGLIP2_NAME in msg, "error must name the affected index-critical model"
    assert "ML_SIGLIP2_ALLOW_OPENCLIP_FALLBACK" in msg, "error must point at the opt-in escape hatch"
    # The original cause is chained for debuggability.
    assert isinstance(ei.value.__cause__, RuntimeError)


@pytest.mark.parametrize("flag", ["1", "true", "YES", "on"])
def test_native_siglip2_load_failure_opt_in_allows_fallback(monkeypatch, flag):
    monkeypatch.setenv("ML_SIGLIP2_ALLOW_OPENCLIP_FALLBACK", flag)

    def boom(self):
        raise RuntimeError("native backend exploded")

    fell_back = []
    monkeypatch.setattr(MLXClip, "_load_siglip2_mlx", boom)
    monkeypatch.setattr(MLXClip, "_load_fallback", lambda self: fell_back.append(True))

    clip = _bare_for_load()
    clip._load_model()  # must NOT raise

    assert fell_back == [True], "explicit opt-in must permit the open_clip fallback"


def test_native_siglip2_successful_load_never_falls_back(monkeypatch):
    """The happy path must not touch the fallback regardless of the env flag."""
    monkeypatch.delenv("ML_SIGLIP2_ALLOW_OPENCLIP_FALLBACK", raising=False)

    loaded = []
    fell_back = []
    monkeypatch.setattr(MLXClip, "_load_siglip2_mlx", lambda self: loaded.append(True))
    monkeypatch.setattr(MLXClip, "_load_fallback", lambda self: fell_back.append(True))

    clip = _bare_for_load()
    clip._load_model()

    assert loaded == [True]
    assert not fell_back


def test_allow_openclip_fallback_default_off(monkeypatch):
    monkeypatch.delenv("ML_SIGLIP2_ALLOW_OPENCLIP_FALLBACK", raising=False)
    assert clip_module._allow_openclip_fallback() is False


# --- SigLIP2 tokenizer source resolution (ml-qax) ----------------------------


def test_tokenizer_json_from_local_dir(tmp_path):
    """A local cache/override dir supplies its own copied-in tokenizer.json."""
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    assert clip_module._resolve_siglip2_tokenizer_json(str(tmp_path)) == str(tok)


def test_tokenizer_json_from_override_repo_not_default(monkeypatch):
    """A non-dir override (custom/quantized HF repo-id) must fetch tokenizer.json
    from THAT repo, not the default SigLIP2 repo — else weights and tokenizer
    mismatch and query embeddings silently diverge from the index (ml-qax)."""
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
