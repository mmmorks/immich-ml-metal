"""Concurrency tests for MLXClip — a model switch must not crash in-flight requests.

Regression coverage: get_clip_model() unloads the shared
_current_model instance when a different model is requested, setting
self._model=None on the instance an in-flight encode_*() call still holds.
The retry loops must notice the swap-to-None and raise a clean RuntimeError
instead of an AttributeError on None.img_processor / None.encode_image.
"""

import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.models.clip import MLXClip


def _red_jpeg() -> bytes:
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeMLXModel:
    """Minimal stand-in for an mlx_clip model (MLX inference path).

    The first img_processor() call blocks until ``proceed`` is set so a test
    can deterministically swap the model out while preprocessing is in flight.
    """

    def __init__(self, started=None, proceed=None):
        self._started = started
        self._proceed = proceed
        self._first = True

    def img_processor(self, images):
        if self._first and self._started is not None and self._proceed is not None:
            self._first = False
            self._started.set()
            self._proceed.wait(5)
        return "pixels"

    def model(self, **kwargs):
        class _Out:
            image_embeds = [np.ones(8, dtype=np.float32)]

        return _Out()

    def text_encoder(self, text):
        return np.ones(8, dtype=np.float32)


def _bare_clip(model):
    """Build an MLXClip without loading real weights."""
    clip = object.__new__(MLXClip)
    clip.model_name = "ViT-B-32__openai"
    clip._model = model
    clip._loaded = True
    clip._inference_lock = threading.Lock()
    clip._processor = None
    clip._tokenizer = None
    return clip


def test_encode_image_concurrent_unload_raises_clean_error():
    """The documented race: model unloaded mid-preprocess -> clean RuntimeError."""
    started = threading.Event()
    proceed = threading.Event()
    clip = _bare_clip(_FakeMLXModel(started, proceed))

    result = {}

    def worker():
        try:
            result["value"] = clip.encode_image(_red_jpeg())
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=worker)
    t.start()

    assert started.wait(5), "encode_image never began preprocessing"
    # Simulate a concurrent get_clip_model() switching to a different model,
    # which calls unload() on the instance this request still holds.
    clip.unload()
    proceed.set()
    t.join(10)
    assert not t.is_alive(), "worker thread hung"

    assert "value" not in result, f"expected failure, got embedding {result.get('value')!r}"
    err = result.get("error")
    assert isinstance(err, RuntimeError), f"expected RuntimeError, got {err!r}"
    assert not isinstance(err, AttributeError), "swap-to-None leaked an AttributeError"


def test_unload_serializes_against_inflight_inference():
    """unload() must hold the inference lock around its teardown.

    A concurrent model switch calls unload(), which frees the MLX buffer pool
    via clear_cache(). If that runs while another thread is mid-eval under the
    same metal_lock, the freed pool collides with the in-flight Metal work and
    crashes the process. So unload() must block until the lock is free.
    """
    clip = _bare_clip(_FakeMLXModel())
    lock = clip._inference_lock
    unload_done = threading.Event()

    # Stand in for an in-flight Metal eval holding the lock.
    lock.acquire()

    def unloader():
        clip.unload()
        unload_done.set()

    t = threading.Thread(target=unloader)
    t.start()
    try:
        # unload() must wait on the held lock — it cannot finish yet.
        assert not unload_done.wait(0.5), "unload() did not wait for the inference lock"
        lock.release()
        assert unload_done.wait(5), "unload() never completed after the lock was released"
    finally:
        if lock.locked():
            lock.release()
        t.join(5)
    assert not t.is_alive(), "unloader thread hung"


def test_encode_image_unloaded_midflight_no_attributeerror():
    """Per-iteration guard: _model=None while _loaded stayed True -> RuntimeError."""
    clip = _bare_clip(_FakeMLXModel())
    clip._model = None  # mid-flight null, top-of-method check already passed
    with pytest.raises(RuntimeError):
        clip.encode_image(_red_jpeg())


def test_encode_text_unloaded_midflight_no_attributeerror():
    clip = _bare_clip(_FakeMLXModel())
    clip._model = None
    with pytest.raises(RuntimeError):
        clip.encode_text("a photo of a cat")


# --- Native SigLIP2 backend (mlx-embeddings) ---------------------------------
#
# These guard the metal_lock contract for the new backend: MLX work is lazy, so
# get_*_features() output MUST be materialized (np.array) *inside* the inference
# lock — otherwise un-evaluated Metal buffers can collide with a concurrent
# Vision (face/OCR) call and crash the process. They also cover the same
# swap-to-None race as the mlx_clip path above. No torch, no real weights,
# no PyObjC needed.


class _FakeSiglip2Processor:
    """Stand-in for the SiglipProcessor returned by mlx_embeddings.load().

    Callable with images=... or text=...; returns the dict key clip.py reads.
    The first call optionally blocks until ``proceed`` so a test can swap the
    model out while preprocessing is in flight (it runs outside the lock).
    """

    def __init__(self, started=None, proceed=None):
        self._started = started
        self._proceed = proceed
        self._first = True

    def __call__(self, images=None, text=None, **kwargs):
        if self._first and self._started is not None and self._proceed is not None:
            self._first = False
            self._started.set()
            self._proceed.wait(5)
        if images is not None:
            return {"pixel_values": "pixels"}
        return {"input_ids": "ids"}


class _FakeSiglip2Model:
    """Stand-in for an mlx-embeddings SigLIP2 model.

    get_image_features / get_text_features return an object whose [0] yields a
    probe; converting that probe with np.array() (as clip.py does to force eval)
    records whether ``lock`` was held at materialization time.
    """

    def __init__(self, eval_record=None, lock=None):
        self._eval_record = eval_record
        self._lock = lock

    def _features(self):
        record, lock = self._eval_record, self._lock

        class _Probe:
            def __array__(self, dtype=None, copy=None):
                if record is not None and lock is not None:
                    record.append(lock.locked())
                arr = np.ones(1152, dtype=np.float32)
                return arr.astype(dtype) if dtype is not None else arr

        class _Feats:
            def __getitem__(self, idx):
                return _Probe()

        return _Feats()

    def get_image_features(self, pixel_values=None):
        return self._features()

    def get_text_features(self, input_ids=None):
        return self._features()


def _bare_siglip2(model, processor):
    """Build a SigLIP2-backed MLXClip without loading real weights.

    The SigLIP2 encode paths preprocess via
    src.models.immich_preprocess (siglip_image_pixels + a SiglipTextTokenizer),
    NOT the SiglipProcessor — so the image path needs no processor and the text
    path uses a callable tokenizer returning (1, ctx) int32 ids.
    """
    clip = object.__new__(MLXClip)
    clip.model_name = "ViT-SO400M-16-SigLIP2-384__webli"
    clip._model = model
    clip._processor = processor
    clip._tokenizer = None
    clip._siglip_tokenizer = lambda text: np.zeros((1, 64), dtype=np.int32)
    clip._loaded = True
    clip._inference_lock = threading.Lock()
    clip._use_mlx_embeddings = True
    return clip


def test_encode_image_siglip2_unloaded_midflight_no_attributeerror():
    clip = _bare_siglip2(_FakeSiglip2Model(), _FakeSiglip2Processor())
    clip._model = None  # mid-flight null, top-of-method check already passed
    with pytest.raises(RuntimeError):
        clip.encode_image(_red_jpeg())


def test_encode_text_siglip2_unloaded_midflight_no_attributeerror():
    clip = _bare_siglip2(_FakeSiglip2Model(), _FakeSiglip2Processor())
    clip._model = None
    with pytest.raises(RuntimeError):
        clip.encode_text("a photo of a cat")


def test_encode_image_siglip2_concurrent_unload_raises_clean_error(monkeypatch):
    """Model unloaded while preprocessing is in flight -> clean RuntimeError.

    Image preprocessing (siglip_image_pixels) runs outside the lock and is
    model-independent; block inside it so the model can be unloaded before the
    encode loop captures self._model. The swap-to-None must surface as a clean
    RuntimeError, never an AttributeError.
    """
    import src.models.clip as clip_module

    started = threading.Event()
    proceed = threading.Event()
    real_pixels = clip_module.siglip_image_pixels

    def blocking_pixels(image):
        started.set()
        proceed.wait(5)
        return real_pixels(image)

    monkeypatch.setattr(clip_module, "siglip_image_pixels", blocking_pixels)
    clip = _bare_siglip2(_FakeSiglip2Model(), _FakeSiglip2Processor())

    result = {}

    def worker():
        try:
            result["value"] = clip.encode_image(_red_jpeg())
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=worker)
    t.start()

    assert started.wait(5), "encode_image never began preprocessing"
    clip.unload()  # concurrent model switch nulls self._model
    proceed.set()
    t.join(10)
    assert not t.is_alive(), "worker thread hung"

    assert "value" not in result, f"expected failure, got {result.get('value')!r}"
    err = result.get("error")
    assert isinstance(err, RuntimeError), f"expected RuntimeError, got {err!r}"
    assert not isinstance(err, AttributeError), "swap-to-None leaked an AttributeError"


def test_encode_image_siglip2_forces_eval_inside_lock():
    """get_image_features output must be materialized while the lock is held."""
    record = []
    clip = _bare_siglip2(None, _FakeSiglip2Processor())
    clip._model = _FakeSiglip2Model(eval_record=record, lock=clip._inference_lock)

    emb = clip.encode_image(_red_jpeg())

    assert emb.shape == (1152,) and emb.dtype == np.float32
    assert record == [True], f"Metal eval must occur inside the lock, got {record}"


def test_encode_text_siglip2_forces_eval_inside_lock():
    """get_text_features output must be materialized while the lock is held."""
    record = []
    clip = _bare_siglip2(None, _FakeSiglip2Processor())
    clip._model = _FakeSiglip2Model(eval_record=record, lock=clip._inference_lock)

    emb = clip.encode_text("a photo of a cat")

    assert emb.shape == (1152,) and emb.dtype == np.float32
    assert record == [True], f"Metal eval must occur inside the lock, got {record}"


# --- Concurrency determinism: N concurrent encodes == serial, per input --------
#
# The crash-safety tests above prove a mid-flight swap fails cleanly. These prove
# the everyday case: many encodes running at once must each return the SAME vector
# they would have serially — no request's preprocessed data leaking into another's
# result. The fakes below make each embedding a deterministic function of the
# input, so cross-talk shows up as a mismatched vector rather than a crash.


def _solid_png(i: int) -> bytes:
    """A distinct solid-colour PNG (lossless, so the fingerprint is exact)."""
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color=(20 * i + 5, 100, 150)).save(buf, format="PNG")
    return buf.getvalue()


def _char_ids(text: str, ctx: int = 64) -> np.ndarray:
    """Tokenizer stand-in: map text to distinct (1, ctx) int32 ids per string."""
    ids = [ord(c) for c in text[:ctx]]
    ids += [0] * (ctx - len(ids))
    return np.array([ids], dtype=np.int32)


class _DetMlxClipModel:
    """mlx_clip-path fake whose output is a deterministic function of the input."""

    def img_processor(self, images):
        return float(np.asarray(images[0], dtype=np.float64).mean())

    def tokenizer(self, texts):
        return float(sum(ord(c) for c in texts[0]))

    def model(self, pixel_values=None, input_ids=None):
        fp = pixel_values if pixel_values is not None else input_ids
        vec = np.array([fp, 1.0, 2.0, 3.0], dtype=np.float32)
        return SimpleNamespace(image_embeds=[vec], text_embeds=[vec])


class _DetSiglip2Model:
    """SigLIP2-path fake: embedding derived from the real preprocessed input."""

    def get_image_features(self, pixel_values=None):
        fp = float(np.array(pixel_values).mean())
        return [np.array([fp, 1.0, 2.0, 3.0], dtype=np.float32)]

    def get_text_features(self, input_ids=None):
        fp = float(np.array(input_ids).sum())
        return [np.array([fp, 1.0, 2.0, 3.0], dtype=np.float32)]


def _assert_concurrent_matches_serial(encode, inputs):
    """Each concurrent encode must equal its serial baseline (per-input
    determinism). First assert distinct inputs yield distinct embeddings, so a
    constant-output regression can't make the determinism check vacuous."""
    serial = [encode(x) for x in inputs]
    for i in range(len(serial)):
        for j in range(i + 1, len(serial)):
            assert not np.array_equal(serial[i], serial[j]), "inputs not distinguishable — determinism check would be vacuous"

    results: dict[int, np.ndarray] = {}
    errors = []
    barrier = threading.Barrier(len(inputs))

    def worker(idx):
        try:
            barrier.wait(5)  # line threads up so the out-of-lock preprocessing overlaps
            results[idx] = encode(inputs[idx])
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(inputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert not any(t.is_alive() for t in threads), "a worker thread hung"
    assert not errors, f"concurrent encode raised: {errors!r}"
    for i in range(len(inputs)):
        np.testing.assert_array_equal(results[i], serial[i], err_msg=f"input {i} got cross-talk under concurrency")


def test_encode_image_mlx_clip_concurrent_matches_serial():
    clip = _bare_clip(_DetMlxClipModel())
    _assert_concurrent_matches_serial(clip.encode_image, [_solid_png(i) for i in range(8)])


def test_encode_image_siglip2_concurrent_matches_serial():
    """The production default path: real siglip_image_pixels preprocessing runs
    concurrently outside the lock, then Metal eval is serialized."""
    clip = _bare_siglip2(_DetSiglip2Model(), _FakeSiglip2Processor())
    _assert_concurrent_matches_serial(clip.encode_image, [_solid_png(i) for i in range(8)])


def test_encode_text_siglip2_concurrent_matches_serial():
    clip = _bare_siglip2(_DetSiglip2Model(), _FakeSiglip2Processor())
    clip._siglip_tokenizer = _char_ids  # distinct ids per text so outputs differ
    _assert_concurrent_matches_serial(clip.encode_text, [f"a caption number {i}" for i in range(8)])
