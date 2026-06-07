"""Concurrency tests for MLXClip — a model switch must not crash in-flight requests.

Regression coverage for ml-7j8.1: get_clip_model() unloads the shared
_current_model instance when a different model is requested, setting
self._model=None on the instance an in-flight encode_*() call still holds.
The retry loops must notice the swap-to-None and raise a clean RuntimeError
instead of an AttributeError on None.img_processor / None.encode_image.
"""
import io
import threading

import numpy as np
import pytest
import torch
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
        if self._first and self._started is not None:
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


def _bare_clip(model, *, fallback=False):
    """Build an MLXClip without loading real weights."""
    clip = object.__new__(MLXClip)
    clip.model_name = "ViT-B-32__openai"
    clip._model = model
    clip._processor = (lambda img: torch.zeros(3, 4, 4)) if fallback else None
    clip._tokenizer = (lambda texts: torch.zeros(1, 4, dtype=torch.long)) if fallback else None
    clip._loaded = True
    clip._inference_lock = threading.Lock()
    if fallback:
        clip._use_fallback = True
        clip._device = torch.device("cpu")
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
        except BaseException as e:  # noqa: BLE001 - capture whatever escapes
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


def test_encode_image_fallback_unloaded_midflight_no_attributeerror():
    clip = _bare_clip(object(), fallback=True)
    clip._model = None
    with pytest.raises(RuntimeError):
        clip.encode_image(_red_jpeg())


def test_encode_text_fallback_unloaded_midflight_no_attributeerror():
    clip = _bare_clip(object(), fallback=True)
    clip._model = None
    with pytest.raises(RuntimeError):
        clip.encode_text("a photo of a cat")
