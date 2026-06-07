"""The /predict handler decodes the request image once and shares the result.

detect_faces and recognize_text only open the image to read its dimensions.
When the caller already knows them (the /predict handler opens the image once
for imageWidth/imageHeight), they accept the dimensions and skip the redundant
PIL open — without changing the result. These tests make a fresh ``Image.open``
fatal so any redundant decode fails loudly.
"""

import io

from PIL import Image

import src.models.face_detect as face_detect
from src.models import ocr


def _blank_image_bytes(color="white"):
    img = Image.new("RGB", (200, 80), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _no_open(mod, monkeypatch):
    """Make mod.Image.open raise, so any redundant decode fails the test."""

    def _boom(*a, **k):
        raise AssertionError("image was re-opened despite caller-provided dimensions")

    monkeypatch.setattr(mod.Image, "open", _boom)


def test_detect_faces_skips_open_when_dimensions_provided(requires_vision_faces, monkeypatch):
    _no_open(face_detect, monkeypatch)

    faces, w, h = face_detect.detect_faces(_blank_image_bytes(), img_width=200, img_height=80)

    assert faces == []
    assert (w, h) == (200, 80)


def test_recognize_text_skips_open_when_dimensions_provided(monkeypatch):
    _no_open(ocr, monkeypatch)

    result = ocr.recognize_text(_blank_image_bytes(), img_width=200, img_height=80)

    assert result == {"text": [], "box": [], "boxScore": [], "textScore": []}


def test_detect_faces_still_opens_when_dimensions_absent(requires_vision_faces):
    """Standalone callers that pass no dimensions keep working (open for size)."""
    faces, w, h = face_detect.detect_faces(_blank_image_bytes())
    assert faces == []
    assert (w, h) == (200, 80)
