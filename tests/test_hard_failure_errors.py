"""Hard decode/Vision/inference failures must surface as errors.

A hard failure (image decode error, Vision-framework error, unexpected
inference exception) must raise so the /predict request fails and Immich
records the job as failed and retries. The genuinely-empty case (an image
with no text / no faces) must still return a structurally-valid empty result
so Immich marks it processed. The two cases must be distinguishable.
"""

import io

import pytest
from PIL import Image, UnidentifiedImageError

import src.main as main
from src.models import ocr


def _blank_image_bytes(color="white"):
    img = Image.new("RGB", (200, 80), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- OCR ---


def test_ocr_raises_on_undecodable_image():
    """A hard decode failure must raise, not return an empty result that would
    mark the asset permanently processed."""
    with pytest.raises(UnidentifiedImageError):
        ocr.recognize_text(b"this is not an image")


def test_ocr_returns_empty_for_text_free_image():
    """A genuinely text-free image still returns a structurally-empty result
    without raising — the empty case stays distinguishable from a failure."""
    result = ocr.recognize_text(_blank_image_bytes())
    assert result == {"text": [], "box": [], "boxScore": [], "textScore": []}


def test_ocr_raises_on_vision_error(monkeypatch):
    """A Vision request that reports failure must raise rather than silently
    return empty text."""

    class _FakeHandler:
        def initWithData_options_(self, *a):
            return self

        def performRequests_error_(self, *a):
            return (False, "vision boom")

    class _FakeHandlerCls:
        @staticmethod
        def alloc():
            return _FakeHandler()

    monkeypatch.setattr(ocr.Vision, "VNImageRequestHandler", _FakeHandlerCls)
    with pytest.raises(RuntimeError, match="Vision OCR request failed"):
        ocr.recognize_text(_blank_image_bytes())


def test_ocr_raises_on_unexpected_exception(monkeypatch):
    """An unexpected exception inside the OCR impl must propagate, not be
    swallowed into an empty result."""

    class _BoomNSData:
        @staticmethod
        def dataWithBytes_length_(*a):
            raise RuntimeError("ns data boom")

    monkeypatch.setattr(ocr, "NSData", _BoomNSData)
    with pytest.raises(RuntimeError, match="ns data boom"):
        ocr.recognize_text(_blank_image_bytes())


# --- Face detection ---


def test_face_detect_raises_on_undecodable_image():
    """A hard decode failure in detection must raise, not return an empty face
    list that would mark the asset permanently processed."""
    import src.models.face_detect as face_detect

    with pytest.raises(ValueError, match="Invalid image data"):
        face_detect.detect_faces(b"this is not an image")


def test_face_detect_returns_empty_for_face_free_image(requires_vision_faces):
    """A genuinely face-free image still returns a structurally-empty result
    without raising — the empty case stays distinguishable from a failure.

    Exercises the real Vision face path, so it skips where that path is
    unavailable (e.g. a headless CI runner); see requires_vision_faces."""
    import src.models.face_detect as face_detect

    faces, w, h = face_detect.detect_faces(_blank_image_bytes())
    assert faces == []
    assert (w, h) == (200, 80)


def test_face_detect_raises_on_vision_error(monkeypatch):
    """A Vision request that reports failure must raise rather than silently
    return an empty face list."""
    import src.models.face_detect as face_detect

    class _FakeHandler:
        def initWithData_options_(self, *a):
            return self

        def performRequests_error_(self, *a):
            return (False, "vision boom")

    class _FakeHandlerCls:
        @staticmethod
        def alloc():
            return _FakeHandler()

    monkeypatch.setattr(face_detect.Vision, "VNImageRequestHandler", _FakeHandlerCls)
    with pytest.raises(RuntimeError, match="Vision face request failed"):
        face_detect.detect_faces(_blank_image_bytes())


def test_face_detect_raises_on_unexpected_exception(monkeypatch):
    """An unexpected exception inside the detection impl must propagate, not be
    swallowed into an empty face list."""
    import src.models.face_detect as face_detect

    class _BoomNSData:
        @staticmethod
        def dataWithBytes_length_(*a):
            raise RuntimeError("ns data boom")

    monkeypatch.setattr(face_detect, "NSData", _BoomNSData)
    with pytest.raises(RuntimeError, match="ns data boom"):
        face_detect.detect_faces(_blank_image_bytes())


# --- Face recognition ---


def test_face_raises_on_undecodable_image(monkeypatch):
    """When cv2 cannot decode the image for embedding, raise so Immich retries
    instead of recording a false 'no faces' result."""
    import src.models.face_detect as face_detect

    # detect_faces would otherwise fail first on garbage bytes; force it to
    # return a qualifying face so we reach the cv2.imdecode hard-failure path.
    fake_face = {"score": 0.99, "boundingBox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}
    monkeypatch.setattr(face_detect, "detect_faces", lambda *a, **k: ([fake_face], None, None))

    with pytest.raises(RuntimeError, match="Failed to decode image for face recognition"):
        main._run_face_recognition_sync(b"not a real image", min_score=0.0, model_name="buffalo_l")
