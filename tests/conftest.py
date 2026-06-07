"""Shared fixtures for immich-ml-metal tests."""

import functools
import io
import os

import pytest

# Force STUB_MODE for unit tests — no real models needed.
os.environ["STUB_MODE"] = "true"


@functools.lru_cache(maxsize=1)
def _vision_face_detection_works() -> bool:
    """Probe once whether Apple Vision face detection runs in this environment.

    The detector talks to Vision directly (STUB_MODE only stubs the /predict
    handler, not face_detect), so any test that calls detect_faces end-to-end
    needs a working Vision face path. It works on a real desktop Mac / the
    Neural Engine, but GitHub's hosted macOS runners are headless and the
    request fails there with 'Vision face request failed: ... Code=9
    Unspecified error'. We probe with a blank image and treat that specific
    RuntimeError as "unavailable" so a runner limitation doesn't masquerade as
    a code regression. Any other error is a real fault and propagates.
    """
    from PIL import Image

    import src.models.face_detect as face_detect

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buf, format="PNG")
    try:
        face_detect.detect_faces(buf.getvalue(), img_width=64, img_height=64)
    except RuntimeError as e:
        if "Vision face request failed" in str(e):
            return False
        raise
    return True


@pytest.fixture
def requires_vision_faces():
    """Skip a test unless Apple Vision face detection works here (see probe above)."""
    if not _vision_face_detection_works():
        pytest.skip("Apple Vision face detection unavailable in this environment (e.g. headless CI runner)")


@pytest.fixture
def test_image_bytes():
    """A minimal valid JPEG for testing (red 100x100)."""
    import io

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def test_image_large_bytes():
    """A larger JPEG for more realistic testing (640x480)."""
    import io

    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(128, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
