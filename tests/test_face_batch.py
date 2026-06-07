"""Tests for batched face embedding — alignment, batching, edge cases.

These tests mock the ONNX model so they run without real weights.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.face_embed import get_face_embeddings_batch


def _fake_img(w=640, h=480):
    """Create a fake BGR image array."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _face_with_bbox(x1, y1, x2, y2, score=0.9):
    return {
        "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "score": score,
    }


def _face_with_landmarks(landmarks, score=0.9):
    return {
        "boundingBox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        "landmarks": landmarks,
        "score": score,
    }


# A valid-ish 5-point set (left_eye, right_eye, nose, left_mouth, right_mouth).
# Any 5 points produce a norm_crop affine warp, so these need not be a real face.
_LANDMARKS = [[120.0, 130.0], [180.0, 130.0], [150.0, 160.0], [125.0, 185.0], [175.0, 185.0]]


def _landmarks_at(dx=0.0, dy=0.0):
    """A distinct 5-point set, offset so multi-face tests have different faces."""
    return [[x + dx, y + dy] for x, y in _LANDMARKS]


def _aligned_face(dx=0.0, dy=0.0, score=0.9):
    return _face_with_landmarks(_landmarks_at(dx, dy), score=score)


@pytest.fixture
def mock_model():
    """Mock the recognition model to return fake 512-dim embeddings."""
    model = MagicMock()

    def fake_get_feat(imgs):
        n = len(imgs) if isinstance(imgs, list) else imgs.shape[0]
        return np.random.randn(n, 512).astype(np.float32)

    model.get_feat = fake_get_feat
    return model


# --- Core batching ---


def test_empty_faces():
    assert get_face_embeddings_batch(_fake_img(), [], "buffalo_l") == []


def test_single_face(mock_model):
    img = _fake_img()
    faces = [_aligned_face()]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].shape == (512,)
    assert abs(np.linalg.norm(results[0]) - 1.0) < 1e-5  # normalized


def test_multiple_faces(mock_model):
    img = _fake_img()
    faces = [_aligned_face(), _aligned_face(dx=40), _aligned_face(dx=80, dy=20)]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 3
    assert all(r is not None and r.shape == (512,) for r in results)


def test_result_order_preserved(mock_model):
    """Results must be in the same order as input faces."""
    img = _fake_img()
    faces = [_aligned_face(), _aligned_face(dx=60, dy=30)]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 2
    assert results[0] is not None and results[1] is not None
    # Each face should get a different embedding (random, so extremely unlikely to match)
    assert not np.array_equal(results[0], results[1])


# --- Edge cases ---


def test_alignment_failure_returns_none(mock_model):
    """A face whose landmarks can't be aligned returns None, not crash."""
    img = _fake_img()
    faces = [_face_with_landmarks([[1.0, 2.0]])]  # wrong shape -> norm_crop raises
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is None


def test_mixed_success_and_failure(mock_model):
    """Good faces get embeddings; a face with bad landmarks gets None."""
    img = _fake_img()
    faces = [
        _aligned_face(),  # valid
        _face_with_landmarks([[1.0, 2.0]]),  # malformed -> None
        _aligned_face(dx=70),  # valid
    ]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 3
    assert results[0] is not None  # valid
    assert results[1] is None  # malformed landmarks
    assert results[2] is not None  # valid


def test_all_faces_fail():
    """All faces unembeddable — returns list of Nones, no crash, no inference."""
    img = _fake_img(w=10, h=10)
    faces = [
        _face_with_bbox(0, 0, 0, 0),  # no landmarks -> skipped
        _face_with_landmarks([[1.0, 2.0]]),  # malformed -> alignment fails
    ]
    # Don't even need to mock the model — should never reach inference
    results = get_face_embeddings_batch(img, faces)
    assert len(results) == 2
    assert all(r is None for r in results)


# --- Landmark-miss handling (ml-6o9) ---


def test_face_without_landmarks_is_skipped(mock_model, caplog):
    """A face lacking 'landmarks' must NOT be bbox-cropped into the
    landmark-aligned index — it is skipped (None) and warned about."""
    import logging

    img = _fake_img()
    faces = [_face_with_bbox(100, 100, 200, 200)]  # no landmarks
    with caplog.at_level(logging.WARNING, logger="src.models.face_embed"), patch(
        "src.models.face_embed.get_recognition_model", return_value=mock_model
    ):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is None
    assert any("landmark" in r.message.lower() and r.levelno == logging.WARNING for r in caplog.records)


def test_landmark_face_still_embedded(mock_model):
    """Faces WITH landmarks are still embedded via the aligned path."""
    img = _fake_img()
    faces = [_face_with_landmarks(_LANDMARKS)]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].shape == (512,)
    assert abs(np.linalg.norm(results[0]) - 1.0) < 1e-5


def test_mixed_landmark_and_landmarkless(mock_model):
    """Landmarked face embedded; landmark-less face skipped — order preserved."""
    img = _fake_img()
    faces = [
        _face_with_landmarks(_LANDMARKS),
        _face_with_bbox(300, 100, 500, 300),  # no landmarks
    ]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None


# --- Embedding normalization ---


def test_embeddings_are_unit_normalized(mock_model):
    img = _fake_img()
    faces = [_aligned_face()]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert results[0] is not None
    norm = np.linalg.norm(results[0])
    assert abs(norm - 1.0) < 1e-5
