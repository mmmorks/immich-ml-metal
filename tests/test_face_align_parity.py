"""Parity guard for the face-alignment migration (ml-9js).

``face_embed._norm_crop`` replaces ``insightface.utils.face_align.norm_crop``,
which internally calls skimage's deprecated ``SimilarityTransform.estimate``
(removed in scikit-image 2.2). The replacement uses the current
``SimilarityTransform.from_estimate`` constructor. These tests assert the
replacement is *bit-for-bit identical* to insightface on valid landmarks, so the
migration did not move the production face index (ml-7j8.13 decided PRESERVE).
"""

import warnings

import numpy as np
import pytest

from src.models.face_embed import ARCFACE_INPUT_SIZE, _norm_crop


def _insightface_norm_crop(img, kps, image_size):
    """Reference alignment via insightface's deprecated path.

    insightface calls skimage's deprecated estimate(); suppress that FutureWarning
    locally so it never reaches the global pytest warnings summary (the whole point
    of ml-9js is that our runtime no longer triggers it)."""
    from insightface.utils import face_align

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return face_align.norm_crop(img, kps, image_size=image_size)


@pytest.mark.parametrize("seed", range(20))
def test_norm_crop_bit_identical_to_insightface(seed):
    """Our from_estimate alignment == insightface's estimate alignment, exactly."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    # Plausible 5-point face landmarks somewhere inside the frame.
    kps = rng.uniform(50, 400, size=(5, 2)).astype(np.float32)

    ours = _norm_crop(img, kps, ARCFACE_INPUT_SIZE)
    theirs = _insightface_norm_crop(img, kps, ARCFACE_INPUT_SIZE)

    assert ours.shape == theirs.shape == (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE, 3)
    assert np.array_equal(ours, theirs), f"alignment drifted from insightface (seed={seed})"


def test_norm_crop_rejects_wrong_shape():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _norm_crop(img, np.array([[1.0, 2.0]], dtype=np.float32), ARCFACE_INPUT_SIZE)


def test_norm_crop_raises_on_degenerate_landmarks():
    """All-identical points are too degenerate to estimate -> ValueError, so the
    caller skips the face (ml-6o9) instead of insightface's silent NaN-matrix crop."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kps = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        _norm_crop(img, kps, ARCFACE_INPUT_SIZE)
