"""Tests for the Vision nose-landmark strategy (ml-eo4).

The fork reconstructs ArcFace's 5-point landmarks from Apple Vision face
*contours*. The nose anchor historically used the *last* point of the nose
contour (``"tip"``). On non-frontal/occluded poses that last contour point can
swing off the actual nose tip and drift the alignment, so ml-eo4 adds a
``"center"`` variant (the nose-contour centroid) to evaluate against it.

These cover the pure point-selection helper and that the strategy is threaded
through ``extract_five_point_landmarks`` — without touching real Vision (the
landmark/bbox objects are faked to the small ``.x``/``.y`` / ``.origin`` /
``.size`` surface the function uses).
"""

import types

import pytest

from src.models.face_detect import (
    DEFAULT_NOSE_STRATEGY,
    NOSE_STRATEGIES,
    _select_nose_point,
    extract_five_point_landmarks,
)


def _pts(*coords):
    """Vision normalizedPoints stand-ins: objects with .x/.y."""
    return [types.SimpleNamespace(x=x, y=y) for x, y in coords]


# --- pure nose-anchor selection --------------------------------------------- #


def test_tip_strategy_returns_last_contour_point():
    pts = _pts((0.1, 0.2), (0.5, 0.6), (0.9, 0.4))
    assert _select_nose_point(pts, "tip") == (0.9, 0.4)


def test_center_strategy_returns_contour_centroid():
    pts = _pts((0.1, 0.2), (0.5, 0.6), (0.9, 0.4))
    nx, ny = _select_nose_point(pts, "center")
    assert nx == pytest.approx((0.1 + 0.5 + 0.9) / 3)
    assert ny == pytest.approx((0.2 + 0.6 + 0.4) / 3)


def test_tip_and_center_differ_on_asymmetric_contour():
    # A contour whose last point is far from the centroid — the non-frontal case
    # the variant targets. The two strategies must disagree.
    pts = _pts((0.4, 0.5), (0.45, 0.55), (0.95, 0.1))
    assert _select_nose_point(pts, "tip") != _select_nose_point(pts, "center")


def test_empty_contour_returns_none():
    assert _select_nose_point([], "tip") is None
    assert _select_nose_point([], "center") is None


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="nose_strategy"):
        _select_nose_point(_pts((0.1, 0.2)), "nostril")


def test_default_strategy_is_tip_preserving_production_behavior():
    assert DEFAULT_NOSE_STRATEGY == "tip"
    assert set(NOSE_STRATEGIES) == {"tip", "center"}


# --- threading through extract_five_point_landmarks ------------------------- #


class _FakeRegion:
    def __init__(self, coords):
        self._pts = _pts(*coords)

    def pointCount(self):
        return len(self._pts)

    def normalizedPoints(self):
        return self._pts


class _FakeLandmarks:
    def __init__(self, left_eye, right_eye, nose, outer_lips):
        self._le = _FakeRegion(left_eye)
        self._re = _FakeRegion(right_eye)
        self._nose = _FakeRegion(nose)
        self._lips = _FakeRegion(outer_lips)

    def leftEye(self):
        return self._le

    def rightEye(self):
        return self._re

    def nose(self):
        return self._nose

    def outerLips(self):
        return self._lips


def _full_image_bbox():
    """A bbox spanning the whole image so normalized coords map linearly:
    landmark_to_image_coords(nx, ny) == [nx*W, (1-ny)*H]."""
    return types.SimpleNamespace(
        origin=types.SimpleNamespace(x=0.0, y=0.0),
        size=types.SimpleNamespace(width=1.0, height=1.0),
    )


# A clearly non-frontal nose contour: last point sits far from the centroid.
_NOSE_CONTOUR = [(0.40, 0.50), (0.45, 0.55), (0.95, 0.10)]
_LEFT_EYE = [(0.3, 0.7)]
_RIGHT_EYE = [(0.7, 0.7)]
_OUTER_LIPS = [(0.35, 0.2), (0.65, 0.2)]
_W, _H = 200, 100


def _nose_point(strategy):
    lm = _FakeLandmarks(_LEFT_EYE, _RIGHT_EYE, _NOSE_CONTOUR, _OUTER_LIPS)
    five = extract_five_point_landmarks(lm, _full_image_bbox(), _W, _H, nose_strategy=strategy)
    assert five is not None
    return five[2]  # nose is the 3rd of the 5 points


def test_extract_threads_tip_strategy():
    nx, ny = _NOSE_CONTOUR[-1]
    assert _nose_point("tip") == pytest.approx([nx * _W, (1 - ny) * _H])


def test_extract_threads_center_strategy():
    cx = sum(x for x, _ in _NOSE_CONTOUR) / len(_NOSE_CONTOUR)
    cy = sum(y for _, y in _NOSE_CONTOUR) / len(_NOSE_CONTOUR)
    assert _nose_point("center") == pytest.approx([cx * _W, (1 - cy) * _H])


def test_extract_default_matches_tip():
    lm = _FakeLandmarks(_LEFT_EYE, _RIGHT_EYE, _NOSE_CONTOUR, _OUTER_LIPS)
    default = extract_five_point_landmarks(lm, _full_image_bbox(), _W, _H)
    assert default[2] == pytest.approx(_nose_point("tip"))
