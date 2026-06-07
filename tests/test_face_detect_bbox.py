"""Tests for Vision face bounding-box pixel conversion + clamping.

Vision can return observations whose box extends past the image edges
(negative or out-of-range coords). The pixel converter must clamp to the
image bounds while preserving x2>=x1 and y2>=y1.
"""

from src.models.face_detect import _vision_bbox_to_pixels


def test_in_frame_box_unchanged():
    # A centered box well inside a 640x480 image, Vision bottom-left origin.
    box = _vision_bbox_to_pixels(0.25, 0.25, 0.5, 0.5, 640, 480)
    # x: 0.25*640=160 .. 0.75*640=480
    # y (flipped): (1-0.25-0.5)*480=120 .. (1-0.25)*480=360
    assert box == {"x1": 160, "y1": 120, "x2": 480, "y2": 360}


def test_out_of_frame_box_is_clamped():
    # Box pushed off the top-right: origin past 1.0 and extending beyond edges.
    box = _vision_bbox_to_pixels(0.8, 0.8, 0.5, 0.5, 640, 480)
    # Raw x2 = 1.3*640 = 832 (> 640); raw y1 = (1-0.8-0.5)*480 = -144 (< 0)
    assert 0 <= box["x1"] <= 640
    assert 0 <= box["x2"] <= 640
    assert 0 <= box["y1"] <= 480
    assert 0 <= box["y2"] <= 480
    # Ordering preserved after clamping.
    assert box["x2"] >= box["x1"]
    assert box["y2"] >= box["y1"]
    # Specifically: right edge clamps to width, top edge clamps to 0.
    assert box["x2"] == 640
    assert box["y1"] == 0


def test_fully_off_frame_box_collapses_in_bounds():
    # A box entirely to the right of the image: both x endpoints > width.
    box = _vision_bbox_to_pixels(1.5, 0.25, 0.3, 0.5, 640, 480)
    assert box["x1"] == 640
    assert box["x2"] == 640
    assert box["x2"] >= box["x1"]
    assert 0 <= box["y1"] <= box["y2"] <= 480


def test_negative_origin_clamped_to_zero():
    # Vision can report a small negative origin near an edge.
    box = _vision_bbox_to_pixels(-0.1, -0.1, 0.3, 0.3, 100, 100)
    assert box["x1"] == 0
    assert box["y1"] >= 0
    assert box["x2"] >= box["x1"]
    assert box["y2"] >= box["y1"]
