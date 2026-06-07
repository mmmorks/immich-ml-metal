"""Unit tests for OCR bbox -> quadrilateral conversion (normalized_bbox_to_box).

Pure-logic coverage for ml-7j8.10: the Vision (bottom-left-origin, normalized)
to Immich (top-left-origin, pixel, 8-coord clockwise quad) conversion was
previously inlined in recognize_text() and untested. Covers the Y-flip, pixel
scaling, integer truncation, and corner ordering.
"""
from src.models.ocr import normalized_bbox_to_box


def test_full_frame_box_maps_to_image_corners():
    # Whole image: origin (0,0), size (1,1). Y-flip leaves it spanning the frame.
    assert normalized_bbox_to_box(0.0, 0.0, 1.0, 1.0, 100, 200) == [
        0, 0, 100, 0, 100, 200, 0, 200,
    ]


def test_y_axis_is_flipped():
    # A Vision box at the bottom (origin_y=0) sits in the LOWER half of the
    # image (large pixel-y), proving the top-left-origin flip.
    box = normalized_bbox_to_box(0.0, 0.0, 0.5, 0.5, 100, 100)
    assert box == [0, 50, 50, 50, 50, 100, 0, 100]

    # A Vision box at the top (origin_y + height == 1.0) sits at pixel-y 0.
    top = normalized_bbox_to_box(0.0, 0.5, 0.5, 0.5, 100, 100)
    assert top[1] == 0  # top-left y


def test_pixel_coordinates_are_truncated_to_int():
    # 0.25*10=2.5 -> 2, (0.25+0.5)*10=7.5 -> 7: int() truncates, never rounds.
    box = normalized_bbox_to_box(0.25, 0.0, 0.5, 0.5, 10, 10)
    assert box == [2, 5, 7, 5, 7, 10, 2, 10]
    assert all(isinstance(c, int) for c in box)


def test_corner_ordering_is_clockwise_from_top_left():
    box = normalized_bbox_to_box(0.1, 0.2, 0.3, 0.4, 640, 480)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    # TL/TR share the top edge; BR/BL share the bottom edge.
    assert y1 == y2 and y3 == y4
    # TL/BL share the left edge; TR/BR share the right edge.
    assert x1 == x4 and x2 == x3
    # Width and height are positive (top-left above bottom-right).
    assert x2 > x1 and y3 > y1
    assert len(box) == 8
