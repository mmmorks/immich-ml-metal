"""Unit tests for OCR bbox -> quadrilateral conversion (normalized_bbox_to_box).

Pure-logic coverage for the Vision (bottom-left-origin, normalized) to Immich
(top-left-origin, 8-coord clockwise quad) conversion. Covers the Y-flip and
corner ordering.

output must be floats normalized to [0, 1] (pixel coord / image
dimension), matching upstream immich_ml's OCR contract — NOT absolute pixel
ints, which downstream overlay consumers would mis-place.
"""

from src.models.ocr import normalized_bbox_to_box


def test_full_frame_box_maps_to_normalized_corners():
    # Whole image: origin (0,0), size (1,1). Y-flip leaves it spanning the
    # frame; the normalized quad is the unit square TL,TR,BR,BL.
    assert normalized_bbox_to_box(0.0, 0.0, 1.0, 1.0, 100, 200) == [
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
    ]


def test_output_values_are_floats_in_unit_range():
    box = normalized_bbox_to_box(0.1, 0.2, 0.3, 0.4, 640, 480)
    assert all(isinstance(c, float) for c in box)
    assert all(0.0 <= c <= 1.0 for c in box)


def test_y_axis_is_flipped():
    # A Vision box at the bottom (origin_y=0) sits in the LOWER half of the
    # image (large normalized-y), proving the top-left-origin flip.
    box = normalized_bbox_to_box(0.0, 0.0, 0.5, 0.5, 100, 100)
    assert box == [0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0]

    # A Vision box at the top (origin_y + height == 1.0) sits at normalized-y 0.
    top = normalized_bbox_to_box(0.0, 0.5, 0.5, 0.5, 100, 100)
    assert top[1] == 0.0  # top-left y


def test_coords_are_pixel_over_image_dimension():
    # 0.25 origin, 0.5 width on a 10px image -> normalized 0.25 and 0.75;
    # no integer truncation (the old pixel-int path returned 2 and 7).
    box = normalized_bbox_to_box(0.25, 0.0, 0.5, 0.5, 10, 10)
    assert box == [0.25, 0.5, 0.75, 0.5, 0.75, 1.0, 0.25, 1.0]


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
