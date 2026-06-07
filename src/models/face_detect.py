"""
Face detection using Apple's Vision framework.

Runs on the Neural Engine (ANE) for hardware acceleration.
"""

import io
import logging

import Vision
from Foundation import NSAutoreleasePool, NSData
from PIL import Image

logger = logging.getLogger(__name__)

# Nose-anchor reconstruction strategy for the ArcFace 5-point landmarks.
# Apple Vision gives a nose *contour*, not a single tip, so we pick one anchor:
#   "tip"    — the last contour point. Production default; matches the index the
#              library was built with (parity-verified PRESERVE on frontal LFW).
#   "center" — the contour centroid (mean). More robust where the last contour
#              point swings off the tip on non-frontal/occluded poses; the
#              candidate evaluated for drift before any adoption.
NOSE_STRATEGIES = ("tip", "center")
DEFAULT_NOSE_STRATEGY = "tip"


def _select_nose_point(nose_points, nose_strategy: str = DEFAULT_NOSE_STRATEGY):
    """Pick the nose anchor from the Vision nose-contour points.

    ``nose_points`` are Vision ``normalizedPoints`` (objects with ``.x``/``.y``,
    in face-bbox-relative coords). Returns ``(norm_x, norm_y)`` in that same
    space — the caller maps it to image pixels — or ``None`` if the contour is
    empty. Raises ``ValueError`` for an unknown strategy (fail loud rather than
    silently aligning on the wrong point).
    """
    if not nose_points:
        return None
    if nose_strategy == "tip":
        p = nose_points[-1]
        return (p.x, p.y)
    if nose_strategy == "center":
        n = len(nose_points)
        return (sum(p.x for p in nose_points) / n, sum(p.y for p in nose_points) / n)
    raise ValueError(f"unknown nose_strategy {nose_strategy!r}; expected one of {NOSE_STRATEGIES}")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _vision_bbox_to_pixels(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> dict[str, int]:
    """Convert a Vision normalized bbox to clamped image pixel coordinates.

    Vision uses a bottom-left origin in normalized [0,1] coords; image pixels
    use a top-left origin (so the Y axis is flipped). Vision can report boxes
    that extend past the image edges, so each coordinate is clamped to the
    image bounds. Both endpoints pass through the same monotonic clamp, so
    x2>=x1 and y2>=y1 are preserved.
    """
    x1 = origin_x * img_width
    y1 = (1.0 - origin_y - height) * img_height
    x2 = (origin_x + width) * img_width
    y2 = (1.0 - origin_y) * img_height

    x1 = _clamp(x1, 0, img_width)
    y1 = _clamp(y1, 0, img_height)
    x2 = _clamp(x2, 0, img_width)
    y2 = _clamp(y2, 0, img_height)

    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def detect_faces(
    image_bytes: bytes,
    nose_strategy: str = DEFAULT_NOSE_STRATEGY,
    img_width: int | None = None,
    img_height: int | None = None,
) -> tuple[list[dict], int, int]:
    """
    Detect faces using Apple's Vision framework.

    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.)
        nose_strategy: Nose-anchor reconstruction for the 5-point landmarks
            ("tip" = last nose-contour point, the production default; "center" =
            nose-contour centroid — the drift-evaluation variant).
        img_width, img_height: Image dimensions the caller already knows (the
            /predict handler opens the image once for these). When both are
            given, the image is not opened again just to read its size.

    Returns:
        Tuple of (faces, image_width, image_height)
        Each face dict contains:
          - boundingBox: {x1, y1, x2, y2} in pixels
          - score: confidence score
          - landmarks: 5-point landmarks for alignment (if available)
    """
    if img_width is None or img_height is None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = pil_image.size
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Invalid image data: {e}") from e

    # Use autorelease pool to prevent memory accumulation in long-running service
    pool = NSAutoreleasePool.alloc().init()
    try:
        return _detect_faces_impl(image_bytes, img_width, img_height, nose_strategy)
    finally:
        del pool


def _detect_faces_impl(image_bytes: bytes, img_width: int, img_height: int, nose_strategy: str = DEFAULT_NOSE_STRATEGY) -> tuple[list[dict], int, int]:
    """Internal face detection implementation (assumes autorelease pool is active)."""
    try:
        ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        request = Vision.VNDetectFaceLandmarksRequest.alloc().init()

        # Vision framework is thread-safe with separate handlers per call.
        # Can overlap with CLIP (MLX) as long as CLIP forces Metal eval
        # inside its own lock. No gpu_lock needed here.
        success, error = handler.performRequests_error_([request], None)

        if not success or error:
            # Hard Vision-framework failure — raise so Immich retries rather
            # than recording a false "no faces" result that permanently drops
            # the asset's faces. A successful request with zero observations
            # falls through to an empty face list below.
            raise RuntimeError(f"Vision face request failed: {error}")

        faces = []
        results = request.results() or []

        for observation in results:
            bbox = observation.boundingBox()

            # Convert to pixel coordinates (flips Y axis, clamps to image bounds)
            face_data = {
                "boundingBox": _vision_bbox_to_pixels(
                    bbox.origin.x,
                    bbox.origin.y,
                    bbox.size.width,
                    bbox.size.height,
                    img_width,
                    img_height,
                ),
                "score": float(observation.confidence()),
            }

            landmarks = observation.landmarks()
            if landmarks:
                five_points = extract_five_point_landmarks(landmarks, bbox, img_width, img_height, nose_strategy)
                if five_points is not None:
                    face_data["landmarks"] = five_points

            faces.append(face_data)

        logger.debug(f"Detected {len(faces)} face(s) in {img_width}x{img_height} image")
        return faces, img_width, img_height

    except Exception as e:
        # Unexpected hard failure during detection — log with traceback, then
        # re-raise so the request fails and Immich retries rather than storing a
        # false "no faces" result. A genuinely face-free image returns an empty
        # list above without raising.
        logger.error(f"Face detection failed: {e}", exc_info=True)
        raise


def extract_five_point_landmarks(
    landmarks: "Vision.VNFaceLandmarks2D",
    face_bbox,
    img_width: int,
    img_height: int,
    nose_strategy: str = DEFAULT_NOSE_STRATEGY,
) -> list[list[float]] | None:
    """
    Extract 5 landmark points for ArcFace alignment:
    - Left eye center
    - Right eye center
    - Nose anchor (``nose_strategy``: "tip" = last contour point, "center" = centroid)
    - Left mouth corner
    - Right mouth corner

    IMPORTANT: Vision framework's normalizedPoints are in the coordinate space
    of the face's bounding box (0..1 within the bbox), NOT the full image.
    We must map them through the face bounding box to get image pixel coords.
    See: VNImagePointForFaceLandmarkPoint() in VNUtils.h

    Args:
        landmarks: VNFaceLandmarks2D from the face observation
        face_bbox: The face observation's boundingBox() (normalized, bottom-left origin)
        img_width: Full image width in pixels
        img_height: Full image height in pixels

    Returns list of [x, y] points in pixel coordinates, or None if not available.
    """
    # Extract the face bounding box in normalized coords (bottom-left origin)
    bbox_x = face_bbox.origin.x
    bbox_y = face_bbox.origin.y
    bbox_w = face_bbox.size.width
    bbox_h = face_bbox.size.height

    def landmark_to_image_coords(norm_x: float, norm_y: float) -> list[float]:
        """
        Convert a landmark point from face-bbox-relative normalized coords
        to full image pixel coords.

        normalizedPoints are in face bbox space (0..1), with bottom-left origin.
        Face bbox is in image normalized space (0..1), with bottom-left origin.
        Final image pixels use top-left origin.
        """
        # Map from face-bbox-relative to image-normalized coords
        img_norm_x = bbox_x + norm_x * bbox_w
        img_norm_y = bbox_y + norm_y * bbox_h

        # Convert to pixel coords (flip Y: Vision uses bottom-left origin)
        px_x = img_norm_x * img_width
        px_y = (1.0 - img_norm_y) * img_height

        return [px_x, px_y]

    def get_region_points(region) -> list:
        """Convert PyObjC varlist to Python list of points."""
        if region is None:
            return []
        point_count = region.pointCount()
        if point_count == 0:
            return []
        raw_points = region.normalizedPoints()
        return [raw_points[i] for i in range(point_count)]

    def get_region_center(region) -> list[float] | None:
        """Get center point of a landmark region in image pixel coordinates."""
        points = get_region_points(region)
        if not points:
            return None

        x_sum = sum(p.x for p in points)
        y_sum = sum(p.y for p in points)
        n = len(points)

        return landmark_to_image_coords(x_sum / n, y_sum / n)

    try:
        # Left eye center
        left_eye = get_region_center(landmarks.leftEye())

        # Right eye center
        right_eye = get_region_center(landmarks.rightEye())

        # Nose anchor - tip (last contour point) or centroid, per nose_strategy
        nose = None
        nose_anchor = _select_nose_point(get_region_points(landmarks.nose()), nose_strategy)
        if nose_anchor is not None:
            nose = landmark_to_image_coords(*nose_anchor)

        # Mouth corners - find leftmost and rightmost points by x-coordinate
        # (Vision framework doesn't guarantee point ordering in contours)
        left_mouth = None
        right_mouth = None
        outer_lips_points = get_region_points(landmarks.outerLips())
        if outer_lips_points:
            # Convert all points to image pixel coordinates
            lips_px = [landmark_to_image_coords(p.x, p.y) for p in outer_lips_points]
            # Find extremes by x-coordinate
            left_mouth = min(lips_px, key=lambda p: p[0])
            right_mouth = max(lips_px, key=lambda p: p[0])

        # All 5 points must be present
        if left_eye and right_eye and nose and left_mouth and right_mouth:
            return [left_eye, right_eye, nose, left_mouth, right_mouth]

        logger.debug("Could not extract all 5 landmark points")
        return None

    except Exception as e:
        logger.warning(f"Landmark extraction failed: {e}")
        return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.models.face_detect <image_path>")
        logger.info("Creating test with a blank image...")

        test_img = Image.new("RGB", (640, 480), color=(200, 180, 170))
        buffer = io.BytesIO()
        test_img.save(buffer, format="JPEG")
        test_bytes = buffer.getvalue()
    else:
        with open(sys.argv[1], "rb") as f:
            test_bytes = f.read()

    logger.info("Testing Vision framework face detection...")
    faces, width, height = detect_faces(test_bytes)

    logger.info(f"Image size: {width}x{height}")
    logger.info(f"Faces detected: {len(faces)}")

    for i, face in enumerate(faces):
        logger.info(f"\nFace {i + 1}:")
        logger.info(f"  Bounding box: {face['boundingBox']}")
        logger.info(f"  Score: {face['score']:.3f}")
        if "landmarks" in face:
            logger.info("  Landmarks (5-point): ✓")
        else:
            logger.info("  Landmarks: not available")

    logger.info("\n✅ Face detection test complete!")
