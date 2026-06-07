"""
OCR (Optical Character Recognition) using Apple's Vision framework.

Uses VNRecognizeTextRequest for hardware-accelerated text recognition on Apple Silicon.
"""

import io
import logging

import Vision
from Foundation import NSAutoreleasePool, NSData
from PIL import Image

logger = logging.getLogger(__name__)


def normalized_bbox_to_box(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> list[float]:
    """Convert a Vision bounding box to Immich's 8-coordinate quadrilateral.

    Vision returns normalized coordinates with the origin at the bottom-left,
    so the Y axis is flipped to Immich's top-left origin. The output matches
    upstream immich_ml's OCR contract: coordinates are floats normalized to
    ``[0, 1]`` (pixel coord / image dimension), with the four corners ordered
    clockwise starting top-left:
    ``[x1, y1, x2, y2, x3, y3, x4, y4]`` = TL, TR, BR, BL. A full-frame region
    maps to ``[0, 0, 1, 0, 1, 1, 0, 1]``.

    ``img_width``/``img_height`` keep the derivation explicit (pixel coord /
    image dimension); Vision's coords are already normalized so the scale
    cancels, but mirroring upstream's pixel-then-divide path keeps the contract
    obvious. Note that Vision only exposes an axis-aligned bounding box, so the
    quad is always a rectangle — rotated text quadrilaterals (which PaddleOCR
    returns) cannot be represented (see README "Known differences").
    """
    x = origin_x * img_width
    y = (1.0 - origin_y - height) * img_height
    w = width * img_width
    h = height * img_height

    x1, y1 = x, y  # top-left
    x2, y2 = x + w, y  # top-right
    x3, y3 = x + w, y + h  # bottom-right
    x4, y4 = x, y + h  # bottom-left

    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height,
        x3 / img_width,
        y3 / img_height,
        x4 / img_width,
        y4 / img_height,
    ]


def recognize_text(image_bytes: bytes, min_confidence: float = 0.0, use_language_correction: bool = True) -> dict:
    """
    Perform OCR using Apple's Vision framework.

    Always uses Vision's "accurate" recognition level (matching Immich, which
    has no fast-OCR mode).

    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.)
        min_confidence: Minimum confidence threshold (0.0 - 1.0)
        use_language_correction: Enable language correction (better for natural text,
                                 disable for technical text, serial numbers, codes)

    Returns:
        Dict matching Immich OCR response format:
        {
            "text": [str, ...],
            "box": [x1, y1, x2, y2, x3, y3, x4, y4, ...],  # 8 normalized [0,1] coords per text
            "boxScore": [float, ...],
            "textScore": [float, ...]
        }
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size
    except Exception as e:
        # Hard decode failure — raise so the request fails (non-2xx) and Immich
        # retries, instead of silently returning an empty result that marks the
        # asset permanently processed and hides the failure (ml-1s2). This is a
        # hard error, distinct from a genuinely text-free image (empty success).
        logger.error(f"Failed to load image for OCR: {e}")
        raise

    # Use autorelease pool to prevent memory accumulation in long-running service
    pool = NSAutoreleasePool.alloc().init()
    try:
        return _recognize_text_impl(image_bytes, img_width, img_height, min_confidence, use_language_correction)
    finally:
        del pool


def _recognize_text_impl(image_bytes: bytes, img_width: int, img_height: int, min_confidence: float, use_language_correction: bool) -> dict:
    """Internal OCR implementation (assumes autorelease pool is active)."""
    try:
        ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()

        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(use_language_correction)

        # Vision framework is thread-safe with separate handlers.
        # No gpu_lock needed — can overlap with CLIP and face detection.
        success, error = handler.performRequests_error_([request], None)

        if not success or error:
            # Hard Vision-framework failure — raise so Immich retries rather
            # than recording a false "no text" result (ml-1s2). A successful
            # request with zero observations falls through to an empty result.
            raise RuntimeError(f"Vision OCR request failed: {error}")

        texts = []
        boxes = []
        box_scores = []
        text_scores = []

        results = request.results() or []

        for observation in results:
            # Get observation confidence first (used for both box and fallback text score)
            observation_confidence = float(observation.confidence())

            candidates = observation.topCandidates_(1)
            if not candidates or len(candidates) == 0:
                continue

            candidate = candidates[0]
            candidate_confidence = float(candidate.confidence())

            # Filter by confidence (use candidate confidence for text filtering)
            if candidate_confidence < min_confidence:
                continue

            text = candidate.string()
            texts.append(text)
            text_scores.append(candidate_confidence)

            # Get bounding box (normalized coordinates, origin at bottom-left)
            bbox = observation.boundingBox()

            # Immich expects 8 coordinates per box (quadrilateral corners,
            # clockwise from top-left) in top-left-origin pixel space.
            boxes.extend(
                normalized_bbox_to_box(
                    bbox.origin.x,
                    bbox.origin.y,
                    bbox.size.width,
                    bbox.size.height,
                    img_width,
                    img_height,
                )
            )

            # Box score uses observation confidence (detection confidence)
            box_scores.append(observation_confidence)

        logger.debug(f"OCR detected {len(texts)} text region(s)")

        return {"text": texts, "box": boxes, "boxScore": box_scores, "textScore": text_scores}

    except Exception as e:
        # Unexpected hard failure during recognition — log with traceback, then
        # re-raise so the request fails and Immich retries rather than storing a
        # false empty result (ml-1s2).
        logger.error(f"OCR failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.models.ocr <image_path>")
        logger.info("Creating test image with text...")

        from PIL import ImageDraw

        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)

        draw.text((20, 30), "Hello World!", fill="black")
        draw.text((20, 80), "immich-ml-metal", fill="blue")
        draw.text((20, 130), "OCR Test 123", fill="darkgreen")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        test_bytes = buffer.getvalue()

        img.save("ocr_test.png")
        logger.info("Saved test image to ocr_test.png")
    else:
        with open(sys.argv[1], "rb") as f:
            test_bytes = f.read()

    logger.info("Testing Vision framework OCR...")
    logger.info("With language correction enabled:")
    result = recognize_text(test_bytes, use_language_correction=True)

    logger.info(f"Detected {len(result['text'])} text region(s):")
    for i, text in enumerate(result["text"]):
        text_score = result["textScore"][i]
        box_score = result["boxScore"][i]
        box_start = i * 8
        coords = result["box"][box_start : box_start + 8]
        logger.info(f'  [text:{text_score:.2f} box:{box_score:.2f}] "{text}"')
        logger.info(f"         Box (normalized): {[round(c, 4) for c in coords]}")

    logger.info("\n✅ OCR test complete!")
