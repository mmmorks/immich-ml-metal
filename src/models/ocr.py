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
) -> list[int]:
    """Convert a Vision bounding box to Immich's 8-coordinate quadrilateral.

    Vision returns normalized coordinates with the origin at the bottom-left,
    so the Y axis is flipped to Immich's top-left-origin pixel space. The
    result is the four corners (as ints) ordered clockwise starting top-left:
    ``[x1, y1, x2, y2, x3, y3, x4, y4]`` = TL, TR, BR, BL.
    """
    x = origin_x * img_width
    y = (1.0 - origin_y - height) * img_height
    w = width * img_width
    h = height * img_height

    x1, y1 = int(x), int(y)  # top-left
    x2, y2 = int(x + w), int(y)  # top-right
    x3, y3 = int(x + w), int(y + h)  # bottom-right
    x4, y4 = int(x), int(y + h)  # bottom-left

    return [x1, y1, x2, y2, x3, y3, x4, y4]


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
            "box": [x1, y1, x2, y2, x3, y3, x4, y4, ...],  # 8 coords per text
            "boxScore": [float, ...],
            "textScore": [float, ...]
        }
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return {"text": [], "box": [], "boxScore": [], "textScore": []}

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
            logger.error(f"Vision OCR error: {error}")
            return {"text": [], "box": [], "boxScore": [], "textScore": []}

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
        logger.error(f"OCR failed: {e}", exc_info=True)
        return {"text": [], "box": [], "boxScore": [], "textScore": []}


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
        logger.info(f"         Box: {coords}")

    logger.info("\n✅ OCR test complete!")
