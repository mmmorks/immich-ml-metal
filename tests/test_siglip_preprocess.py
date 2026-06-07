"""Pin the MLX SigLIP2 preprocessing to the standard Immich ML server's exactly.

The MLX backend's embeddings must be interchangeable with the Immich server's, so
these tests assert src.models.immich_preprocess reproduces Immich's
transforms.py / clip{visual,textual} algorithms bit-for-bit.
"""

import glob
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from src.models.immich_preprocess import (
    SIGLIP2_IMAGE_SIZE,
    clean_text,
    siglip_image_pixels,
)


def _immich_image_ref(pil: Image.Image, size: int = 384, mean: float = 0.5, std: float = 0.5):
    """Independent re-derivation of immich_ml OpenClipVisualEncoder.transform
    (resize_pil shortest-side BICUBIC + crop_pil center + /255 + normalize)."""
    img = pil.convert("RGB")
    if img.width < img.height:
        img = img.resize((size, int((img.height / img.width) * size)), Image.Resampling.BICUBIC)
    else:
        img = img.resize((int((img.width / img.height) * size), size), Image.Resampling.BICUBIC)
    left = int((img.size[0] / 2) - (size / 2))
    upper = int((img.size[1] / 2) - (size / 2))
    img = img.crop((left, upper, left + size, upper + size))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
    return np.expand_dims(arr.transpose(2, 0, 1), 0)


def _make_image(w: int, h: int) -> Image.Image:
    rng = np.random.RandomState(w * 7919 + h)
    return Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


# Fixed per-mode seed offset so the random source image is reproducible (no
# PYTHONHASHSEED dependence) — both the function and the reference convert the
# SAME object, so the seed only matters for repeatable failures.
_MODE_SEED = {"L": 1, "RGBA": 2, "CMYK": 3, "P": 4}


def _make_image_mode(w: int, h: int, mode: str) -> Image.Image:
    """A deterministic non-RGB PIL image in ``mode`` (L/RGBA/CMYK/P)."""
    rng = np.random.RandomState(w * 7919 + h + _MODE_SEED[mode])
    if mode == "L":
        return Image.fromarray(rng.randint(0, 256, (h, w), dtype=np.uint8), "L")
    if mode in ("RGBA", "CMYK"):
        return Image.fromarray(rng.randint(0, 256, (h, w, 4), dtype=np.uint8), mode)
    if mode == "P":
        # Build from RGB so the palette is real; P->RGB then goes through the
        # palette, which both sides resolve identically.
        rgb = Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
        return rgb.convert("P")
    raise ValueError(f"unsupported mode {mode}")


# --------------------------------------------------------------------------- #
# clean_text — Immich's canonicalization (the step HF SiglipProcessor omits)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a photo of a cat", "a photo of a cat"),
        ("  A  PHOTO   of a CAT!! ", "a photo of a cat"),  # collapse + lower + strip punct
        ("close-up of a flower", "closeup of a flower"),  # hyphen stripped -> one word
        ("People Walking, at Night.", "people walking at night"),
    ],
)
def test_clean_text_canonicalize(raw, expected):
    assert clean_text(raw, canonicalize=True) == expected


def test_clean_text_no_canonicalize_only_collapses_whitespace():
    assert clean_text("  Hello,  World!  ", canonicalize=False) == "Hello, World!"


# --------------------------------------------------------------------------- #
# image transform — must equal the independent Immich replication, bit-for-bit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w,h", [(640, 480), (480, 640), (384, 384), (1000, 200), (200, 1000), (501, 333)])
def test_image_pixels_match_immich_exactly(w, h):
    img = _make_image(w, h)
    got = siglip_image_pixels(img)
    ref = _immich_image_ref(img)
    assert got.shape == (1, 3, SIGLIP2_IMAGE_SIZE, SIGLIP2_IMAGE_SIZE)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)


def test_image_normalize_range():
    # mean=std=0.5 maps [0,1] pixels to [-1,1].
    img = _make_image(640, 480)
    px = siglip_image_pixels(img)
    assert px.min() >= -1.0 - 1e-6
    assert px.max() <= 1.0 + 1e-6


def test_image_is_crop_not_squash():
    # A wide image: center-crop keeps a 384-wide slice of the 512-wide resize,
    # so the left/right edges are dropped — unlike a squash, which keeps them.
    # Verify our output differs from a naive squash-to-384x384.
    img = _make_image(800, 400)
    crop = siglip_image_pixels(img)
    squash = np.expand_dims(
        (np.asarray(img.resize((384, 384), Image.Resampling.BICUBIC), np.float32) / 255.0 - 0.5) / 0.5,
        0,
    ).transpose(0, 3, 1, 2)
    assert not np.allclose(crop, squash)


# --------------------------------------------------------------------------- #
# non-RGB inputs — convert("RGB") must run and stay bit-for-bit with Immich
# --------------------------------------------------------------------------- #
# Immich's transform opens every image and converts to RGB before resize; our
# siglip_image_pixels does the same (image.convert("RGB") for any non-RGB mode).
# Feed the SAME non-RGB image to both: since both apply PIL's identical
# convert("RGB"), the pixels must match exactly. Pins the L/RGBA/CMYK/P convert
# paths that the RGB-only tests above never exercised.
@pytest.mark.parametrize("mode", ["L", "RGBA", "CMYK", "P"])
@pytest.mark.parametrize("w,h", [(640, 480), (384, 384), (501, 333)])
def test_image_pixels_non_rgb_match_immich(mode, w, h):
    img = _make_image_mode(w, h, mode)
    assert img.mode == mode  # the input really is non-RGB
    got = siglip_image_pixels(img)
    ref = _immich_image_ref(img)
    assert got.shape == (1, 3, SIGLIP2_IMAGE_SIZE, SIGLIP2_IMAGE_SIZE)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)


def test_grayscale_expands_to_three_equal_channels():
    # L -> RGB replicates the single channel; resize/normalize are per-channel
    # identical, so all three output planes must stay equal. Independent of the
    # Immich reference.
    px = siglip_image_pixels(_make_image_mode(640, 480, "L"))[0]  # (3, H, W)
    np.testing.assert_array_equal(px[0], px[1])
    np.testing.assert_array_equal(px[1], px[2])


def test_exif_orientation_is_not_applied():
    """Immich sends an already-oriented preview, so siglip_image_pixels must NOT
    auto-transpose on the EXIF Orientation tag — doing so would double-rotate and
    diverge from the server's index. Pin that orientation is ignored."""
    base = _make_image(120, 200)  # portrait, non-square so a 90deg swap changes dims
    exif = base.getexif()
    exif[0x0112] = 6  # Orientation = "rotate 90 CW on display"
    buf = io.BytesIO()
    base.save(buf, format="JPEG", exif=exif, quality=95)
    reloaded = Image.open(io.BytesIO(buf.getvalue()))
    assert reloaded.getexif().get(0x0112) == 6  # tag survived the round-trip

    got = siglip_image_pixels(reloaded)
    # Processed as stored (orientation ignored), exactly like the independent
    # Immich reference — which also does not consult EXIF.
    np.testing.assert_array_equal(got, _immich_image_ref(reloaded))
    # And it differs from the transposed image: proof we did not silently rotate.
    transposed = ImageOps.exif_transpose(reloaded)
    assert transposed is not None  # in_place defaults False -> returns a new image
    assert transposed.size != reloaded.size  # 90deg swap changed (w, h)
    assert not np.array_equal(got, siglip_image_pixels(transposed))


# --------------------------------------------------------------------------- #
# tokenizer — needs the model's tokenizer.json (skip if not cached)
# --------------------------------------------------------------------------- #
def _cached_tokenizer_json():
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--siglip2-so400m-patch16-384/snapshots/*/tokenizer.json"))
    return hits[0] if hits else None


@pytest.mark.skipif(_cached_tokenizer_json() is None, reason="tokenizer.json not cached")
def test_tokenizer_canonicalizes_and_pads():
    from src.models.immich_preprocess import SIGLIP2_CONTEXT_LENGTH, SiglipTextTokenizer

    tokenizer_json = _cached_tokenizer_json()
    assert tokenizer_json is not None  # guaranteed by the skipif above
    tok = SiglipTextTokenizer(tokenizer_json)
    a = tok("a photo of a cat")
    b = tok("  A  PHOTO of a CAT!!! ")  # canonicalizes to the same string
    assert a.shape == (1, SIGLIP2_CONTEXT_LENGTH)
    assert a.dtype == np.int32
    np.testing.assert_array_equal(a, b)
