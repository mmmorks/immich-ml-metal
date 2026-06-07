"""Pin the MLX SigLIP2 preprocessing to the standard Immich ML server's exactly.

The MLX backend's embeddings must be interchangeable with the Immich server's, so
these tests assert src.models.immich_preprocess reproduces Immich's
transforms.py / clip{visual,textual} algorithms bit-for-bit.
"""

import glob
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
