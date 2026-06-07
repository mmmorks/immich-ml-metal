"""Immich-faithful CLIP preprocessing for the native MLX SigLIP2 backend.

The goal of this MLX server is to produce embeddings **interchangeable with the
standard Immich ML server's**, so smart-search queries keep matching the existing
index. That requires reproducing Immich's image + text preprocessing exactly —
not HF `SiglipProcessor`'s, which differs (it squashes images and skips text
canonicalization).

Mirrors immich-app/immich `machine-learning`:
  * `immich_ml/models/transforms.py`  — `resize_pil`, `crop_pil`, `to_numpy`,
    `normalize`, `clean_text`
  * `immich_ml/models/clip/visual.py` — `OpenClipVisualEncoder.transform`
    (resize-shortest-side + center-crop, NOT squash)
  * `immich_ml/models/clip/textual.py`— `OpenClipTextualEncoder.tokenize`
    (`clean_text` then a raw `tokenizers.Tokenizer`, padded/truncated to
    `context_length`)

SigLIP2 SO400M patch16-384 constants (fixed-resolution variant): image size 384,
mean/std 0.5, text context length 64, pad token ``<pad>``, canonicalize=True
(open_clip sets ``tokenizer_kwargs.clean == "canonicalize"`` for SigLIP).
"""

from __future__ import annotations

import string

import numpy as np
from PIL import Image

# SigLIP2 SO400M patch16-384 fixed-res preprocessing constants.
SIGLIP2_IMAGE_SIZE = 384
SIGLIP2_MEAN = (0.5, 0.5, 0.5)
SIGLIP2_STD = (0.5, 0.5, 0.5)
SIGLIP2_CONTEXT_LENGTH = 64
SIGLIP2_PAD_TOKEN = "<pad>"

# Immich's clean_text punctuation table (str.maketrans over string.punctuation).
_PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)


def clean_text(text: str, canonicalize: bool = True) -> str:
    """Immich `clean_text`: collapse whitespace, then (for SigLIP) lowercase and
    strip punctuation. Applied BEFORE tokenization — this is the step HF
    `SiglipProcessor` omits, which is why caps/punctuation queries diverge.
    """
    text = " ".join(text.split())
    if canonicalize:
        text = text.translate(_PUNCTUATION_TRANS).lower()
    return text


def _resize_shortest_side(img: Image.Image, size: int) -> Image.Image:
    """Immich `resize_pil`: PIL BICUBIC, scale the SHORTEST side to `size`,
    preserving aspect ratio. (PIL `resize` takes (width, height).)"""
    if img.width < img.height:
        return img.resize((size, int((img.height / img.width) * size)), Image.Resampling.BICUBIC)
    return img.resize((int((img.width / img.height) * size), size), Image.Resampling.BICUBIC)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    """Immich `crop_pil`: center crop to size x size (note the int() truncation,
    matched exactly so pixels line up bit-for-bit with the server)."""
    left = int((img.size[0] / 2) - (size / 2))
    upper = int((img.size[1] / 2) - (size / 2))
    return img.crop((left, upper, left + size, upper + size))


def siglip_image_pixels(
    image: Image.Image,
    size: int = SIGLIP2_IMAGE_SIZE,
    mean: tuple[float, float, float] = SIGLIP2_MEAN,
    std: tuple[float, float, float] = SIGLIP2_STD,
) -> np.ndarray:
    """Immich `OpenClipVisualEncoder.transform`: resize-shortest + center-crop +
    /255 + normalize. Returns NCHW float32 of shape (1, 3, size, size), the
    layout `get_image_features(pixel_values=...)` expects.
    """
    img = image if image.mode == "RGB" else image.convert("RGB")
    img = _resize_shortest_side(img, size)
    img = _center_crop(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.expand_dims(arr.transpose(2, 0, 1), 0)


class SiglipTextTokenizer:
    """Immich `OpenClipTextualEncoder` tokenization: `clean_text(canonicalize)`
    then a raw `tokenizers.Tokenizer` (loaded from the model's `tokenizer.json`),
    padded and truncated to `context_length`. Returns input_ids (1, ctx) int32.

    Uses the raw `tokenizers.Tokenizer` rather than an HF fast-tokenizer wrapper
    so the token ids match the server exactly.
    """

    def __init__(
        self,
        tokenizer_json_path: str,
        context_length: int = SIGLIP2_CONTEXT_LENGTH,
        pad_token: str = SIGLIP2_PAD_TOKEN,
        canonicalize: bool = True,
    ) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_json_path))
        pad_id = self._tok.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError(f"Pad token '{pad_token}' not found in tokenizer vocab")
        self._tok.enable_padding(length=context_length, pad_token=pad_token, pad_id=pad_id)
        self._tok.enable_truncation(max_length=context_length)
        self._canonicalize = canonicalize

    def __call__(self, text: str) -> np.ndarray:
        ids = self._tok.encode(clean_text(text, self._canonicalize)).ids
        return np.array([ids], dtype=np.int32)
