# Copyright © 2023-2024 Apple Inc.
# Vendored from harperreed/mlx_clip (MIT), itself a port of Apple's
# mlx-examples CLIP. Inlined here so this service owns the OpenAI-CLIP backend
# instead of depending on a dormant third-party package.
#
# ONE deliberate change from the upstream port: the transformer activation is
# read from config (``hidden_act``) and overridable per load, instead of a
# hardcoded ``quick_gelu``. Immich's shipped OpenAI-CLIP ONNX exports run
# STANDARD ``gelu`` — not the checkpoint's native ``quick_gelu`` — so to produce
# embeddings interchangeable with an existing Immich smart-search index, the
# OpenAI ports must run ``gelu``. quick_gelu against that index drifts ~0.97
# cosine (both towers, scaling with depth); standard gelu matches it to 1.0000.
# See scripts/clip_parity.py and the README "CLIP parity" notes.

import glob
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import regex
from mlx.core import linalg as LA
from PIL.Image import Image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Activation
# --------------------------------------------------------------------------- #
def quick_gelu(x: mx.array) -> mx.array:
    """A fast GELU approximation (x * sigmoid(1.702 x)) — OpenAI CLIP's native
    activation. https://github.com/hendrycks/GELUs"""
    return x * mx.sigmoid(1.702 * x)


# Maps a config ``hidden_act`` string to the MLX activation function. Stable
# module-level functions (not lambdas) so identity comparison holds in tests.
_ACTIVATIONS: dict[str, Callable[[mx.array], mx.array]] = {
    "quick_gelu": quick_gelu,
    "gelu": nn.gelu,  # exact (erf) GELU — matches Immich's ONNX export
    "gelu_pytorch_tanh": nn.gelu_approx,
    "gelu_new": nn.gelu_approx,
}


def resolve_activation(name: str) -> Callable[[mx.array], mx.array]:
    """Activation function for a config ``hidden_act`` string."""
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"Unsupported CLIP activation {name!r}; known: {sorted(_ACTIVATIONS)}") from None


# --------------------------------------------------------------------------- #
# Config + output dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class CLIPVisionOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array
    hidden_states: mx.array | None


@dataclass
class CLIPTextOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array


@dataclass
class CLIPModelOutput:
    loss: mx.array | None
    text_embeds: mx.array | None
    image_embeds: mx.array | None
    text_model_output: CLIPTextOutput | None
    vision_model_output: CLIPVisionOutput | None


@dataclass
class CLIPTextConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    max_position_embeddings: int
    vocab_size: int
    layer_norm_eps: float
    hidden_act: str = "quick_gelu"


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    hidden_act: str = "quick_gelu"


@dataclass
class CLIPConfig:
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int


# --------------------------------------------------------------------------- #
# Model — a faithful port of Apple's mlx-examples CLIP
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, bias: bool = False):
        super().__init__()
        if (dims % num_heads) != 0:
            raise ValueError(f"The input feature dimensions should be divisible by the number of heads ({dims} % {num_heads}) != 0")

        self.num_heads = num_heads
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat)


class MLP(nn.Module):
    def __init__(self, config: CLIPTextConfig | CLIPVisionConfig):
        super().__init__()
        self.config = config
        # Config-driven activation (the deliberate change from the upstream port,
        # which hardcoded quick_gelu). See module docstring.
        self.activation_fn = resolve_activation(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation_fn(self.fc1(x)))


class EncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, config: CLIPTextConfig | CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class TextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding.weight[: x.shape[1]]
        return embeddings


class Encoder(nn.Module):
    def __init__(self, config: CLIPTextConfig | CLIPVisionConfig):
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class ClipTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embeddings = TextEmbeddings(config)
        self.encoder = Encoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> CLIPTextOutput:
        B, N = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        x = self.embeddings(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        for layer in self.encoder.layers:
            x = layer(x, mask)
        last_hidden_state = self.final_layer_norm(x)
        pooler_output = last_hidden_state[mx.arange(B), eot_tokens]
        return CLIPTextOutput(pooler_output=pooler_output, last_hidden_state=last_hidden_state)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.zeros((config.hidden_size,))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding.weight
        return embeddings


class ClipVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)  # spelling matches HF checkpoint keys
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> CLIPVisionOutput:
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        encoder_states = (x,) if output_hidden_states else None
        for layer in self.encoder.layers:
            x = layer(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)
        pooler_output = self.post_layernorm(x[:, 0, :])
        return CLIPVisionOutput(pooler_output=pooler_output, last_hidden_state=x, hidden_states=encoder_states)


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        self.text_model = ClipTextModel(config.text_config)
        self.vision_model = ClipVisionModel(config.vision_config)

        text_embed_dim = config.text_config.hidden_size
        vision_embed_dim = config.vision_config.hidden_size
        projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim, bias=False)
        self.logit_scale = mx.array(0.0)

    def get_text_features(self, x: mx.array) -> mx.array:
        return self.text_projection(self.text_model(x).pooler_output)

    def get_image_features(self, x: mx.array) -> mx.array:
        return self.visual_projection(self.vision_model(x).pooler_output)

    def __call__(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None) -> CLIPModelOutput:
        text_embeds = text_model_output = None
        if input_ids is not None:
            text_model_output = self.text_model(input_ids)
            text_embeds = self.text_projection(text_model_output.pooler_output)
            text_embeds = text_embeds / LA.norm(text_embeds, axis=-1, keepdims=True)

        image_embeds = vision_model_output = None
        if pixel_values is not None:
            vision_model_output = self.vision_model(pixel_values)
            image_embeds = self.visual_projection(vision_model_output.pooler_output)
            image_embeds = image_embeds / LA.norm(image_embeds, axis=-1, keepdims=True)

        return CLIPModelOutput(
            loss=None,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            vision_model_output=vision_model_output,
            text_model_output=text_model_output,
        )

    @staticmethod
    def from_pretrained(path: str, hidden_act: str | None = None):
        """Load a converted CLIP model from ``path``.

        ``hidden_act`` overrides the activation in BOTH towers (else the
        per-tower ``config.json`` value is used). Immich's OpenAI-CLIP ONNX
        exports run standard ``gelu``, so the production loader passes
        ``hidden_act="gelu"`` to match the existing index — see module docstring.
        """
        base = Path(path)
        with open(base / "config.json") as fid:
            config = json.load(fid)

        tc, vc = config["text_config"], config["vision_config"]
        text_config = CLIPTextConfig(
            num_hidden_layers=tc["num_hidden_layers"],
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_attention_heads=tc["num_attention_heads"],
            max_position_embeddings=tc["max_position_embeddings"],
            vocab_size=tc["vocab_size"],
            layer_norm_eps=tc["layer_norm_eps"],
            hidden_act=hidden_act or tc.get("hidden_act", "quick_gelu"),
        )
        vision_config = CLIPVisionConfig(
            num_hidden_layers=vc["num_hidden_layers"],
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_attention_heads=vc["num_attention_heads"],
            num_channels=3,
            image_size=vc["image_size"],
            patch_size=vc["patch_size"],
            layer_norm_eps=vc["layer_norm_eps"],
            hidden_act=hidden_act or vc.get("hidden_act", "quick_gelu"),
        )
        model = CLIPModel(CLIPConfig(text_config=text_config, vision_config=vision_config, projection_dim=config["projection_dim"]))

        weight_files = glob.glob(str(base / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        weights = CLIPModel.sanitize(weights)
        model.load_weights(list(weights.items()))
        return model

    @staticmethod
    def sanitize(weights):
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue  # unused buffer
            if "patch_embedding.weight" in k:
                # torch conv2d weight [out,in,kH,kW] -> mlx conv2d [out,kH,kW,in]
                sanitized[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized[k] = v
        return sanitized


# --------------------------------------------------------------------------- #
# Image processor — a port of HF transformers' CLIPImageProcessor
# --------------------------------------------------------------------------- #
class CLIPImageProcessor:
    def __init__(
        self,
        crop_size: int = 224,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        image_mean: list[float] = [0.48145466, 0.4578275, 0.40821073],
        image_std: list[float] = [0.26862954, 0.26130258, 0.27577711],
        size: int = 224,
        **kwargs,
    ) -> None:
        self.crop_size = crop_size
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.image_mean = mx.array(image_mean)
        self.image_std = mx.array(image_std)
        self.size = size

    def __call__(self, images: list[Image]) -> mx.array:
        return mx.concatenate([self._preprocess(image)[None] for image in images], axis=0)

    def _preprocess(self, image: Image) -> mx.array:
        if self.do_resize:
            image = _resize_short(image, self.size)
        if self.do_center_crop:
            image = _center_crop(image, (self.crop_size, self.crop_size))
        arr = mx.array(np.array(image)).astype(mx.float32) * (1 / 255.0)
        if self.do_normalize:
            arr = (arr - self.image_mean) / self.image_std
        return arr

    @staticmethod
    def from_pretrained(path: str):
        with open(Path(path) / "preprocessor_config.json", encoding="utf-8") as f:
            config = json.load(f)
        return CLIPImageProcessor(**config)


def _resize_short(image: Image, short_size: int) -> Image:
    width, height = image.size
    short, long = min(width, height), max(width, height)
    if short == short_size:
        return image
    new_short = short_size
    new_long = int(short_size * long / short)
    new_size = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize(new_size)


def _center_crop(image: Image, size: tuple[int, int]) -> Image:
    if size[0] % 2 != 0 or size[1] % 2 != 0:
        raise ValueError("Only even crop sizes supported.")
    original_width, original_height = image.size
    crop_height, crop_width = size
    top = (original_height - crop_height) // 2
    left = (original_width - crop_width) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


# --------------------------------------------------------------------------- #
# Tokenizer — a port of HF transformers' CLIPTokenizer
# --------------------------------------------------------------------------- #
class CLIPTokenizer:
    def __init__(self, bpe_ranks, vocab):
        self.bpe_ranks = bpe_ranks
        self.vocab = vocab
        self.pat = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )
        self._cache = {self.bos: self.bos, self.eos: self.eos}

    @property
    def bos(self):
        return "<|startoftext|>"

    @property
    def bos_token(self):
        return self.vocab[self.bos]

    @property
    def eos(self):
        return "<|endoftext|>"

    @property
    def eos_token(self):
        return self.vocab[self.eos]

    def bpe(self, text):
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:]))
        if not unique_bigrams:
            return unigrams

        while unique_bigrams:
            bigram = min(unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:]):
                if skip:
                    skip = False
                    continue
                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True
                else:
                    new_unigrams.append(a)
            if not skip:
                new_unigrams.append(b)
            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:]))

        self._cache[text] = unigrams
        return unigrams

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text, prepend_bos=True, append_eos=True) -> mx.array:
        if isinstance(text, list):
            return mx.array([self.tokenize(t, prepend_bos, append_eos) for t in text])

        clean = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean)
        bpe_tokens = [ti for t in tokens for ti in self.bpe(t)]

        ids = []
        if prepend_bos:
            ids.append(self.bos_token)
        ids.extend(self.vocab[t] for t in bpe_tokens)
        if append_eos:
            ids.append(self.eos_token)
        return mx.array(ids)

    @staticmethod
    def from_pretrained(path: str):
        base = Path(path)
        with open(base / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(base / "merges.txt", encoding="utf-8") as f:
            bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_merges = [tuple(m.split()) for m in bpe_merges]
        bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))
        return CLIPTokenizer(bpe_ranks, vocab)


# --------------------------------------------------------------------------- #
# One-time HF -> MLX weight conversion
# --------------------------------------------------------------------------- #
def _require_torch():
    """Import torch, raising an actionable error if it isn't installed.

    torch is an OPTIONAL, convert-only dependency — it is NOT in requirements.txt,
    to keep the default install slim. It is needed solely to read the source
    ``pytorch_model.bin`` pickle when first converting an OpenAI CLIP port to MLX
    weights. The default SigLIP2 smart-search path never needs it, and serving
    already-converted weights needs only mlx. Surface a clear instruction instead
    of a bare ImportError so a default (torch-free) install that requests an
    OpenAI CLIP model knows exactly what to do.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required to convert OpenAI CLIP weights to MLX on first use, "
            "but it is not installed. torch is an optional, convert-only dependency "
            "(not in requirements.txt). Install it with `pip install torch>=2.2.0`, "
            "or use the default SigLIP2 smart-search model, which needs no torch."
        ) from e
    return torch


def convert_weights(hf_repo: str, mlx_path: str, dtype: str = "float32") -> None:
    """Download an OpenAI CLIP checkpoint from HF and convert it to MLX weights.

    Needed only on first load of a given model (the converted weights are then
    cached and reloaded). torch is imported lazily here via ``_require_torch`` —
    it is required ONLY for this one-time conversion (the source
    ``pytorch_model.bin`` is a torch pickle); serving the cached weights needs
    only mlx.
    """
    import shutil

    torch = _require_torch()
    from huggingface_hub import snapshot_download

    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(snapshot_download(repo_id=hf_repo, allow_patterns=["*.bin", "*.json", "*.txt"]))

    torch_weights = torch.load(src / "pytorch_model.bin", map_location="cpu")
    mlx_dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]
    mlx_weights = {k: mx.array(v.to(torch.float32).numpy(), mlx_dtype) for k, v in torch_weights.items()}
    mx.save_safetensors(str(out / "model.safetensors"), mlx_weights)

    for fn in ("config.json", "merges.txt", "vocab.json", "preprocessor_config.json"):
        s = src / fn
        if s.is_file():
            shutil.copyfile(str(s), str(out / fn))
        else:
            logger.warning("%s not found in %s, skipping", fn, src)


# --------------------------------------------------------------------------- #
# Loader wrapper — the entry point src.models.clip uses
# --------------------------------------------------------------------------- #
class VendoredMlxClip:
    """Loads (converting on first use) a CLIP model + tokenizer + image processor.

    ``hidden_act`` selects the transformer activation; the production loader
    passes ``"gelu"`` for the OpenAI ports to match Immich's ONNX export. Exposes
    ``.model`` / ``.tokenizer`` / ``.img_processor`` — the same surface the
    previous third-party package did, so the call sites are unchanged.
    """

    def __init__(self, model_dir: str, hf_repo: str = "openai/clip-vit-base-patch32", hidden_act: str | None = None):
        self.hf_repo = hf_repo
        self.model_dir = model_dir
        path = Path(model_dir)
        if not path.exists() or not any(path.iterdir()):
            logger.info("Converting CLIP weights %s -> %s (one-time)", hf_repo, model_dir)
            convert_weights(hf_repo, model_dir)
        self.model = CLIPModel.from_pretrained(model_dir, hidden_act=hidden_act)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_dir)
        self.img_processor = CLIPImageProcessor.from_pretrained(model_dir)
