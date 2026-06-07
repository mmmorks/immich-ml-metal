"""
CLIP model implementation using MLX for Apple Silicon acceleration.

Supports dynamic model loading based on Immich requests.
Thread-safe for both loading and inference.
"""

import contextlib
import gc
import io
import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from src.models.immich_preprocess import siglip_image_pixels

logger = logging.getLogger(__name__)


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding, guarding against a zero vector.

    A genuinely zero pooled output (degenerate input or fp16 underflow) has a
    zero norm; dividing by it yields an all-NaN vector that silently poisons the
    smart-search index or query. Return the (zero) vector unchanged instead.
    """
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def _l2_normalize_torch(embedding):
    """Torch counterpart of :func:`_l2_normalize` for the open_clip fallbacks.

    Same hazard: a zero pooled output has a zero norm, and dividing by it yields
    an all-NaN embedding that silently poisons the smart-search index or query.
    Leave a zero (sub-)vector unchanged instead. ``torch.where`` evaluates both
    branches, so the NaN from the zero-norm division is still computed — but it
    lands only in the discarded branch, never in the returned tensor.
    """
    import torch

    norm = embedding.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, embedding / norm, embedding)


# Model name mapping: Immich name -> MLX repo (or None to use open_clip fallback)
MODEL_MAP = {
    # OpenAI CLIP models -> MLX
    "ViT-B-32__openai": "mlx-community/clip-vit-base-patch32",
    "ViT-B-16__openai": "mlx-community/clip-vit-base-patch16",
    "ViT-L-14__openai": "mlx-community/clip-vit-large-patch14",
    # LAION CLIP models -> MLX
    "ViT-B-32__laion2b-s34b-b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    "ViT-B-32__laion2b_s34b_b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    # SigLIP / SigLIP2 models -> None here. SigLIP2 names handled natively via
    # MLX_EMBEDDINGS_MAP below (checked first in _load_model); anything that
    # stays None falls through to the open_clip fallback.
    "ViT-B-16-SigLIP__webli": None,
    "ViT-B-16-SigLIP2__webli": None,
    "ViT-SO400M-16-SigLIP2-384__webli": None,
    # Default fallback
    "default": "mlx-community/clip-vit-base-patch32",
}

# Native MLX SigLIP2 backend via Blaizzy/mlx-embeddings. Maps Immich's
# open_clip-style model name to the HF repo id the mlx-embeddings loader
# understands. The repo/dir name MUST contain a 'patchNN-NNN' token or the
# loader's regex (which is the only patch_size source — config.json omits it)
# crashes. mlx-embeddings loads the HF bf16 safetensors directly, so no
# separate conversion step is required (ml-ycd.7 productionizes caching).
# See the ml-ycd.1 spike writeup for the full rationale.
#
# This stays the *original bf16* repo on purpose: it is the on-demand convert
# source and the last-resort direct-load fallback. The pre-converted *fp16*
# download home is separate — `_siglip2_hf_repo()`, now mlx-community (ml-5t0).
MLX_EMBEDDINGS_MAP = {
    "ViT-SO400M-16-SigLIP2-384__webli": "google/siglip2-so400m-patch16-384",
}


# --- Local converted-weight cache (ml-ycd.7) ---------------------------------
#
# The native SigLIP2 backend can load the HF bf16 safetensors directly, but a
# one-time fp16 convert (scripts/convert_siglip2_mlx.py) is ~2.2 GB and avoids
# re-downloading/re-converting on every install. The accelerator auto-prefers a
# complete local convert over the HF repo, with an explicit override on top.

# Files a converted dir must contain to be usable. *.safetensors is checked
# separately (glob) since the shard count/name varies.
_SIGLIP2_REQUIRED_FILES = ("config.json", "tokenizer.json")


def _ml_model_cache_root() -> Path:
    """Root dir for locally-converted/cached MLX weights.

    Defaults to the repo's gitignored ``models/`` dir (see .gitignore); override
    with ``ML_MODEL_CACHE_DIR`` for installs outside the source tree.
    """
    env = os.getenv("ML_MODEL_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    # src/models/clip.py -> parents[2] == ml repo root
    return Path(__file__).resolve().parents[2] / "models"


def siglip2_cache_dir(repo_id: str) -> Path:
    """Default local cache dir for a converted SigLIP2 repo.

    The dir name is the repo basename, which for the supported model keeps the
    'patchNN-NNN' token the mlx-embeddings loader regex requires (ml-ycd.1) —
    config.json omits patch_size, so the path string is the only source.
    """
    return _ml_model_cache_root() / repo_id.split("/")[-1]


def siglip2_dir_is_complete(path) -> bool:
    """True if ``path`` holds a usable converted model.

    Requires config.json, tokenizer.json, and at least one ``*.safetensors`` so a
    half-written/aborted convert is ignored rather than loaded and crashing.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    if not all((p / f).is_file() for f in _SIGLIP2_REQUIRED_FILES):
        return False
    return any(p.glob("*.safetensors"))


def resolve_siglip2_source(repo_id: str) -> tuple[str, str]:
    """Resolve where to load SigLIP2 weights from, preferring local converts.

    Order: explicit ``ML_SIGLIP2_MLX_PATH`` override > a complete default cache
    dir (``siglip2_cache_dir``) > the HF repo id (bf16 safetensors, downloaded +
    cached by HF on first use). Returns ``(path_or_repo, source)`` where source
    is ``'override' | 'cache' | 'hf'``.
    """
    override = os.getenv("ML_SIGLIP2_MLX_PATH")
    if override:
        return override, "override"
    cache = siglip2_cache_dir(repo_id)
    if siglip2_dir_is_complete(cache):
        return str(cache), "cache"
    return repo_id, "hf"


def _siglip2_auto_convert_enabled() -> bool:
    """On-demand fp16 convert toggle (default on).

    Set ``ML_SIGLIP2_AUTO_CONVERT=0`` (or false/no) to keep loading the HF bf16
    weights directly instead of converting on first use.
    """
    return os.getenv("ML_SIGLIP2_AUTO_CONVERT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _siglip2_hf_repo() -> str:
    """Pre-converted fp16 SigLIP2 repo to snapshot before converting locally.

    Defaults to the canonical ``mlx-community`` convert (ml-yo9) — the same fp16
    bytes as the original ``mmmorks/...`` publish, but in the community org so
    installs pull from an upstream home. NOTE this is the fp16 *download* source,
    NOT ``MLX_EMBEDDINGS_MAP`` (which stays ``google/...``: it is the bf16
    *convert* source / last-resort load, and re-converting our own fp16 is
    untested). Set ``ML_SIGLIP2_HF_REPO=`` (empty) to disable the download step
    and convert the bf16 source locally instead.
    """
    return os.getenv("ML_SIGLIP2_HF_REPO", "mlx-community/siglip2-so400m-patch16-384").strip()


def _download_siglip2_repo(hf_repo: str, out: Path) -> bool:
    """Snapshot a pre-converted fp16 SigLIP2 repo into the local cache dir.

    Returns True iff the snapshot leaves ``out`` a complete cache. Raises on a
    download error so the caller can treat it as "not available" and fall through.
    """
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_repo, local_dir=str(out))
    return siglip2_dir_is_complete(out)


def _convert_siglip2(repo_id: str, out: Path) -> bool:
    """Convert the HF bf16 ``repo_id`` to fp16 in ``out``. True iff complete."""
    from mlx_embeddings.convert import convert

    out.parent.mkdir(parents=True, exist_ok=True)
    convert(hf_path=repo_id, mlx_path=str(out), dtype="float16")
    return siglip2_dir_is_complete(out)


def ensure_siglip2_source(repo_id: str) -> tuple[str, str]:
    """Resolve the SigLIP2 weights source, materializing a local fp16 cache if needed.

    Resolution order (ml-u2d / ml-ivw):

    1. ``ML_SIGLIP2_MLX_PATH`` override, or a complete local cache
       (:func:`resolve_siglip2_source`).
    2. Snapshot a pre-converted fp16 repo (``ML_SIGLIP2_HF_REPO``, default
       ``mlx-community/...``) into the local cache dir — fast, no local convert.
    3. On-demand fp16 convert of the HF bf16 repo into the cache dir
       (``ML_SIGLIP2_AUTO_CONVERT``, default on; ~2.2 GB write).
    4. The HF bf16 repo id itself (loaded + cached by HF) as a last resort.

    Steps 2 and 3 both yield ``source='cache'``. Any failure at a step logs and
    falls through to the next, so a load never breaks.
    """
    path_or_repo, source = resolve_siglip2_source(repo_id)
    if source != "hf":
        return path_or_repo, source

    out = siglip2_cache_dir(repo_id)
    # The loader regex parses patch size from the dir path, so the local cache
    # dir name must carry a 'patchNN-NNN' token; without it, skip local
    # materialization entirely and load the HF bf16 repo.
    if not re.search(r"patch\d+-\d+", out.name):
        logger.warning(f"Cache dir name {out.name!r} lacks a 'patchNN-NNN' token the loader regex needs; loading HF bf16 instead of materializing a local cache")
        return repo_id, "hf"

    # 2. Pre-converted fp16 repo download.
    hf_repo = _siglip2_hf_repo()
    if hf_repo:
        logger.info(f"Fetching pre-converted SigLIP2 fp16 weights {hf_repo} -> {out}")
        try:
            if _download_siglip2_repo(hf_repo, out):
                logger.info(f"Loaded pre-converted SigLIP2 from {hf_repo} -> {out}")
                return str(out), "cache"
            logger.warning(f"Downloaded {hf_repo} but {out} is incomplete; trying local convert")
        except Exception as e:
            logger.warning(
                f"Pre-converted SigLIP2 download from {hf_repo} failed ({e}); trying local convert",
                exc_info=True,
            )

    # 3. On-demand local convert.
    if _siglip2_auto_convert_enabled():
        logger.info(f"Converting SigLIP2 {repo_id} -> {out} once (fp16, ~2.2 GB; set ML_SIGLIP2_AUTO_CONVERT=0 to load HF bf16)")
        try:
            if _convert_siglip2(repo_id, out):
                logger.info(f"On-demand SigLIP2 convert complete -> {out}")
                return str(out), "cache"
            logger.warning(f"On-demand SigLIP2 convert left {out} incomplete")
        except Exception as e:
            logger.warning(f"On-demand SigLIP2 convert failed ({e})", exc_info=True)

    # 4. Last resort: HF bf16.
    logger.warning(f"Falling back to HF bf16 for SigLIP2: {repo_id}")
    return repo_id, "hf"


def _resolve_siglip2_tokenizer_json(path_or_repo: str) -> str:
    """Path to the tokenizer.json that matches the SigLIP2 weights at ``path_or_repo``.

    The tokenizer MUST come from the SAME source as the weights, or query
    embeddings silently diverge from the index (ml-qax). ``path_or_repo`` is
    whatever ``ensure_siglip2_source`` resolved — a local dir (cache or local
    override) or an HF repo id (a non-dir override, or the default bf16 repo).

    * Local dir: read the copied-in ``tokenizer.json`` (convert() / snapshot
      writes it alongside the weights).
    * Non-dir: download ``tokenizer.json`` from *that* repo id — never a
      hardcoded default, so a custom/quantized override gets its own tokenizer.
    """
    if os.path.isdir(path_or_repo):
        return os.path.join(path_or_repo, "tokenizer.json")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(path_or_repo, "tokenizer.json")


# open_clip model name mappings for fallback
OPENCLIP_MAP = {
    "ViT-B-32__openai": ("ViT-B-32-quickgelu", "openai"),
    "ViT-B-16__openai": ("ViT-B-16", "openai"),
    "ViT-L-14__openai": ("ViT-L-14", "openai"),
    "ViT-B-32__laion2b-s34b-b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-32__laion2b_s34b_b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-16-SigLIP__webli": ("ViT-B-16-SigLIP", "webli"),
    "ViT-B-16-SigLIP2__webli": ("ViT-B-16-SigLIP2", "webli"),
    "ViT-SO400M-16-SigLIP2-384__webli": ("ViT-SO400M-16-SigLIP2-384", "webli"),
}


def resolve_fallback_arch(model_name: str) -> tuple[str, str]:
    """Resolve an Immich CLIP model name to an open_clip ``(arch, pretrained)``.

    Resolution order:
    1. Exact match in ``OPENCLIP_MAP``.
    2. ``arch__pretrained`` split on the first ``__``. For the OpenAI weights,
       open_clip expects the quickgelu variant, so ``-quickgelu`` is appended
       unless the arch already carries it (or is a SigLIP arch, which has no
       quickgelu variant).
    3. Anything else falls back to ``ViT-B-32-quickgelu`` / ``openai``.
    """
    if model_name in OPENCLIP_MAP:
        return OPENCLIP_MAP[model_name]
    if "__" in model_name:
        arch, pretrained = model_name.split("__", 1)
        if pretrained == "openai" and "quickgelu" not in arch.lower() and "siglip" not in arch.lower():
            arch = arch + "-quickgelu"
        return arch, pretrained
    return "ViT-B-32-quickgelu", "openai"


class MLXClip:
    """CLIP model using MLX for Apple Silicon acceleration."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._tokenizer = None
        # Set on the native SigLIP2 load path (a SiglipTextTokenizer); tests
        # inject a plain callable, so type it as a generic text->ids callable.
        self._siglip_tokenizer: Callable[[str], Any] | None = None
        self._loaded = False
        self._repo_id = MODEL_MAP.get(model_name, MODEL_MAP.get("default"))
        # Use the global metal_lock — Vision framework also touches Metal
        # and concurrent MLX + Vision Metal access crashes the process.
        from ..gpu_lock import metal_lock

        self._inference_lock = metal_lock
        self._load_model()

    def _load_model(self):
        """Load the MLX CLIP model, or fallback to open_clip."""
        # Native MLX SigLIP2 backend (mlx-embeddings) takes precedence for
        # mapped names; on any failure fall back to open_clip.
        if self.model_name in MLX_EMBEDDINGS_MAP:
            try:
                self._load_siglip2_mlx()
                return
            except Exception as e:
                logger.error(f"mlx-embeddings SigLIP2 load failed: {e}", exc_info=True)
                logger.info("Falling back to open_clip for SigLIP2")
                self._load_fallback()
                return

        self._repo_id = MODEL_MAP.get(self.model_name)

        if self._repo_id is None and self.model_name not in OPENCLIP_MAP:
            logger.warning(f"Unknown model '{self.model_name}', using MLX default (ViT-B-32)")
            self._repo_id = MODEL_MAP["default"]

        if self._repo_id is None:
            logger.info(f"No MLX version for {self.model_name}, using open_clip fallback")
            self._load_fallback()
            return

        try:
            from mlx_clip import mlx_clip

            logger.info(f"Loading MLX CLIP model: {self.model_name} -> {self._repo_id}")
            self._model = mlx_clip(self._repo_id)
            self._loaded = True
            logger.info(f"Successfully loaded CLIP model via MLX: {self.model_name}")

        except ImportError:
            logger.warning("mlx_clip not available, falling back to open_clip with MPS")
            self._load_fallback()
        except Exception as e:
            logger.error(f"MLX model loading failed: {e}", exc_info=True)
            logger.info("Falling back to open_clip")
            self._load_fallback()

    def _load_siglip2_mlx(self):
        """Load a SigLIP2 model natively via mlx-embeddings.

        Returns (model, SiglipProcessor); the processor exposes both an
        image_processor and a tokenizer. Inference uses
        get_image_features / get_text_features (single-modality, un-normalized)
        — NOT Model.__call__, which requires both modalities (see ml-ycd.1).
        """
        from mlx_embeddings.utils import load

        repo = MLX_EMBEDDINGS_MAP[self.model_name]
        # ml-ycd.7 / ml-u2d: prefer a local fp16 convert over the HF bf16 download,
        # converting on demand the first time none exists. ensure_siglip2_source
        # picks (in order) the ML_SIGLIP2_MLX_PATH override, a complete cache dir,
        # else converts into the cache dir once (unless ML_SIGLIP2_AUTO_CONVERT=0),
        # falling back to the HF repo id on any failure. Any local dir name must
        # still contain 'patchNN-NNN' (the loader regex needs it).
        path_or_repo, source = ensure_siglip2_source(repo)

        logger.info(f"Loading SigLIP2 via mlx-embeddings: {self.model_name} -> {path_or_repo} (source={source})")
        # load() returns (model, SiglipProcessor), but the SigLIP2 paths
        # preprocess via src.models.immich_preprocess (siglip_image_pixels +
        # SiglipTextTokenizer, see ml-ycd.4), so the processor is unused here —
        # discard it rather than storing a dead reference.
        self._model, _ = load(path_or_repo)
        self._repo_id = path_or_repo

        # Immich-faithful text tokenizer (ml-ycd.4): the standard Immich server
        # applies clean_text (canonicalize) then a raw tokenizer.json. HF
        # SiglipProcessor skips canonicalization and diverges on caps/punctuation,
        # so we tokenize exactly like the server to keep query embeddings aligned
        # with the existing index. tokenizer.json ships with the converted dir
        # (override) or the HF repo snapshot.
        from src.models.immich_preprocess import SiglipTextTokenizer

        tokenizer_json = _resolve_siglip2_tokenizer_json(path_or_repo)
        self._siglip_tokenizer = SiglipTextTokenizer(tokenizer_json)

        self._use_mlx_embeddings = True
        self._loaded = True
        logger.info(f"Successfully loaded SigLIP2 via mlx-embeddings: {self.model_name}")

    def _load_fallback(self):
        """Fallback to open_clip with MPS acceleration."""
        try:
            import open_clip
            import torch
        except ImportError as e:
            logger.error(f"open_clip not available and MLX failed: {e}")
            raise RuntimeError("Neither mlx_clip nor open_clip available. Install one with: pip install open-clip-torch") from e

        arch, pretrained = resolve_fallback_arch(self.model_name)

        logger.info(f"Loading open_clip model: {arch} / {pretrained}")

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(arch)
        except Exception as e:
            logger.warning(f"Failed to load {arch}/{pretrained}: {e}")
            logger.info("Falling back to ViT-B-32-quickgelu/openai")
            arch, pretrained = "ViT-B-32-quickgelu", "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(arch)

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            model = model.to(self._device)
            logger.info("Using MPS (Metal) acceleration")
        else:
            self._device = torch.device("cpu")
            logger.warning("MPS not available, using CPU")

        model.eval()

        self._model = model
        self._processor = preprocess
        self._tokenizer = tokenizer
        self._use_fallback = True
        self._loaded = True

        logger.info(f"Successfully loaded CLIP model via open_clip: {arch}/{pretrained}")

    def _infer_with_swap_retry(self, label: str, prepare, run):
        """Shared scaffolding for every encode path.

        ``prepare(model_ref)`` runs OUTSIDE the inference lock (CPU-only image
        preprocessing / tokenization). ``run(model_ref, prepared)`` runs INSIDE
        the lock (GPU inference) and returns its result — any cheap host-side
        post-processing should be done by the caller on the returned value so
        the lock is held only for Metal/MPS work.

        A reference to the model is captured before ``prepare`` and re-checked
        after acquiring the lock, so a concurrent ``get_clip_model()`` swap
        (which can set ``self._model`` to ``None`` or a different instance) is
        caught instead of crashing on ``None``. On a swap detected mid-flight,
        the work is retried once against the current model.
        """
        for attempt in range(2):
            model_ref = self._model
            if model_ref is None:
                # A concurrent get_clip_model() switched models and unloaded
                # the instance we still hold (self._model -> None). Bail out
                # cleanly instead of crashing on a None attribute access.
                raise RuntimeError("CLIP model was unloaded during a concurrent model switch")
            prepared = prepare(model_ref)

            with self._inference_lock:
                if self._model is not model_ref:
                    if attempt == 0:
                        logger.warning("CLIP model changed during %s, retrying", label)
                        continue
                    raise RuntimeError(f"CLIP model changed during {label} after retry")
                return run(model_ref, prepared)

        # Should never reach here — range(2) always runs and either
        # returns or raises. Defensive guard.
        raise RuntimeError(f"CLIP {label} failed to produce an embedding")

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate CLIP embedding for an image.
        Thread-safe — only GPU inference is serialized. Image decode and
        preprocessing run outside the lock. A reference to the model is
        captured before preprocessing and verified after acquiring the lock
        so a concurrent model switch cannot cause a mismatch. If the model
        was swapped mid-flight, preprocessing is re-run once against the
        new model.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Decode outside lock — this is CPU work, not GPU
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if getattr(self, "_use_mlx_embeddings", False):
            return self._encode_image_siglip2(image)

        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._encode_image_fallback(image)

        # MLX path — preprocess the PIL image directly (no temp file needed).
        def prepare(model_ref):
            return model_ref.img_processor([image])

        def run(model_ref, processed):
            output = model_ref.model(pixel_values=processed)
            embedding = output.image_embeds[0]
            # Force Metal evaluation inside the lock — MLX arrays are lazy,
            # and Metal work must complete before releasing the lock so
            # Vision framework calls don't collide with in-flight Metal ops.
            if isinstance(embedding, mx.array):
                embedding = np.array(embedding)
            return embedding

        embedding = self._infer_with_swap_retry("preprocessing", prepare, run)
        embedding = _l2_normalize(embedding)
        return embedding.flatten().astype(np.float32)

    def _encode_image_fallback(self, image: Image.Image) -> np.ndarray:
        """Encode image using open_clip fallback.

        Preprocessing (resize/normalize) runs outside the lock since it's
        CPU-only. Only the MPS/GPU inference is serialized. Model reference
        is captured before preprocessing and verified after lock acquisition.
        One retry on model swap, same as encode_image.
        """
        import torch

        def prepare(model_ref):
            assert self._processor is not None
            return self._processor(image).unsqueeze(0).to(self._device)

        def run(model_ref, image_tensor):
            with torch.no_grad():
                embedding = model_ref.encode_image(image_tensor)
                return _l2_normalize_torch(embedding)

        embedding = self._infer_with_swap_retry("preprocessing (fallback)", prepare, run)
        # .cpu() triggers MPS device sync — safe outside the lock because MPS
        # uses its own command queue (unlike MLX which shares the Metal command
        # buffer with Vision framework).
        return embedding.squeeze().cpu().numpy().astype(np.float32)

    def _encode_image_siglip2(self, image: Image.Image) -> np.ndarray:
        """Encode image via the native MLX SigLIP2 backend (mlx-embeddings).

        Preprocessing replicates the standard Immich ML server exactly —
        resize-shortest-side to 384 + center-crop + normalize 0.5 (see
        src.models.immich_preprocess.siglip_image_pixels), NOT HF
        SiglipProcessor, which squashes to 384x384 and would diverge from the
        existing index on non-square photos. Preprocessing is model-independent
        and runs outside the lock; only Metal inference is serialized, with one
        retry on a concurrent model swap. get_image_features returns an
        un-normalized (1, 1152) pooled output, so we L2-normalize manually.
        """

        # Immich-faithful preprocessing is model-independent; run() does only
        # the lazy Metal inference inside the lock.
        def prepare(model_ref):
            return mx.array(siglip_image_pixels(image))

        def run(model_ref, pixel_values):
            features = model_ref.get_image_features(pixel_values=pixel_values)
            # Force Metal evaluation inside the lock — MLX arrays are lazy.
            return np.array(features[0])

        embedding = self._infer_with_swap_retry("preprocessing (siglip2)", prepare, run)
        embedding = _l2_normalize(embedding)
        return embedding.flatten().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text.
        Thread-safe — only GPU inference is serialized, matching encode_image.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if getattr(self, "_use_mlx_embeddings", False):
            return self._encode_text_siglip2(text)

        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._encode_text_fallback(text)

        with self._inference_lock:
            model_ref = self._model
            if model_ref is None:
                raise RuntimeError("CLIP model was unloaded during a concurrent model switch")
            embedding = model_ref.text_encoder(text)
            if isinstance(embedding, mx.array):
                embedding = np.array(embedding)
            embedding = _l2_normalize(embedding)
            return embedding.flatten().astype(np.float32)

    def _encode_text_fallback(self, text: str) -> np.ndarray:
        """Encode text using open_clip fallback.

        Tokenization runs outside the lock since it's CPU-only.
        Only the MPS/GPU inference is serialized, matching _encode_image_fallback.
        One retry on model swap, same as encode_image paths.
        """
        import torch

        def prepare(model_ref):
            assert self._tokenizer is not None
            return self._tokenizer([text]).to(self._device)

        def run(model_ref, tokens):
            with torch.no_grad():
                embedding = model_ref.encode_text(tokens)
                return _l2_normalize_torch(embedding)

        embedding = self._infer_with_swap_retry("tokenization (text fallback)", prepare, run)
        return embedding.squeeze().cpu().numpy().astype(np.float32)

    def _encode_text_siglip2(self, text: str) -> np.ndarray:
        """Encode text via the native MLX SigLIP2 backend (mlx-embeddings).

        Tokenization replicates the standard Immich ML server exactly —
        clean_text (canonicalize) then a raw tokenizer.json padded/truncated to
        64 (see src.models.immich_preprocess.SiglipTextTokenizer). HF
        SiglipProcessor skips canonicalization and diverges on caps/punctuation,
        so this keeps query embeddings aligned with the index. Tokenization is
        model-independent and runs outside the lock; only Metal inference is
        serialized, with one retry on a concurrent swap. get_text_features
        returns an un-normalized (1, 1152) pooled output, so we L2-normalize.
        """

        # Immich-faithful tokenization is model-independent; run() does only
        # the lazy Metal inference inside the lock.
        def prepare(model_ref):
            assert self._siglip_tokenizer is not None
            return mx.array(self._siglip_tokenizer(text))

        def run(model_ref, input_ids):
            features = model_ref.get_text_features(input_ids=input_ids)
            # Force Metal evaluation inside the lock — MLX arrays are lazy.
            return np.array(features[0])

        embedding = self._infer_with_swap_retry("tokenization (siglip2)", prepare, run)
        embedding = _l2_normalize(embedding)
        return embedding.flatten().astype(np.float32)

    def unload(self):
        """Unload model and free memory."""
        logger.info(f"Unloading CLIP model: {self.model_name}")
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._siglip_tokenizer = None
        self._loaded = False
        self._use_mlx_embeddings = False

        gc.collect()

        try:
            mx.clear_cache()
        except AttributeError:
            with contextlib.suppress(Exception):
                mx.metal.clear_cache()


# Global model cache with thread safety
_current_model: MLXClip | None = None
_current_model_name: str | None = None
_model_lock = threading.Lock()


def get_loaded_clip_model_name() -> str | None:
    """Name of the currently-loaded CLIP model, or None if none is loaded.

    Lets callers reuse the live model instead of forcing a switch to a configured
    default. The /health probe uses this so it never evicts the production model
    from the single CLIP slot (settings.clip_model can differ from what Immich
    actually requests).
    """
    return _current_model_name


def get_clip_model(model_name: str = "ViT-B-32__openai") -> MLXClip:
    """
    Get CLIP model, loading or switching as needed (thread-safe).

    If a different model is requested, unloads current model first to free memory.
    """
    global _current_model, _current_model_name

    normalized_name = model_name.replace("::", "__")

    with _model_lock:
        if _current_model is not None and _current_model_name != normalized_name:
            logger.info(f"Switching CLIP model: {_current_model_name} -> {normalized_name}")
            _current_model.unload()
            _current_model = None
            _current_model_name = None

        if _current_model is None:
            logger.info(f"Loading CLIP model: {normalized_name}")
            _current_model = MLXClip(normalized_name)
            _current_model_name = normalized_name

        return _current_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import sys

    logger.info("Testing CLIP model loading...")
    logger.info(f"Supported MLX models: {[k for k, v in MODEL_MAP.items() if v is not None and k != 'default']}")
    logger.info(f"Supported open_clip models: {list(OPENCLIP_MAP.keys())}")

    logger.info("\n--- Testing MLX model ---")
    clip = get_clip_model("ViT-B-32__openai")
    text_emb = clip.encode_text("a photo of a cat")
    logger.info(f"Text embedding shape: {text_emb.shape}")
    logger.info(f"Text embedding norm: {np.linalg.norm(text_emb):.4f}")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            img_emb = clip.encode_image(f.read())
        logger.info(f"Image embedding shape: {img_emb.shape}")
        similarity = np.dot(text_emb, img_emb)
        logger.info(f"Text-image similarity: {similarity:.4f}")

    logger.info("\n✅ CLIP tests passed!")
