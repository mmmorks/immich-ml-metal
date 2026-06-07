"""
Model implementations for immich-ml-metal.

- clip: CLIP image/text embeddings (native MLX SigLIP2 + mlx-clip)
- face_detect: Face detection (Apple Vision framework)
- face_embed: Face embeddings (InsightFace ArcFace)
- ocr: Text recognition (Apple Vision framework)
"""

from .clip import MLXClip, get_clip_model
from .face_detect import detect_faces
from .face_embed import get_face_embedding, get_face_embeddings_batch, get_recognition_model
from .ocr import recognize_text

__all__ = [
    "MLXClip",
    "detect_faces",
    "get_clip_model",
    "get_face_embedding",
    "get_face_embeddings_batch",
    "get_recognition_model",
    "recognize_text",
]
