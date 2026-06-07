"""Configuration settings for immich-ml-metal."""

import os
from dataclasses import dataclass
from typing import Literal
import logging

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_VALID_LOG_LEVELS: frozenset[str] = frozenset(
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
)


def _normalize_log_level(raw: str) -> LogLevel:
    """Coerce a user-supplied log level to a valid one, defaulting to INFO.

    An invalid ML_LOG_LEVEL would otherwise crash configure_logging() at
    getattr(logging, level) — guard it at the source so the stored value is
    always a real logging level.
    """
    level = raw.upper()
    if level not in _VALID_LOG_LEVELS:
        logging.warning("Invalid ML_LOG_LEVEL %r; falling back to INFO", raw)
        return "INFO"
    return level  # type: ignore[return-value]


@dataclass
class Settings:
    """Application settings with sensible defaults."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 3003

    # CLIP settings - using smaller model for low-memory systems
    clip_model: str = "ViT-B-32__openai"
    
    # Face recognition settings
    # buffalo_l is Immich's default, provides best accuracy
    # buffalo_s and buffalo_m are smaller alternatives
    face_model: str = "buffalo_l"
    
    # Face detection threshold - Immich default is 0.7
    # Lower values = more faces detected (more false positives)
    # Higher values = fewer faces detected (more false negatives)
    face_min_score: float = 0.7
    
    # OCR settings
    # Detection/recognition minScore thresholds are supplied per-request by
    # Immich (task_config["detection"]/["recognition"] options); they are not
    # configured here. Only language correction is a local setting.
    ocr_use_language_correction: bool = True  # Disable for technical text/codes
    
    # Performance settings
    use_coreml: bool = True
    use_ane: bool = True  # Apple Neural Engine
    max_concurrent_requests: int = 4  # Queued requests before backpressure
    
    # Resource limits
    max_image_size: int = 50 * 1024 * 1024  # 50MB max upload
    request_timeout: int = 120  # max seconds a request waits for a free slot (queue backpressure); does not cap in-flight inference
    
    # Logging settings
    log_level: LogLevel = "INFO"
    log_requests: bool = True  # Log individual requests (disable for high volume)
    
    # Debug mode - when True, expose error details in responses
    # Should be False when service is network-accessible
    debug_mode: bool = False
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            host=os.getenv("ML_HOST", "0.0.0.0"),
            port=int(os.getenv("ML_PORT", "3003")),
            clip_model=os.getenv("ML_CLIP_MODEL", "ViT-B-32__openai"),
            face_model=os.getenv("ML_FACE_MODEL", "buffalo_l"),
            face_min_score=float(os.getenv("ML_FACE_MIN_SCORE", "0.7")),
            ocr_use_language_correction=os.getenv("ML_OCR_LANGUAGE_CORRECTION", "true").lower() == "true",
            use_coreml=os.getenv("ML_USE_COREML", "true").lower() == "true",
            use_ane=os.getenv("ML_USE_ANE", "true").lower() == "true",
            max_concurrent_requests=int(os.getenv("ML_MAX_CONCURRENT_REQUESTS", "4")),
            max_image_size=int(os.getenv("ML_MAX_IMAGE_SIZE", str(50 * 1024 * 1024))),
            request_timeout=int(os.getenv("ML_REQUEST_TIMEOUT", "120")),
            log_level=_normalize_log_level(os.getenv("ML_LOG_LEVEL", "INFO")),
            log_requests=os.getenv("ML_LOG_REQUESTS", "true").lower() == "true",
            debug_mode=os.getenv("ML_DEBUG_MODE", "false").lower() == "true",
        )
    
    def configure_logging(self):
        """Configure logging based on settings."""
        level = getattr(logging, self.log_level, logging.INFO)
        if not isinstance(level, int):
            level = logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# Global settings instance
settings = Settings.from_env()