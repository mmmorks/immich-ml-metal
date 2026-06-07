"""Configuration settings for immich-ml-metal."""

import logging
import os
from dataclasses import dataclass
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_VALID_LOG_LEVELS: frozenset[str] = frozenset(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])


# Captures an invalid ML_LOG_LEVEL so configure_logging() can warn about it
# AFTER basicConfig runs. Settings.from_env() executes at module import (well
# before main.py calls configure_logging), so warning here would emit through an
# unconfigured root logger and print unformatted.
_invalid_log_level: str | None = None


def _normalize_log_level(raw: str) -> LogLevel:
    """Coerce a user-supplied log level to a valid one, defaulting to INFO.

    An invalid ML_LOG_LEVEL would otherwise crash configure_logging() at
    getattr(logging, level) — guard it at the source so the stored value is
    always a real logging level. The warning is deferred to configure_logging()
    so it prints through the configured handler (see _invalid_log_level).
    """
    global _invalid_log_level
    level = raw.upper()
    if level not in _VALID_LOG_LEVELS:
        _invalid_log_level = raw
        return "INFO"
    return level  # type: ignore[return-value]


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 3003

    # CLIP settings. Defaults to the SigLIP2 model Immich requests by default for
    # smart search — it runs on the native MLX backend and needs no torch. (The
    # OpenAI CLIP ports also work but need a one-time torch conversion; see
    # requirements.txt.) Immich sends the model name per request, so this is only a
    # fallback for requests that omit it; keeping it torch-free keeps a default,
    # torch-free install fully functional.
    clip_model: str = "ViT-SO400M-16-SigLIP2-384__webli"

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
            clip_model=os.getenv("ML_CLIP_MODEL", "ViT-SO400M-16-SigLIP2-384__webli"),
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
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # Deferred from _normalize_log_level (runs at import, pre-basicConfig)
        # so the warning prints through the formatter configured just above.
        if _invalid_log_level is not None:
            logging.getLogger(__name__).warning("Invalid ML_LOG_LEVEL %r; fell back to INFO", _invalid_log_level)


# Global settings instance
settings = Settings.from_env()
