"""Decide whether weights-gated parity tests run, skip, or hard-fail.

Default: auto-detect (skip if deps or golden artifacts are unavailable).
ML_RUN_PARITY=1: a missing prerequisite is a HARD FAILURE, so CI can guarantee
the gate actually executed instead of silently skipping.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
_REQUIRED = ("mlx", "onnxruntime", "numpy", "PIL")


class ParityUnavailable(RuntimeError):
    """Raised when ML_RUN_PARITY=1 but prerequisites are missing."""


def _deps_available() -> tuple[bool, str]:
    for mod in _REQUIRED:
        if importlib.util.find_spec(mod) is None:
            return False, f"missing dependency: {mod}"
    return True, ""


def _golden_present(stem: str) -> bool:
    return (GOLDEN_DIR / f"{stem}.npz").is_file()


def gate_reason(stem: str) -> str | None:
    """Return None to RUN, or a skip-reason string to SKIP.

    With ML_RUN_PARITY=1, an unavailable prerequisite raises ParityUnavailable
    (a hard failure) instead of returning a skip reason.
    """
    forced = os.getenv("ML_RUN_PARITY") == "1"
    ok, reason = _deps_available()
    if not ok:
        if forced:
            raise ParityUnavailable(reason)
        return reason
    if not _golden_present(stem):
        reason = f"golden artifact missing: {stem}.npz (run scripts/gen_parity_golden.py)"
        if forced:
            raise ParityUnavailable(reason)
        return reason
    return None
