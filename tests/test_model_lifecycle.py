"""Tests for model memory management — busy guards, unload decisions, tracking."""

import time

from src.main import (
    MODEL_MEMORY_FLOOR_MB,
    _available_memory_mb,
    _mark_model_busy,
    _model_busy,
    _model_last_used,
    _should_unload,
    _track_model_use,
)


def _cleanup():
    """Reset global state between tests."""
    _model_busy.discard("clip")
    _model_busy.discard("face")
    _model_last_used.pop("clip", None)
    _model_last_used.pop("face", None)


def setup_function():
    _cleanup()


def teardown_function():
    _cleanup()


# --- Busy guard ---


def test_busy_model_never_unloads():
    _mark_model_busy("clip")
    assert "clip" in _model_busy
    # Even with 0MB available, busy model should not unload
    assert not _should_unload("clip", time.monotonic(), 0)


def test_track_clears_busy():
    _mark_model_busy("clip")
    _track_model_use("clip")
    assert "clip" not in _model_busy
    assert "clip" in _model_last_used


# --- Pressure strategy ---


def test_pressure_recently_used_not_unloaded():
    """Model used just now should not unload even under pressure."""
    _track_model_use("clip")
    assert not _should_unload("clip", time.monotonic(), 0)


def test_pressure_idle_and_low_memory_unloads():
    """Model idle for >30s with low memory should unload."""
    _model_last_used["clip"] = time.monotonic() - 60
    assert _should_unload("clip", time.monotonic(), 0)


def test_pressure_idle_but_enough_memory_stays():
    """Model idle for >30s but plenty of memory — don't unload."""
    _model_last_used["clip"] = time.monotonic() - 60
    assert not _should_unload("clip", time.monotonic(), MODEL_MEMORY_FLOOR_MB + 100)


# --- Available-memory measurement ---
#
# The decision logic above is fed by _available_memory_mb(). Every test in this
# file passes avail_mb IN, so none of them ever exercised the measurement — and
# the measurement was the broken half. Measured on a 24 GB host 2026-08-01:
# free+inactive reported 2452 MB (well above the 500 MB floor, so nothing ever
# unloaded) while free was 68 MB, the compressor held 7.2 GB, and swap churned
# ~147 MB/s in BOTH directions. macOS keeps a large inactive list and answers
# pressure by compressing/swapping rather than draining it, so free+inactive
# cannot fall below the floor on a thrashing machine.


def test_available_memory_is_zero_when_kernel_reports_pressure_warn(monkeypatch):
    """Kernel says warn (2) -> report no memory, whatever the page counts say."""
    monkeypatch.setattr("src.main._vm_pressure_level", lambda: 2, raising=False)
    assert _available_memory_mb() == 0


def test_available_memory_is_zero_when_kernel_reports_pressure_critical(monkeypatch):
    """Kernel says critical (4) -> report no memory."""
    monkeypatch.setattr("src.main._vm_pressure_level", lambda: 4, raising=False)
    assert _available_memory_mb() == 0


def test_idle_model_unloads_when_kernel_reports_pressure(monkeypatch):
    """End-to-end: the bug was that this combination never triggered an unload."""
    monkeypatch.setattr("src.main._vm_pressure_level", lambda: 2, raising=False)
    _model_last_used["clip"] = time.monotonic() - 60
    assert _should_unload("clip", time.monotonic(), _available_memory_mb())


def test_available_memory_reports_pages_when_pressure_normal(monkeypatch):
    """Regression guard: a healthy machine must keep models loaded."""
    monkeypatch.setattr("src.main._vm_pressure_level", lambda: 1, raising=False)
    assert _available_memory_mb() > MODEL_MEMORY_FLOOR_MB


def test_available_memory_falls_back_to_page_math_when_pressure_unreadable(monkeypatch):
    """An unreadable sysctl must not be read as 'no memory' — that would unload constantly."""
    monkeypatch.setattr("src.main._vm_pressure_level", lambda: None, raising=False)
    assert _available_memory_mb() > 0


# --- Timeout strategy ---


def test_timeout_strategy(monkeypatch):
    monkeypatch.setattr("src.main.MODEL_UNLOAD_STRATEGY", "timeout")
    monkeypatch.setattr("src.main.MODEL_IDLE_TIMEOUT", 10)
    _model_last_used["clip"] = time.monotonic() - 15
    assert _should_unload("clip", time.monotonic(), 9999)


def test_timeout_strategy_not_yet(monkeypatch):
    monkeypatch.setattr("src.main.MODEL_UNLOAD_STRATEGY", "timeout")
    monkeypatch.setattr("src.main.MODEL_IDLE_TIMEOUT", 10)
    _track_model_use("clip")
    assert not _should_unload("clip", time.monotonic(), 9999)


# --- Never strategy ---


def test_never_strategy(monkeypatch):
    monkeypatch.setattr("src.main.MODEL_UNLOAD_STRATEGY", "never")
    _model_last_used["clip"] = time.monotonic() - 9999
    assert not _should_unload("clip", time.monotonic(), 0)


# --- Multiple models ---


def test_independent_model_tracking():
    """CLIP and face have independent busy/tracking state."""
    _mark_model_busy("clip")
    _track_model_use("face")
    assert "clip" in _model_busy
    assert "face" not in _model_busy
    assert "face" in _model_last_used
    assert "clip" not in _model_last_used
