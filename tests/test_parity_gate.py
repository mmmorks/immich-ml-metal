"""Hermetic tests for the parity-gate skip/hard-fail decision logic."""
import pytest

from tests import _parity_gate as g


def test_available_returns_run(monkeypatch):
    monkeypatch.setattr(g, "_deps_available", lambda: (True, ""))
    monkeypatch.setattr(g, "_golden_present", lambda stem: True)
    assert g.gate_reason("openai_clip") is None


def test_missing_deps_skips_by_default(monkeypatch):
    monkeypatch.delenv("ML_RUN_PARITY", raising=False)
    monkeypatch.setattr(g, "_deps_available", lambda: (False, "no mlx"))
    assert g.gate_reason("openai_clip") == "no mlx"


def test_missing_golden_skips_by_default(monkeypatch):
    monkeypatch.delenv("ML_RUN_PARITY", raising=False)
    monkeypatch.setattr(g, "_deps_available", lambda: (True, ""))
    monkeypatch.setattr(g, "_golden_present", lambda stem: False)
    reason = g.gate_reason("face")
    assert reason is not None and "golden" in reason.lower()


def test_forced_missing_is_hard_fail(monkeypatch):
    monkeypatch.setenv("ML_RUN_PARITY", "1")
    monkeypatch.setattr(g, "_deps_available", lambda: (False, "no onnxruntime"))
    with pytest.raises(g.ParityUnavailable, match="no onnxruntime"):
        g.gate_reason("openai_clip")
