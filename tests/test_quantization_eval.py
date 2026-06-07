"""Tests for scripts/quantization_eval.py — ml-383.

Covers two edge bugs:

  1. ``recommend()`` divided by ``base.disk_bytes`` / ``base.peak_mem_bytes`` with
     no guard. ``mx.get_peak_memory()`` can return 0 on some paths, which made the
     OPT-IN line raise ZeroDivisionError (or print a misleading 100% saving).
  2. The reuse-existing-cache early return in ``ensure_convert()`` skipped
     ``_strip_skip_vision_key()``, so a dir from an interrupted run (converted but
     not yet stripped) or an older code version still carried ``skip_vision`` in
     ``config.json`` and crashed the loader with the exact TypeError the helper
     prevents.

The script imports ``mlx.core`` and ``embedding_parity`` at module top, so these
need the ml venv (``.venv/bin/python -m pytest``).
"""
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quantization_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("quantization_eval", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the frozen @dataclass can resolve its own module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _result(key, *, disk=1_000_000, peak=1_000_000, img_cos=1.0, txt_cos=1.0,
            top1=1.0, error=None):
    """Build a Result that clears recommend()'s gates by default."""
    r = mod.Result(
        key=key, note="", disk_bytes=disk, peak_mem_bytes=peak,
        img_ms=1.0, txt_ms=1.0, error=error,
    )
    if error is None and key != "fp16":
        stat = {"min": img_cos, "mean": img_cos, "median": img_cos}
        r.img_cos = stat
        r.txt_cos = {"min": txt_cos, "mean": txt_cos, "median": txt_cos}
        r.agree = {"top1_agreement": top1, "matrix_corr": 1.0}
    return r


# --------------------------------------------------------------------------- #
# Bug 1: recommend() div-by-zero on a zero baseline
# --------------------------------------------------------------------------- #
def test_recommend_handles_zero_peak_mem_baseline():
    """A 0 peak-mem baseline must not crash; the saving is reported n/a."""
    base = _result("fp16", peak=0)
    cand = _result("4bit-textonly", disk=500_000, peak=400_000)
    pick, rationale = mod.recommend([base, cand])
    assert pick == "fp16"
    text = "\n".join(rationale)
    assert "n/a" in text  # mem saving can't be computed against a 0 baseline
    assert "4bit-textonly" in text


def test_recommend_handles_zero_disk_baseline():
    base = _result("fp16", disk=0, peak=1_000_000)
    cand = _result("4bit-textonly", disk=500_000, peak=400_000)
    pick, rationale = mod.recommend([base, cand])  # must not raise
    assert pick == "fp16"
    assert "n/a" in "\n".join(rationale)


def test_recommend_reports_normal_savings():
    """Sanity: with a real baseline the percentages are still computed."""
    base = _result("fp16", disk=1_000_000, peak=1_000_000)
    cand = _result("4bit-textonly", disk=500_000, peak=750_000)
    _, rationale = mod.recommend([base, cand])
    text = "\n".join(rationale)
    assert "50%" in text  # 50% smaller disk
    assert "25%" in text  # 25% lower peak memory


# --------------------------------------------------------------------------- #
# Bug 2: reuse path must still strip skip_vision
# --------------------------------------------------------------------------- #
def test_ensure_convert_strips_skip_vision_on_reuse(tmp_path, monkeypatch):
    """A complete-but-unstripped quant dir is repaired on reuse, not returned raw."""
    base_name = "siglip2-so400m-patch16-384"
    out_root = tmp_path / "quant_eval"
    cfg = mod.CONFIGS["4bit-textonly"]
    out = out_root / f"{base_name}-{cfg.key}"
    out.mkdir(parents=True)
    # An interrupted/older convert left skip_vision in the vision_config.
    (out / "config.json").write_text(json.dumps({
        "vision_config": {"num_hidden_layers": 27, "skip_vision": True},
    }))

    # ensure_convert imports these from src.models.clip at call time.
    import src.models.clip as clip_mod
    monkeypatch.setattr(clip_mod, "siglip2_cache_dir", lambda repo: out_root / base_name)
    monkeypatch.setattr(clip_mod, "siglip2_dir_is_complete", lambda p: True)

    returned = mod.ensure_convert(cfg, "ignored-source", out_root)

    assert returned == out
    cfg_after = json.loads((out / "config.json").read_text())
    assert "skip_vision" not in cfg_after["vision_config"]
