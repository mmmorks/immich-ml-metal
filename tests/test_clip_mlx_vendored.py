"""Unit tests for the vendored MLX CLIP backend (``src/models/clip_mlx.py``).

This backend replaces the dormant third-party ``mlx_clip`` package (a thin port
of Apple's mlx-examples CLIP). The ONE behavioral change from that port is that
the transformer activation is **configurable**: the port hardcoded
``quick_gelu``, but Immich's shipped OpenAI-CLIP ONNX exports run STANDARD
``gelu``. To stay interchangeable with the existing Immich smart-search index,
the OpenAI ports must therefore run ``gelu`` — NOT the checkpoint's native
``quick_gelu``. These tests pin the activation wiring that makes that possible;
end-to-end embedding parity is covered by ``scripts/clip_parity.py``.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


def test_resolve_activation_quick_gelu_matches_formula():
    from src.models.clip_mlx import resolve_activation

    f = resolve_activation("quick_gelu")
    x = mx.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.allclose(np.array(f(x)), np.array(x * mx.sigmoid(1.702 * x)), atol=1e-6)


def test_resolve_activation_gelu_matches_mlx_exact_gelu():
    from src.models.clip_mlx import resolve_activation

    f = resolve_activation("gelu")
    x = mx.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.allclose(np.array(f(x)), np.array(nn.gelu(x)), atol=1e-6)


def test_gelu_and_quick_gelu_are_distinct():
    """The two activations must genuinely differ — this gap is the entire ~0.97
    cosine divergence between the OpenAI checkpoint's native activation and
    Immich's standard-gelu ONNX export."""
    from src.models.clip_mlx import resolve_activation

    x = mx.array([1.0, 2.0, 3.0])
    g = np.array(resolve_activation("gelu")(x))
    q = np.array(resolve_activation("quick_gelu")(x))
    assert not np.allclose(g, q, atol=1e-3)


def test_resolve_activation_unknown_raises():
    from src.models.clip_mlx import resolve_activation

    with pytest.raises(ValueError):
        resolve_activation("not_an_activation")


def _text_config(hidden_act):
    from src.models.clip_mlx import CLIPTextConfig

    return CLIPTextConfig(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=1,
        max_position_embeddings=77,
        vocab_size=100,
        layer_norm_eps=1e-5,
        hidden_act=hidden_act,
    )


def test_mlp_uses_configured_activation():
    """The MLP must take its activation from the config (the fix for the port's
    hardcoded ``quick_gelu``)."""
    from src.models.clip_mlx import MLP, resolve_activation

    assert MLP(_text_config("gelu")).activation_fn is resolve_activation("gelu")
    assert MLP(_text_config("quick_gelu")).activation_fn is resolve_activation("quick_gelu")


def test_config_defaults_to_quick_gelu():
    """Default activation stays ``quick_gelu`` — the OpenAI checkpoint's native
    activation — so the model is faithful to a checkpoint by default and only
    diverges where a caller explicitly overrides (the Immich-parity case)."""
    from src.models.clip_mlx import CLIPTextConfig

    cfg = CLIPTextConfig(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=1,
        max_position_embeddings=77,
        vocab_size=100,
        layer_norm_eps=1e-5,
    )
    assert cfg.hidden_act == "quick_gelu"
