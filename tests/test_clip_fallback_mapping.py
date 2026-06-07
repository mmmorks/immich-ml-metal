"""Unit tests for open_clip fallback name resolution (resolve_fallback_arch).

Pure-logic coverage for ml-7j8.10: the Immich -> open_clip
``(arch, pretrained)`` mapping was previously inlined in
MLXClip._load_fallback() and untested. Covers the OPENCLIP_MAP lookup, the
``arch__pretrained`` split, quickgelu suffixing rules, and the default.
"""
import pytest

from src.models.clip import OPENCLIP_MAP, resolve_fallback_arch


@pytest.mark.parametrize("name,expected", list(OPENCLIP_MAP.items()))
def test_explicit_map_entries_take_priority(name, expected):
    # Every curated entry must round-trip exactly, even when the generic
    # split/suffix rules would produce something different (e.g.
    # "ViT-B-32__openai" -> "ViT-B-32-quickgelu" is curated, not derived).
    assert resolve_fallback_arch(name) == expected


def test_openai_arch_gets_quickgelu_suffix():
    # Not in OPENCLIP_MAP -> derived. OpenAI weights need the quickgelu variant.
    assert resolve_fallback_arch("ViT-H-14__openai") == ("ViT-H-14-quickgelu", "openai")


def test_quickgelu_not_doubled_when_already_present():
    assert resolve_fallback_arch("ViT-H-14-quickgelu__openai") == (
        "ViT-H-14-quickgelu",
        "openai",
    )


def test_siglip_arch_never_gets_quickgelu():
    # SigLIP has no quickgelu variant; the suffix must be skipped even for the
    # openai pretrained tag.
    assert resolve_fallback_arch("ViT-B-16-SigLIP__openai") == (
        "ViT-B-16-SigLIP",
        "openai",
    )


def test_non_openai_pretrained_keeps_arch_verbatim():
    assert resolve_fallback_arch("ViT-B-32__laion400m_e32") == (
        "ViT-B-32",
        "laion400m_e32",
    )


def test_split_uses_only_first_separator():
    # split("__", 1): a pretrained tag containing "__" stays intact.
    assert resolve_fallback_arch("ViT-B-32__laion2b__s34b") == (
        "ViT-B-32",
        "laion2b__s34b",
    )


@pytest.mark.parametrize("name", ["RN50", "garbage", "", "no-separator-here"])
def test_names_without_separator_fall_back_to_default(name):
    assert resolve_fallback_arch(name) == ("ViT-B-32-quickgelu", "openai")
