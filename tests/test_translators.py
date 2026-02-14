"""Unit tests for subtitle translators.

Uses a few lines of JFK's famous speech in Chinese, translated to English.
Checks that the output is in English and that key words appear.
"""

import os
import pytest

from audio2sub import Segment
from audio2sub.translators import Gemini, Grok, OpenAI


CHINESE_SEGMENTS = [
    Segment(index=1, start=0.0, end=3.0, text="因此，我的美国同胞们，"),
    Segment(index=2, start=3.0, end=5.0, text="不要问你的国家能为你做什么，"),
    Segment(index=3, start=5.0, end=8.0, text="而要问你能为你的国家做什么。"),
]

EXPECTED_KEYWORDS = ["country", "ask", "fellow"]


def _is_mostly_english(text: str) -> bool:
    """Check if the text is predominantly English (ASCII characters)."""
    if not text.strip():
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.7


def _assert_translation_quality(translated_segments):
    """Assert that translation output is valid and in English."""
    assert len(translated_segments) == len(CHINESE_SEGMENTS)

    full_text = " ".join(seg.text for seg in translated_segments).lower()

    # Check that output is in English
    assert _is_mostly_english(
        full_text
    ), f"Translation does not appear to be in English: {full_text}"

    # Check that at least some expected keywords appear
    found = [kw for kw in EXPECTED_KEYWORDS if kw in full_text]
    assert len(found) >= 2, (
        f"Expected at least 2 of {EXPECTED_KEYWORDS} in translation, "
        f"but only found {found}. Full text: {full_text}"
    )

    # Check that each segment has non-empty text
    for seg in translated_segments:
        assert seg.text.strip(), f"Segment {seg.index} has empty text"

    # Check that indices are preserved
    for orig, trans in zip(CHINESE_SEGMENTS, translated_segments):
        assert (
            orig.index == trans.index
        ), f"Index mismatch: expected {orig.index}, got {trans.index}"


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_GEMINI_TEST")),
    reason="SKIP_GEMINI_TEST environment variable is set",
)
def test_gemini_translator():
    translator = Gemini()
    # Copy segments to avoid mutation
    segments = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in CHINESE_SEGMENTS
    ]
    stats = {}
    result = translator.translate(segments, "zh", "en", stats=stats)
    _assert_translation_quality(result)
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_GROK_TEST")),
    reason="SKIP_GROK_TEST environment variable is set",
)
def test_grok_translator():
    translator = Grok()
    segments = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in CHINESE_SEGMENTS
    ]
    stats = {}
    result = translator.translate(segments, "zh", "en", stats=stats)
    _assert_translation_quality(result)
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_OPENAI_TEST")),
    reason="SKIP_OPENAI_TEST environment variable is set",
)
def test_openai_translator():
    translator = OpenAI()
    segments = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in CHINESE_SEGMENTS
    ]
    stats = {}
    result = translator.translate(segments, "zh", "en", stats=stats)
    _assert_translation_quality(result)
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0
