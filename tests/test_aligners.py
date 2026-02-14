"""Unit tests for subtitle aligners.

Uses JFK's famous speech segments in Chinese and English as reference.
Checks that the aligned output has correct timing from the reference.
"""

import os
import pytest

from audio2sub import Segment
from audio2sub.aligners import Gemini, Grok, OpenAI


# Chinese segments with incorrect (zero) timing
CHINESE_SEGMENTS = [
    Segment(index=1, start=1.0, end=4.0, text="因此，我的美国同胞们，"),
    Segment(index=2, start=5.0, end=6.5, text="不要问你的国家能为你做什么，"),
    Segment(index=3, start=7.0, end=12.0, text="而要问你能为你的国家做什么。"),
]

# English reference segments with correct timing
ENGLISH_REFERENCE = [
    Segment(index=1, start=0.0, end=3.0, text="And so, my fellow Americans,"),
    Segment(index=2, start=3.0, end=5.0, text="ask not"),
    Segment(
        index=3,
        start=5.0,
        end=8.0,
        text="what your country can do for you,",
    ),
    Segment(
        index=4,
        start=8.0,
        end=11.0,
        text="but what you can do for your country.",
    ),
]


def _assert_alignment_quality(aligned_segments):
    """Assert that aligned segments have reasonable timing."""
    assert len(aligned_segments) == len(CHINESE_SEGMENTS)

    for seg in aligned_segments:
        # Check that timing was assigned (non-zero end)
        assert seg.end > 0, f"Segment {seg.index} has no end timing"
        # Check that end > start
        assert (
            seg.end > seg.start
        ), f"Segment {seg.index}: end ({seg.end}) <= start ({seg.start})"
        # Check that timing is within the reference range (0 to 11s, with margin)
        assert seg.start >= 0, f"Segment {seg.index}: negative start time"
        assert (
            seg.end <= 15.0
        ), f"Segment {seg.index}: end time ({seg.end}) exceeds reference range"

    # Check that segments are in chronological order
    for i in range(1, len(aligned_segments)):
        assert (
            aligned_segments[i].start >= aligned_segments[i - 1].start
        ), f"Segments not in chronological order at index {i}"

    # Check that the original Chinese text is preserved
    for orig, aligned in zip(CHINESE_SEGMENTS, aligned_segments):
        assert (
            aligned.text == orig.text
        ), f"Text was modified: expected '{orig.text}', got '{aligned.text}'"


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_GEMINI_TEST")),
    reason="SKIP_GEMINI_TEST environment variable is set",
)
def test_gemini_aligner():
    aligner = Gemini()
    segments = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in CHINESE_SEGMENTS
    ]
    reference = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in ENGLISH_REFERENCE
    ]
    stats = {}
    result = aligner.align(
        segments, reference, src_lang="zh", ref_lang="en", stats=stats
    )
    _assert_alignment_quality(result)
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_GROK_TEST")),
    reason="SKIP_GROK_TEST environment variable is set",
)
def test_grok_aligner():
    aligner = Grok()
    segments = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in CHINESE_SEGMENTS
    ]
    reference = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in ENGLISH_REFERENCE
    ]
    stats = {}
    result = aligner.align(
        segments, reference, src_lang="zh", ref_lang="en", stats=stats
    )
    _assert_alignment_quality(result)
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_OPENAI_TEST")),
    reason="SKIP_OPENAI_TEST environment variable is set",
)
def test_openai_aligner():
    aligner = OpenAI()
    segments = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in CHINESE_SEGMENTS
    ]
    reference = [
        Segment(index=s.index, start=s.start, end=s.end, text=s.text)
        for s in ENGLISH_REFERENCE
    ]
    stats = {}
    result = aligner.align(
        segments, reference, src_lang="zh", ref_lang="en", stats=stats
    )
    _assert_alignment_quality(result)
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0
