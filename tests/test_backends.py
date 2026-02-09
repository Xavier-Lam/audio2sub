from pathlib import Path
from difflib import SequenceMatcher
import os
import pytest

from audio2sub import transcribe, transcribers
from audio2sub.detectors import Silero


SAMPLE_AUDIO = Path(__file__).parent / "jfk.flac"

EXPECTED = [
    {"text": "and so, my fellow americans,", "start": 0.0, "end": 3.0},
    {"text": "ask not", "start": 3.0, "end": 5.0, "ratio": 0.6},
    {"text": "what your country can do for you", "start": 5.0, "end": 8.0},
    {"text": "but what you can do for your country.", "start": 8.0, "end": 11.0},
]


def _assert_segments_match(segments):
    assert len(segments) == 4
    for exp, act in zip(EXPECTED, segments):
        ratio = SequenceMatcher(None, exp["text"].lower(), act.text.lower()).ratio()
        msg = (
            f"Text mismatch (ratio={ratio:.3f}):\n"
            f"  expected: '{exp['text']}'\n"
            f"  actual:   '{act.text}'"
        )
        assert ratio >= exp.get("ratio", 0.7), msg
        start_diff = abs(act.start - exp["start"]) <= 1.0
        assert (
            start_diff
        ), f"Start time mismatch:\n  expected: {exp['start']}\n  actual:   {act.start}"
        end_diff = abs(act.end - exp["end"]) <= 1.0
        assert (
            end_diff
        ), f"End time mismatch:\n  expected: {exp['end']}\n  actual:   {act.end}"


def test_whisper_backend():
    segments = transcribe(
        SAMPLE_AUDIO, Silero(), transcribers.Whisper(model_name="tiny.en"), lang="en"
    )
    _assert_segments_match(segments)


def test_faster_whisper_backend():
    segments = transcribe(
        SAMPLE_AUDIO,
        Silero(),
        transcribers.FasterWhisper(model_name="tiny.en"),
        lang="en",
    )
    _assert_segments_match(segments)


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_GEMINI_TEST")),
    reason="SKIP_GEMINI_TEST environment variable is set",
)
def test_gemini_backend():
    stats = {}
    segments = transcribe(
        SAMPLE_AUDIO, Silero(), transcribers.Gemini(), lang="en", stats=stats
    )
    _assert_segments_match(segments)
    assert "tokens_in" in stats or "tokens_out" in stats
    assert stats.get("tokens_in", 0) > 0
    assert stats.get("tokens_out", 0) > 0
