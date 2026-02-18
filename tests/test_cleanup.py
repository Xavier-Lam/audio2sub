"""Tests that transcription subprocesses exit cleanly (no crash).

The exit-code bug (Windows STATUS_STACK_BUFFER_OVERRUN 0xC0000409 /
-1073740791) is caused by ctranslate2's C++ thread-pool teardown during
``Py_FinalizeEx``.  It only manifests when enough VAD segments are
transcribed (roughly > 20 segments, i.e. ~55 s of speech).

The fix is an ``atexit.register(os._exit, 0)`` registered in
``FasterWhisper._ensure_model()`` that terminates the process before the
native DLL unload phase.

These tests run a minimal transcription in a **subprocess** and assert
that the process exits with code 0.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent
SHORT_AUDIO = _TESTS_DIR / "jfk.flac"
LONG_AUDIO = _TESTS_DIR / "jfk_long.flac"


def _ensure_long_audio() -> Path:
    """Create a ~55 s test file by looping jfk.flac (if it doesn't exist)."""
    if LONG_AUDIO.exists():
        return LONG_AUDIO
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "4",
            "-i",
            str(SHORT_AUDIO),
            "-t",
            "60",
            str(LONG_AUDIO),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return LONG_AUDIO


# --- code templates (formatted with {audio}) ---------------------------------

_FASTER_WHISPER = textwrap.dedent(
    """\
    from pathlib import Path
    from audio2sub.detectors import Silero
    from audio2sub.transcribers import FasterWhisper
    from audio2sub.transcribe import transcribe

    segments = transcribe(
        Path(r"{audio}"),
        Silero(),
        FasterWhisper(model_name="tiny.en"),
        lang="en",
    )
    assert len(segments) > 0
"""
)

_WHISPER = textwrap.dedent(
    """\
    from pathlib import Path
    from audio2sub.detectors import Silero
    from audio2sub.transcribers import Whisper
    from audio2sub.transcribe import transcribe

    segments = transcribe(
        Path(r"{audio}"),
        Silero(),
        Whisper(model_name="tiny.en"),
        lang="en",
    )
    assert len(segments) > 0
"""
)


def test_faster_whisper_clean_exit():
    """FasterWhisper + long audio must exit 0 (regression for ctranslate2 crash)."""
    audio = _ensure_long_audio()
    code = _FASTER_WHISPER.format(audio=audio)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"faster_whisper exited with code {result.returncode}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )


def test_whisper_clean_exit():
    """OpenAI Whisper + short audio must exit 0 (sanity check)."""
    code = _WHISPER.format(audio=SHORT_AUDIO)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"whisper exited with code {result.returncode}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
