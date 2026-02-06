"""Audio2Sub package: convert media to subtitles."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional
from dataclasses import dataclass

import pysrt

__all__ = ["__version__", "transcribe", "segments_to_srt", "Segment", "Usage"]
__title__ = "audio2sub"
__description__ = "Transcribe media files to SRT subtitles."
__url__ = "https://github.com/Xavier-Lam/audio2sub"
__version__ = "0.1.1"
__author__ = "Xavier-Lam"
__author_email__ = "xavierlam7@hotmail.com"


ReporterCallback = Callable[[str, dict], None]


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str = ""
    audio: Optional[Path] = None


from .audio import convert_media_to_wav, cut_wav_segment  # noqa: E402
from .transcribers.base import Base, Usage  # noqa: E402
from .vad import SileroVAD  # noqa: E402


def transcribe(
    input_media: str | Path,
    transcriber: Base,
    lang: Optional[str] = None,
    reporter: Optional[ReporterCallback] = None,
    stats: Optional[Usage | dict] = None,
    opts: Optional[dict] = None,
) -> List[Segment]:
    """Convert media to segments using Silero VAD and batch transcription."""

    input_media = Path(input_media)
    if not input_media.exists():
        raise FileNotFoundError(f"Input media not found: {input_media}")

    _output = lambda message: reporter and reporter("status", message=message)
    _progress = lambda name, current, total, **payload: reporter and reporter(
        "progress",
        name=name,
        current=current,
        total=total,
        **payload,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        _output("Converting audio...")
        convert_media_to_wav(input_media, wav_path)

        vad = SileroVAD(sample_rate=16_000)
        _output("Running voice activity detection (VAD)...")
        segments = vad.detect_segments(wav_path, reporter=reporter)
        if not segments:
            raise RuntimeError("No speech detected by Silero VAD")
        total_segments = len(segments)
        _output((f"Detected {total_segments} speech segment(s)."))
        _output("Cutting audio into clips...")

        # Attach indices and extract audio clips for each segment
        for idx, seg in enumerate(segments, start=1):
            seg.index = idx
            seg_path = Path(tmpdir) / f"segment_{idx}.wav"
            cut_wav_segment(wav_path, seg.start, seg.end, seg_path)
            seg.audio = seg_path

        _output("Starting transcription...")
        _progress("transcription", 0, total_segments, unit="seg")

        # Batch transcribe for potential backend optimizations (generator)
        transcribed_segments: List[Segment] = []
        completed = 0
        for seg in transcriber.batch_transcribe(
            segments, lang=lang, stats=stats, **(opts or {})
        ):
            if seg.text.strip():
                transcribed_segments.append(seg)
            completed += 1
            _progress("transcription", completed, total_segments, unit="seg")

        if len(transcribed_segments) == 0:
            raise RuntimeError("Transcription produced no subtitle lines.")

        _output("Transcription completed.")
        return transcribed_segments


def segments_to_srt(segments: Iterable[Segment]) -> pysrt.SubRipFile:
    srt = pysrt.SubRipFile()
    for seg in segments:
        item = pysrt.SubRipItem(
            index=seg.index,
            start=pysrt.SubRipTime(seconds=seg.start),
            end=pysrt.SubRipTime(seconds=seg.end),
            text=seg.text,
        )
        srt.append(item)
    return srt
