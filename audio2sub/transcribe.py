"""Audio-to-subtitle transcription pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

from . import detectors, transcribers
from .audio import convert_media_to_wav, cut_wav_segment
from .common import ReporterCallback, Segment, Usage


def transcribe(
    input_media: str | Path,
    detector: detectors.Base,
    transcriber: transcribers.Base,
    lang: Optional[str] = None,
    reporter: Optional[ReporterCallback] = None,
    stats: Optional[Usage | dict] = None,
    transcriber_opts: Optional[dict] = None,
) -> List[Segment]:
    """Convert media to segments using VAD and batch transcription."""

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

        _output("Running voice activity detection (VAD)...")
        segments = detector.detect(wav_path, reporter=reporter)
        if not segments:
            raise RuntimeError("No speech detected by VAD")
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
            segments, lang=lang, stats=stats, **(transcriber_opts or {})
        ):
            if seg.text.strip():
                transcribed_segments.append(seg)
            completed += 1
            _progress("transcription", completed, total_segments, unit="seg")

        if len(transcribed_segments) == 0:
            raise RuntimeError("Transcription produced no subtitle lines.")

        _output("Transcription completed.")
        return transcribed_segments
