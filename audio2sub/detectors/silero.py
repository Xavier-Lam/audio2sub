from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import ffmpeg
import numpy as np
import torch

from audio2sub.common import ReporterCallback, Segment

from .base import Base


class Silero(Base):
    """Silero VAD detector for speech timestamp detection using snakers4/silero-vad."""

    name: str = "silero"

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = 30,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        min_silence_at_max_speech: int = 98,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_silence_at_max_speech = min_silence_at_max_speech

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        """Add Silero VAD-specific CLI arguments."""
        parser.add_argument(
            "--silero-threshold",
            type=float,
            default=0.5,
            help=(
                "Speech threshold (0.0-1.0). Higher means more selective "
                "(default: 0.5)"
            ),
        )
        parser.add_argument(
            "--min-speech-duration",
            type=int,
            default=250,
            help="Minimum duration of speech chunk in milliseconds (default: 250)",
        )
        parser.add_argument(
            "--max-speech-duration",
            type=float,
            default=30,
            help="Maximum duration of speech chunk in seconds (default: 30)",
        )
        parser.add_argument(
            "--min-silence-duration",
            type=int,
            default=100,
            help=(
                "Minimum silence duration between speech chunks in "
                "milliseconds (default: 100)"
            ),
        )
        parser.add_argument(
            "--speech-pad",
            type=int,
            default=30,
            help="Padding to add to speech chunks in milliseconds (default: 30)",
        )
        parser.add_argument(
            "--min-silence-at-max-speech",
            type=int,
            default=98,
            help=(
                "Minimum silence duration at max speech duration for chunk split "
                "in milliseconds (default: 98)"
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Silero":
        """Instantiate SileroVAD from CLI args."""
        return cls(
            threshold=args.silero_threshold,
            min_speech_duration_ms=args.min_speech_duration,
            max_speech_duration_s=args.max_speech_duration,
            min_silence_duration_ms=args.min_silence_duration,
            speech_pad_ms=args.speech_pad,
            min_silence_at_max_speech=args.min_silence_at_max_speech,
        )

    def detect(
        self,
        wav_path: str | Path,
        reporter: Optional[ReporterCallback] = None,
    ) -> List[Segment]:
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            get_speech_timestamps = utils[0]
        except Exception as exc:
            raise RuntimeError(f"Failed to load silero-vad: {exc}") from exc

        # Read WAV via ffmpeg pipe (float32 mono at target sample rate)
        wav = self._read_audio(wav_path)

        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            min_silence_at_max_speech=self.min_silence_at_max_speech,
            progress_tracking_callback=lambda p: reporter
            and reporter(
                "progress",
                name="VAD",
                current=math.floor(min(100.0, p) * 100) / 100.0,
                total=100.0,
                unit="%",
            ),
        )

        segments: List[Segment] = []
        for idx, ts in enumerate(timestamps, start=1):
            start = ts.get("start", 0) / self.sample_rate
            end = ts.get("end", 0) / self.sample_rate
            if end > start:
                segments.append(Segment(index=idx, start=start, end=end))
        return segments

    def _read_audio(self, wav_path: str | Path) -> torch.Tensor:
        """Reads audio file and returns a mono float32 tensor at target sample rate."""
        process = (
            ffmpeg.input(str(wav_path))
            .output(
                "pipe:",
                format="f32le",
                ac=1,
                ar=self.sample_rate,
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio_bytes, stderr = process

        wav_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        if wav_np.size == 0:
            raise RuntimeError("No audio data decoded from WAV.")
        return torch.from_numpy(wav_np)
