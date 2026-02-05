from __future__ import annotations

from pathlib import Path
from typing import List

import ffmpeg
import numpy as np
import torch

from audio2sub import Segment


class SileroVAD:
    """Thin wrapper around snakers4/silero-vad for speech timestamp detection."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        window_size_samples: int = 512,
        sample_rate: int = 16_000,
    ) -> None:
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.window_size_samples = window_size_samples
        self.sample_rate = sample_rate

    def detect_segments(self, wav_path: str | Path) -> List[Segment]:
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
        wav = torch.from_numpy(wav_np)

        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            window_size_samples=self.window_size_samples,
        )

        segments: List[Segment] = []
        for idx, ts in enumerate(timestamps, start=1):
            start = ts.get("start", 0) / self.sample_rate
            end = ts.get("end", 0) / self.sample_rate
            if end > start:
                segments.append(Segment(index=idx, start=start, end=end))
        return segments
