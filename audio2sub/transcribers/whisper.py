from __future__ import annotations

from pathlib import Path
from typing import Optional

from audio2sub.common import MissingDependencyException
from .base import WhisperBase


class Whisper(WhisperBase):
    """Whisper-based transcriber (openai/whisper) for single audio segments."""

    name = "whisper"

    def _transcribe(
        self,
        audio_path: Path,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        model, whisper = self._ensure_model()

        audio = whisper.load_audio(str(audio_path))
        result = model.transcribe(
            audio,
            language=lang or "en",
            task="transcribe",
            fp16=model.device.type == "cuda",
        )
        text = result.get("text", "")
        return str(text).strip()

    def _ensure_model(self):
        try:
            import whisper
        except ImportError as exc:
            raise MissingDependencyException(self) from exc

        if self._model is not None:
            return self._model, whisper

        device = self._get_device()
        self._model = whisper.load_model(self.model_name, device=device)
        return self._model, whisper
