from __future__ import annotations

from pathlib import Path
from typing import Optional

from audio2sub.common import MissingDependencyException
from .base import WhisperBase


class FasterWhisper(WhisperBase):
    """Transcriber using faster-whisper (ctranslate2 backend)."""

    name = "faster_whisper"

    def _transcribe(
        self,
        audio_path: Path,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        model = self._ensure_model()

        segments, _info = model.transcribe(
            str(audio_path),
            language=lang,
        )

        return " ".join(seg.text.strip() for seg in segments).strip()

    def _ensure_model(self):  # pragma: no cover - exercised in integration
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyException(self) from exc

        device = self._get_device()
        compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self.model_name, device=device, compute_type=compute_type
        )
        return self._model
