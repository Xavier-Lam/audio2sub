from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .base import Base, MissingDependencyException


class FasterWhisper(Base):
    """Transcriber using faster-whisper (ctranslate2 backend)."""

    name = "faster_whisper"

    def __init__(self, model_name: str = "turbo") -> None:
        self.model_name = model_name
        self._model = None

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default="turbo",
            help="Faster-Whisper model name (default: turbo)",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "FasterWhisper":
        return cls(model_name=args.model)

    def transcribe(
        self,
        audio_path: str,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        model = self._ensure_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        segments, _info = model.transcribe(
            str(audio_path),
            language=lang,
        )

        return " ".join(seg.text.strip() for seg in segments).strip()

    def _ensure_model(self):  # pragma: no cover - exercised in integration
        if self._model is not None:
            return self._model
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyException(self) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self.model_name, device=device, compute_type=compute_type
        )
        return self._model
