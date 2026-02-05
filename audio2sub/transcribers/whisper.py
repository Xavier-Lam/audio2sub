from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .base import Base, MissingDependencyException


class Whisper(Base):
    """Whisper-based transcriber (openai/whisper) for single audio segments."""

    name = "whisper"

    def __init__(self, model_name: str = "turbo") -> None:
        self.model_name = model_name
        self._model = None

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default="turbo",
            help="Whisper model name (default: turbo)",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Whisper":
        return cls(args.model)

    def transcribe(
        self,
        audio_path: str,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        model, whisper = self._ensure_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

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
            import torch
            import whisper
        except ImportError as exc:
            raise MissingDependencyException(self) from exc

        if self._model is not None:
            return self._model, whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = whisper.load_model(self.model_name, device=device)
        return self._model, whisper
