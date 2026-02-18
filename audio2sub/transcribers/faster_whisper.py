from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Optional

from audio2sub.common import MissingDependencyException
from .base import WhisperBase

# Module-level references that prevent ctranslate2 models from being
# garbage-collected before process exit.  On Windows the C++ thread-pool
# teardown inside ctranslate2 triggers STATUS_STACK_BUFFER_OVERRUN
# (0xC0000409 / exit-code -1073740791) whenever the model is freed —
# whether during normal GC (e.g. a temporary FasterWhisper going out of
# scope) or during interpreter finalisation.
#
# Keeping a reference here ensures the model stays alive until module
# cleanup, which happens *after* atexit handlers.  The atexit handler
# calls ``os._exit(0)`` first, so the model is never actually torn down.
#
# https://github.com/SYSTRAN/faster-whisper/issues/1293
_atexit_registered = False
_models: list = []  # prevent GC of ctranslate2 models


def _register_atexit() -> None:
    """Register ``os._exit(0)`` to run at interpreter shutdown."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(os._exit, 0)
        _atexit_registered = True


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
        # Pin model at module level so GC of the FasterWhisper instance
        # does not free the underlying ctranslate2 model.
        _models.append(self._model)
        _register_atexit()
        return self._model
