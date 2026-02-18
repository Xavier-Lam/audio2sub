from __future__ import annotations

from typing import List, Optional, Tuple

from audio2sub.ai import GeminiMixin
from audio2sub.common import Segment, Usage
from .base import AIAPITranscriber


class Gemini(GeminiMixin, AIAPITranscriber):
    """Transcriber using Gemini API (google-genai)."""

    def _request_transcription(
        self,
        client,
        batch: List[Segment],
        prompt: List[str],
        retries: Optional[int] = None,
    ) -> Tuple[str, Optional[Usage]]:
        parts = [{"text": "\n\n".join(prompt)}]
        parts.extend(self._build_parts(batch=batch))
        contents = [{"role": "user", "parts": parts}]
        return self._call(client, contents, retries=retries)

    def _build_parts(self, batch: List[Segment]) -> List[dict]:
        parts: List[dict] = []
        for seg, audio_bytes in self._segments_to_audio_bytes(batch):
            parts.append({"text": f"Clip {seg.index}"})
            parts.append(
                {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}}
            )
        return parts
