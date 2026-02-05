from __future__ import annotations

from typing import List, Optional, Tuple

from audio2sub import Segment
from .base import AIAPITranscriber, MissingDependencyException, Usage


class Gemini(AIAPITranscriber):
    """Transcriber using Gemini API (google-genai)."""

    name = "gemini"
    default_model = "gemini-2.5-flash"
    api_key_env_var = "GEMINI_API_KEY"

    def _create_client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise MissingDependencyException(self) from exc

        api_key = self._resolve_api_key()
        return genai.Client(api_key=api_key)

    def _request_transcription(
        self,
        client,
        batch: List[Segment],
        prompt: List[str],
    ) -> Tuple[str, Optional[Usage]]:
        parts = [{"text": "\n\n".join(prompt)}]
        parts.extend(self._build_parts(batch=batch))
        contents = [{"role": "user", "parts": parts}]

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        raw_text = response.text.strip() if hasattr(response, "text") else ""

        usage = Usage(
            tokens_in=getattr(response.usage_metadata, "prompt_token_count", 0),
            tokens_out=getattr(response.usage_metadata, "candidates_token_count", 0),
        )
        return raw_text, usage

    def _build_parts(self, batch: List[Segment]) -> List[dict]:
        parts: List[dict] = []
        for seg, audio_bytes in self._segments_to_audio_bytes(batch):
            parts.append({"text": f"Clip {seg.index}"})
            parts.append(
                {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}}
            )

        return parts
