from __future__ import annotations

from typing import List, Optional, Tuple

from audio2sub.common import Usage
from audio2sub.ai import GeminiMixin
from .base import AITranslator


class Gemini(GeminiMixin, AITranslator):
    """Translator using Gemini API (google-genai)."""

    def _request(
        self,
        client,
        input_data: List[dict],
        prompt: List[str],
        retries: Optional[int] = None,
    ) -> Tuple[str, Optional[Usage]]:
        return self._call_text(client, prompt, input_data, retries=retries)
