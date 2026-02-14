from __future__ import annotations

from typing import List, Optional, Tuple

from audio2sub.ai import OpenAIMixin
from audio2sub.common import Usage
from .base import AITranslator


class OpenAI(OpenAIMixin, AITranslator):
    """Translator using OpenAI API."""

    def _request(
        self,
        client,
        input_data: List[dict],
        prompt: List[str],
    ) -> Tuple[str, Optional[Usage]]:
        return self._call_text(client, prompt, input_data)
