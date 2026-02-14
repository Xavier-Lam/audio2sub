from __future__ import annotations

import json
import argparse
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from audio2sub.ai import AIBackendBase
from audio2sub.common import Segment, Usage


class Base(ABC):
    """Base class for translation backends."""

    name: str = "base"

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for CLI option registration."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Base":
        """Instantiate translator from CLI args."""
        return cls()  # pragma: no cover - overridden when needed

    @classmethod
    def opts_from_cli(cls, args: argparse.Namespace) -> dict:
        """Extract translator-specific options from CLI args."""
        return {}

    @abstractmethod
    def translate(
        self,
        segments: List[Segment],
        src_lang: str,
        dst_lang: str,
        stats: Optional[dict] = None,
    ) -> List[Segment]:
        """Translate segments from source to destination language."""
        raise NotImplementedError


class AITranslator(AIBackendBase, Base, ABC):
    """Base class for AI-based translators."""

    base_prompt: str = (
        "You are a professional subtitle translator. You will receive subtitle "
        "segments as a JSON array of objects with `index` and `text` fields. "
        "Translate each segment's text from {src_lang} to {dst_lang}. "
        "Return a JSON array of objects with `index` and `text` fields in the "
        "same order. Preserve the meaning, tone, and natural expression. "
        "Do not add, remove, or merge segments."
    )

    def translate(
        self,
        segments: List[Segment],
        src_lang: str,
        dst_lang: str,
        stats: Optional[dict] = None,
        chunk: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> List[Segment]:
        """Translate segments using AI API with chunking support."""
        chunk_size = chunk if chunk and chunk > 0 else self.default_chunk
        prompt_cfg = self._build_prompt(
            src_lang=src_lang, dst_lang=dst_lang, prompt=prompt
        )
        client = self._ensure_client()
        usage_tracker = Usage()

        result: List[Segment] = []
        for batch in self._iter_chunks(segments, chunk_size):
            input_data = [{"index": seg.index, "text": seg.text} for seg in batch]
            raw_text, usage = self._request(client, input_data, prompt_cfg)
            self._parse_response_text(raw_text, batch)
            if usage:
                usage_tracker.tokens_in += usage.tokens_in
                usage_tracker.tokens_out += usage.tokens_out
                usage_tracker.export(stats)
            result.extend(batch)
        return result

    @abstractmethod
    def _request(
        self,
        client,
        input_data: List[dict],
        prompt: List[str],
    ) -> Tuple[str, Optional[Usage]]:
        """Call the API and return (raw_text, Usage)."""

    def _build_prompt(
        self,
        src_lang: str,
        dst_lang: str,
        prompt: Optional[str] = None,
    ) -> List[str]:
        prompt_text = self.base_prompt.format(src_lang=src_lang, dst_lang=dst_lang)
        prompt_text += (
            " Each object's `text` must be the translation of the "
            "corresponding input segment. Respond as plain JSON text "
            "only; do not include markdown or code fences such as ``` "
            "or other wrappers."
        )
        prompt_text += (
            '\nRespond with JSON array of objects: [{"index": '
            '<segment index>, "text": <translated text>}, ...].'
        )

        system_prompts = [prompt_text]
        if prompt:
            system_prompts.append("Additional instructions:\n" + prompt)
        return system_prompts

    def _parse_response_text(self, raw_text: str, batch: List[Segment]) -> None:
        raw_text = raw_text.strip()
        parsed: List[dict] = json.loads(raw_text)

        by_index = {
            entry.get("index"): entry
            for entry in parsed
            if isinstance(entry, dict) and "index" in entry
        }

        for seg in batch:
            entry = by_index.get(seg.index)
            if entry:
                seg.text = entry.get("text", "").strip()


class TraditionalTranslator(Base, ABC):
    """Base class for traditional translation tools
    (DeepL, Google Translate, etc.)."""

    pass
