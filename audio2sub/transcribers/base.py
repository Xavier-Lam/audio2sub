from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from audio2sub import Segment


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0

    def export(self, stats: Optional[dict]) -> None:
        if stats is None:
            return
        stats["tokens_in"] = self.tokens_in
        stats["tokens_out"] = self.tokens_out


class Base(ABC):
    """Base class for transcription backends."""

    name: str = "base"

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for CLI option registration."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Base":
        """Instantiate transcriber from CLI args."""
        return cls()  # pragma: no cover - overridden when needed

    @classmethod
    def opts_from_cli(cls, args: argparse.Namespace) -> dict:
        """Extract transcriber-specific options from CLI args."""
        return {}

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        """Transcribe a single audio segment and return text."""
        raise NotImplementedError

    def batch_transcribe(
        self,
        segments: List[Segment],
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> Iterable[Segment]:
        """Transcribe a list of segments. Yields updated segments."""
        for seg in segments:
            if seg.audio is None:
                raise FileNotFoundError("Segment has no audio path set")
            text = self.transcribe(str(seg.audio), lang=lang, stats=stats)
            seg.text = text
            yield seg


class AIAPITranscriber(Base, ABC):
    """Base class for AI API transcribers"""

    base_prompt: str = (
        "You will receive multiple audio clips. Return a JSON array of objects "
        "with `index` and `text` fields in the same order. Transcribe each "
        "clip verbatim (no paraphrasing). Omit non-speech clips or return "
        "empty text."
    )

    default_model: str = ""
    default_chunk: int = 20

    api_key_env_var: Optional[str] = None

    def __init__(self, model="", api_key=None) -> None:
        self.model = model or self.default_model
        self.api_key = api_key

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default=cls.default_model or None,
            help=(
                "Model name to use"
                + (f" (default: {cls.default_model})" if cls.default_model else "")
            ),
        )
        parser.add_argument(
            "--api-key",
            dest="api_key",
            required=False,
            help=(
                f"API key (optional; env {cls.api_key_env_var or 'API key env var'} "
                "is used if not provided)"
            ),
        )

        # Add batch transcription options
        parser.add_argument(
            "--chunk",
            type=int,
            default=cls.default_chunk,
            help=("Number of clips per API request " f"(default: {cls.default_chunk})"),
        )
        parser.add_argument(
            "--outline",
            dest="outline",
            required=False,
            help=("Context outline to guide transcription"),
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            required=False,
            help=("Additional system prompt/instructions"),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "AIAPITranscriber":
        return cls(model=args.model, api_key=args.api_key)

    @classmethod
    def opts_from_cli(cls, args: argparse.Namespace) -> dict:
        return {
            "chunk": args.chunk,
            "outline": args.outline,
            "prompt": args.prompt,
        }

    def transcribe(
        self, audio_path: str, lang: Optional[str] = None, stats: Optional[dict] = None
    ) -> str:
        segments_iter = self.batch_transcribe(
            [Segment(index=1, start=0, end=0, audio=Path(audio_path))],
            lang=lang,
            stats=stats,
        )
        segments = list(segments_iter)
        return segments[0].text if segments else ""

    def batch_transcribe(
        self,
        segments: List[Segment],
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
        chunk: Optional[int] = None,
        outline: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Iterable[Segment]:
        """Transcribe segments with shared chunking, prompt, and stats."""

        chunk_size = chunk if chunk and chunk > 0 else self.default_chunk
        prompt_cfg = self._build_prompt(lang=lang, outline=outline, prompt=prompt)
        client = self._ensure_client()
        usage_tracker = Usage()

        for batch in self._iter_chunks(segments, chunk_size):
            raw_text, usage = self._request_transcription(client, batch, prompt_cfg)
            self._parse_response_text(raw_text, batch)
            if usage:
                usage_tracker.tokens_in += usage.tokens_in
                usage_tracker.tokens_out += usage.tokens_out
                usage_tracker.export(stats)
            for seg in batch:
                yield seg

    def _ensure_client(self):
        if getattr(self, "_client", None):
            return self._client
        self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self):
        """Instantiate the API client."""

    @abstractmethod
    def _request_transcription(
        self,
        client,
        batch: List[Segment],
        prompt: List[str],
    ) -> Tuple[str, Optional[Usage]]:
        """Call the provider and return (raw_text_response, Usage)."""

    def _iter_chunks(self, items: List[Segment], size: int) -> Iterable[List[Segment]]:
        if size <= 0:
            size = len(items)
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _build_prompt(
        self,
        lang: Optional[str],
        outline: Optional[str],
        prompt: Optional[str],
    ) -> List[str]:
        prompt_text = self.base_prompt
        if lang:
            prompt_text += (
                " Primary language is "
                f"{lang}, but audio may include other languages."
            )
        prompt_text += (
            " Each object's `text` must be the transcription of that specific "
            "clip, with no labels or formatting. Respond as plain JSON text "
            "only; do not include markdown or code fences such as ``` or "
            "other wrappers."
        )
        prompt_text += (
            '\nRespond with JSON array of objects: [{"index": <clip '
            'index>, "text": <transcription>}, ...].'
        )

        system_prompts = [prompt_text]
        if outline:
            system_prompts.append(
                "Outline to guide transcription (context only). *Use the outline "
                "only to make minor corrections to what you hear in the audio "
                "(for example: fix homophones, obvious mis-hearings, or minor "
                "punctuation). Do NOT use the outline or any external knowledge "
                "to create or add words that are not present in the audio*:\n" + outline
            )
        if prompt:
            system_prompts.append("Additional instructions:\n" + prompt)

        # Return raw list of system prompts (was PromptConfig.system_prompts)
        return system_prompts

    def _parse_response_text(self, raw_text: str, batch: List[Segment]) -> None:
        raw_text = raw_text.strip()
        parsed: List[dict] = json.loads(raw_text)

        by_index = {
            entry.get("index"): entry
            for entry in parsed
            if isinstance(entry, dict) and "index" in entry
        }

        for idx, seg in enumerate(batch):
            entry = by_index.get(seg.index)
            if entry:
                seg.text = entry.get("text", "").strip()

    def _segments_to_audio_bytes(
        self, batch: List[Segment]
    ) -> List[Tuple[Segment, bytes]]:
        payloads: List[Tuple[Segment, bytes]] = []
        for seg in batch:
            if seg.audio is None:
                raise FileNotFoundError("Segment has no audio path set")
            audio_path = Path(seg.audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            payloads.append((seg, audio_path.read_bytes()))
        return payloads

    def _resolve_api_key(self) -> str:
        api_key = self.api_key
        if not api_key and self.api_key_env_var:
            api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            env_hint = self.api_key_env_var or "API key"
            raise RuntimeError(f"{env_hint} is required for {self.name} transcriber.")
        return api_key


class MissingDependencyException(RuntimeError):
    def __init__(self, transcriber) -> None:
        name = transcriber.name
        msg = (
            f"Transcriber '{name}' is not installed. Install with `pip install "
            f"audio2sub[{name}]`."
        )
        super().__init__(msg)
