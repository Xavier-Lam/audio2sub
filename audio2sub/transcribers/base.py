from __future__ import annotations

import json
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import warnings

from audio2sub.ai import AIBackendBase
from audio2sub.common import MissingDependencyException, Segment, Usage


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


class AIAPITranscriber(AIBackendBase, Base, ABC):
    """Base class for AI API transcribers."""

    base_prompt: str = (
        "You will receive multiple audio clips. Return a JSON array of objects "
        "with `index` and `text` fields in the same order. Transcribe each "
        "clip verbatim (no paraphrasing). Omit non-speech clips or return "
        "empty text."
    )

    default_chunk: int = 40

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        super().contribute_to_cli(parser)
        parser.add_argument(
            "--outline",
            dest="outline",
            required=False,
            help="Context outline to guide transcription",
        )

    @classmethod
    def opts_from_cli(cls, args: argparse.Namespace) -> dict:
        opts = super().opts_from_cli(args)
        opts["outline"] = args.outline
        return opts

    def transcribe(
        self,
        audio_path: str,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
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
        retries: Optional[int] = None,
    ) -> Iterable[Segment]:
        """Transcribe segments with shared chunking, prompt, and stats."""
        chunk_size = chunk if chunk and chunk > 0 else self.default_chunk
        prompt_cfg = self._build_prompt(lang=lang, outline=outline, prompt=prompt)
        client = self._ensure_client()
        usage_tracker = Usage()

        for batch in self._iter_chunks(segments, chunk_size):
            raw_text, usage = self._request_transcription(
                client, batch, prompt_cfg, retries=retries
            )
            self._parse_response_text(raw_text, batch)
            if usage:
                usage_tracker.tokens_in += usage.tokens_in
                usage_tracker.tokens_out += usage.tokens_out
                usage_tracker.export(stats)
            for seg in batch:
                yield seg

    @abstractmethod
    def _request_transcription(
        self,
        client,
        batch: List[Segment],
        prompt: List[str],
        retries: Optional[int] = None,
    ) -> Tuple[str, Optional[Usage]]:
        """Call the provider and return (raw_text_response, Usage)."""

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
                "Outline to guide transcription (context only). *Use the "
                "outline only to make minor corrections to what you hear in "
                "the audio (for example: fix homophones, obvious "
                "mis-hearings, or minor punctuation). Do NOT use the outline "
                "or any external knowledge to create or add words that are "
                "not present in the audio*:\n" + outline
            )
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


class WhisperBase(Base, ABC):
    """Base class for Whisper-based transcribers"""

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
    def from_cli_args(cls, args: argparse.Namespace) -> "WhisperBase":
        return cls(model_name=args.model)

    def transcribe(
        self,
        audio_path: str,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        """Transcribe a single audio segment and return text."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        return self._transcribe(audio_path, lang, stats)

    @abstractmethod
    def _transcribe(
        self,
        audio_path: Path,
        lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> str:
        """Implementation-specific transcription logic."""
        raise NotImplementedError

    @abstractmethod
    def _ensure_model(self):
        """Lazy-load and return the Whisper model instance."""
        raise NotImplementedError

    def _get_device(self) -> str:
        """Detect and return the appropriate device (cuda or cpu)."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            warnings.warn(
                "CUDA is not available; performance may be degraded "
                "significantly. For more information, please refer to the "
                "README.md of the project."
            )
            return "cpu"
        except ImportError as exc:
            raise MissingDependencyException(self) from exc
