from __future__ import annotations

import json
import argparse
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple

from audio2sub.common import Segment, Usage
from audio2sub.ai import AIBackendBase


class Base(ABC):
    """Base class for subtitle alignment backends."""

    name: str = "base"

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for CLI option registration."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Base":
        """Instantiate aligner from CLI args."""
        return cls()  # pragma: no cover - overridden when needed

    @classmethod
    def opts_from_cli(cls, args: argparse.Namespace) -> dict:
        """Extract aligner-specific options from CLI args."""
        return {}

    @abstractmethod
    def align(
        self,
        segments: List[Segment],
        reference: List[Segment],
        src_lang: Optional[str] = None,
        ref_lang: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> List[Segment]:
        """Align segments to match reference subtitle timing."""
        raise NotImplementedError


class AIAligner(AIBackendBase, Base, ABC):
    """Base class for AI-based subtitle aligners."""

    base_prompt: str = (
        "You are a professional subtitle alignment tool. You will receive two "
        "sets of subtitles as JSON:\n"
        '1. "segments": Subtitles whose timing needs to be corrected.\n'
        '2. "reference": Reference subtitles with correct timing.\n\n'
        "The two sets correspond to the same content but may be in different "
        "languages. Your task is to assign correct timing from the reference "
        "subtitles to the segments.\n\n"
        "Return a JSON array of objects with `index`, `start`, `end`, and "
        "`text` fields:\n"
        "- `index`: the index from the segments to align\n"
        "- `text`: the text from the segments to align (unchanged)\n"
        "- `start` and `end`: aligned timing in seconds (as float numbers)\n"
    )

    def align(
        self,
        segments: List[Segment],
        reference: List[Segment],
        src_lang: Optional[str] = None,
        ref_lang: Optional[str] = None,
        stats: Optional[dict] = None,
        chunk: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> List[Segment]:
        """Align segments to reference timing using AI API."""
        prompt_cfg = self._build_prompt(
            src_lang=src_lang, ref_lang=ref_lang, prompt=prompt
        )
        client = self._ensure_client()
        usage_tracker = Usage()
        chunk_size = chunk if chunk and chunk > 0 else self.default_chunk

        result: List[Segment] = []
        for seg_batch, ref_batch in self._iter_alignment_chunks(
            segments, reference, chunk_size
        ):
            seg_data = [{"index": s.index, "text": s.text} for s in seg_batch]
            ref_data = [
                {
                    "index": r.index,
                    "start": r.start,
                    "end": r.end,
                    "text": r.text,
                }
                for r in ref_batch
            ]
            input_data = {
                "segments": seg_data,
                "reference": ref_data,
            }
            raw_text, usage = self._request(client, input_data, prompt_cfg)
            self._parse_response_text(raw_text, seg_batch)
            if usage:
                usage_tracker.tokens_in += usage.tokens_in
                usage_tracker.tokens_out += usage.tokens_out
                usage_tracker.export(stats)
            result.extend(seg_batch)
        return result

    @abstractmethod
    def _request(
        self,
        client,
        input_data: dict,
        prompt: List[str],
    ) -> Tuple[str, Optional[Usage]]:
        """Call the API and return (raw_text, Usage)."""

    def _iter_alignment_chunks(
        self,
        segments: List[Segment],
        reference: List[Segment],
        chunk_size: int,
    ) -> Iterable[Tuple[List[Segment], List[Segment]]]:
        """Chunk segments for alignment, keeping full reference."""
        if chunk_size <= 0 or chunk_size >= len(segments):
            yield segments, reference
            return
        for i in range(0, len(segments), chunk_size):
            seg_batch = segments[i : i + chunk_size]
            yield seg_batch, reference

    def _build_prompt(
        self,
        src_lang: Optional[str] = None,
        ref_lang: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> List[str]:
        prompt_text = self.base_prompt
        if src_lang:
            prompt_text += f"The segments to align are in {src_lang}. "
        if ref_lang:
            prompt_text += f"The reference subtitles are in {ref_lang}. "
        prompt_text += (
            "Respond as plain JSON text only; do not include markdown "
            "or code fences such as ``` or other wrappers."
        )
        prompt_text += (
            '\nRespond with JSON array: [{"index": <index>, '
            '"start": <seconds>, "end": <seconds>, "text": '
            "<original text>}, ...]."
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
                seg.start = float(entry.get("start", seg.start))
                seg.end = float(entry.get("end", seg.end))
                if "text" in entry:
                    seg.text = entry["text"]
