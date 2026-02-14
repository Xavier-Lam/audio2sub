from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pysrt


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str = ""
    audio: Optional[Path] = None


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0

    def export(self, stats: Optional[dict]) -> None:
        if stats is None:
            return
        stats["tokens_in"] = self.tokens_in
        stats["tokens_out"] = self.tokens_out


class MissingDependencyException(RuntimeError):
    def __init__(self, backend) -> None:
        name = backend.name
        msg = (
            f"Backend '{name}' is not installed. Install with "
            f"`pip install audio2sub[{name}]`."
        )
        super().__init__(msg)


ReporterCallback = Callable[[str, dict], None]


def segments_to_srt(segments: Iterable[Segment]) -> pysrt.SubRipFile:
    srt = pysrt.SubRipFile()
    for seg in segments:
        item = pysrt.SubRipItem(
            index=seg.index,
            start=pysrt.SubRipTime(seconds=seg.start),
            end=pysrt.SubRipTime(seconds=seg.end),
            text=seg.text,
        )
        srt.append(item)
    return srt


def srt_to_segments(srt_path: str | Path) -> List[Segment]:
    """Read an SRT file and return a list of Segment objects."""
    srt = pysrt.open(str(srt_path))
    segments: List[Segment] = []
    for item in srt:
        seg = Segment(
            index=item.index,
            start=item.start.ordinal / 1000.0,
            end=item.end.ordinal / 1000.0,
            text=item.text,
        )
        segments.append(seg)
    return segments
