from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from audio2sub import ReporterCallback, Segment


class Base(ABC):
    """Base class for voice activity detection backends."""

    name: str = "base"

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for CLI option registration."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Base":
        """Instantiate detector from CLI args."""
        return cls()  # pragma: no cover - overridden when needed

    @abstractmethod
    def detect(
        self,
        wav_path: str | Path,
        reporter: Optional[ReporterCallback] = None,
    ) -> List[Segment]:
        """Detect speech segments in audio file and return list of segments."""
        raise NotImplementedError
