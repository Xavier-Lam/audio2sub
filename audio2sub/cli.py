from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import inspect
import os
from pathlib import Path
from typing import Dict, Type

from tqdm.auto import tqdm

from . import transcribers, __version__
from .common import segments_to_srt
from .detectors import Silero
from .transcribe import transcribe


class BaseCLI(ABC):
    """Base class for all CLI tools."""

    prog: str = ""
    description: str = ""
    backend_arg_name: str = ""
    backend_short_flag: str = ""
    default_backend_env: str = ""
    default_backend: str = ""
    backend_module = None
    backend_base_class: Type = None

    def _available_backends(self) -> Dict[str, Type]:
        module = self.backend_module
        base_cls = self.backend_base_class
        if module is None or base_cls is None:
            return {}
        return {
            obj.name: obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, base_cls) and not inspect.isabstract(obj)
        }

    def _build_backend_parser(self, choices: list[str]) -> argparse.ArgumentParser:
        default = os.environ.get(self.default_backend_env, self.default_backend)
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            self.backend_short_flag,
            f"--{self.backend_arg_name}",
            choices=choices,
            default=default,
            help=f"Backend to use (default: {default})",
        )
        return parser

    @abstractmethod
    def _add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add tool-specific arguments."""

    def _add_backend_args(
        self,
        parser: argparse.ArgumentParser,
        backend_args: argparse.Namespace,
        available: Dict[str, Type],
    ) -> None:
        """Add backend-specific CLI arguments."""
        backend_name = getattr(backend_args, self.backend_arg_name)
        available[backend_name].contribute_to_cli(parser)

    def build_parser(
        self, available: Dict[str, Type], args=None
    ) -> argparse.ArgumentParser:
        backend_parser = self._build_backend_parser(choices=sorted(available.keys()))
        backend_args, _ = backend_parser.parse_known_args(args)

        parser = argparse.ArgumentParser(
            prog=self.prog,
            description=self.description,
            parents=[backend_parser],
        )
        self._add_arguments(parser)
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
        )
        self._add_backend_args(parser, backend_args, available)
        return parser

    @abstractmethod
    def _run(self, args: argparse.Namespace, available: Dict[str, Type]) -> int:
        """Execute the tool logic."""

    def run(self) -> int:
        """Entry point: discover backends, parse args, run."""
        available = self._available_backends()
        parser = self.build_parser(available)
        args = parser.parse_args()
        return self._run(args, available)


class Audio2SubCLI(BaseCLI):
    """CLI for audio2sub: transcribe audio to subtitles."""

    prog = "audio2sub"
    description = (
        "Convert media files to SRT subtitles using FFmpeg, Silero VAD, "
        "and transcription backends."
    )
    backend_arg_name = "transcriber"
    backend_short_flag = "-t"
    default_backend_env = "DEFAULT_TRANSCRIBER"
    default_backend = "faster_whisper"
    backend_module = transcribers
    backend_base_class = transcribers.Base

    def _add_arguments(self, parser):
        parser.add_argument("input", help="Path to input media file (audio or video)")
        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Output SRT file path",
        )
        parser.add_argument(
            "--lang",
            default=None,
            help=(
                "Language code (e.g., en, es, fr). If omitted, backend may "
                "default to en. See "
                "https://github.com/openai/whisper/blob/main/whisper/"
                "tokenizer.py for a list of available languages."
            ),
        )

    def _add_backend_args(self, parser, backend_args, available):
        Silero.contribute_to_cli(parser)
        super()._add_backend_args(parser, backend_args, available)

    def _run(self, args, available) -> int:
        backend = args.transcriber
        input_media = Path(args.input)
        output_srt = Path(args.output)

        bars: dict[str, tqdm] = {}

        def reporter(kind: str, **payload):
            if kind == "status":
                print(payload.get("message", ""))
            if kind == "progress":
                name = payload.pop("name")
                current = payload.pop("current", 0)
                total = payload.pop("total", 0)
                bar = bars.get(name)
                if bar is None:
                    bar = tqdm(
                        total=total,
                        desc=name.capitalize(),
                        leave=True,
                        **payload,
                    )
                    bars[name] = bar
                bar.n = current
                bar.refresh()
                if current >= total:
                    bar.close()
                    bars.pop(name, None)

        stats = {}
        detector = Silero.from_cli_args(args)
        transcriber_cls = available[backend]
        transcriber_inst = transcriber_cls.from_cli_args(args)
        batch_opts = transcriber_cls.opts_from_cli(args)

        try:
            segments = transcribe(
                input_media,
                detector,
                transcriber_inst,
                lang=args.lang,
                reporter=reporter,
                stats=stats,
                transcriber_opts=batch_opts,
            )
            segments_to_srt(segments).save(str(output_srt))
            print(f"SRT written to {output_srt}")
        finally:
            if stats:
                print("Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
        return 0

def main() -> int:
    """Entry point for audio2sub."""
    return Audio2SubCLI().run()

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
