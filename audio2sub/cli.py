from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Dict, Type
import warnings

import torch
from tqdm.auto import tqdm

from . import __version__, segments_to_srt, transcribe, transcribers
from .transcribers import Base
from .detectors import Silero


def _available_transcribers() -> Dict[str, Type[Base]]:
    return {
        obj.name: obj
        for _, obj in inspect.getmembers(transcribers, inspect.isclass)
        if issubclass(obj, Base) and not inspect.isabstract(obj)
    }


def _build_backend_parser(choices: list[str]) -> argparse.ArgumentParser:
    default = os.environ.get("DEFAULT_TRANSCRIBER", "faster_whisper")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-t",
        "--transcriber",
        choices=choices,
        default=default,
        help=f"Transcription backend to use (default: {default})",
    )
    return parser


def build_parser(
    available: Dict[str, Type[Base]], args=None
) -> argparse.ArgumentParser:
    backend_parser = _build_backend_parser(choices=sorted(available.keys()))
    backend_args, _remaining = backend_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(
        prog="audio2sub",
        description=(
            "Convert media files to SRT subtitles using FFmpeg, Silero VAD, "
            "and transcription backends."
        ),
        parents=[backend_parser],
    )

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
            "Language code (e.g., en, es, fr). If omitted, backend may default to en. "
            "See https://github.com/openai/whisper/blob/main/whisper/tokenizer.py for "
            "a list of available languages."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    Silero.contribute_to_cli(parser)
    available[backend_args.transcriber].contribute_to_cli(parser)
    return parser


def main() -> int:
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available; performance may be degraded significantly. "
            "For more information, please refer to the README.md of the project."
        )

    available = _available_transcribers()
    parser = build_parser(available)
    args = parser.parse_args()
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
    transcriber = transcriber_cls.from_cli_args(args)
    batch_opts = transcriber_cls.opts_from_cli(args)

    try:
        segments = transcribe(
            input_media,
            detector,
            transcriber,
            lang=args.lang,
            reporter=reporter,
            stats=stats,
            transcriber_opts=batch_opts,
        )
        segments_to_srt(segments).save(str(output_srt))
        print(f"SRT written to {output_srt}")
    finally:
        print("Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
