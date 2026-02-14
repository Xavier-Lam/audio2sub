from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

try:
    from .. import transcribers, detectors
except ImportError:
    print(
        "Error: Missing dependencies for audio2sub. Please install with "
        "`pip install audio2sub[backend]`. See the documentation for more details"
    )
    raise
from ..common import segments_to_srt
from ..transcribe import transcribe
from .base import BaseCLI


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
        detectors.Silero.contribute_to_cli(parser)
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
        detector = detectors.Silero.from_cli_args(args)
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
