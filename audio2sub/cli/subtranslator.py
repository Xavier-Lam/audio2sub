from __future__ import annotations

from pathlib import Path

from .. import translators
from ..common import segments_to_srt, srt_to_segments
from .base import BaseCLI


class SubTranslatorCLI(BaseCLI):
    """CLI for subtranslator: translate subtitle files."""

    prog = "subtranslator"
    description = (
        "Translate subtitle files between languages using AI-powered "
        "translation backends."
    )
    backend_arg_name = "translator"
    backend_short_flag = "-t"
    default_backend_env = "DEFAULT_TRANSLATOR"
    default_backend = "gemini"
    backend_module = translators
    backend_base_class = translators.Base

    def _add_arguments(self, parser):
        parser.add_argument("input", help="Path to input SRT subtitle file")
        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Output SRT file path",
        )
        parser.add_argument(
            "-s",
            "--src-lang",
            required=True,
            help="Source language code (e.g., zh, en, es, fr)",
        )
        parser.add_argument(
            "-d",
            "--dst-lang",
            required=True,
            help="Destination language code (e.g., en, zh, fr, ja)",
        )

    def _run(self, args, available) -> int:
        input_srt = Path(args.input)
        output_srt = Path(args.output)

        if not input_srt.exists():
            raise FileNotFoundError(f"Input file not found: {input_srt}")

        segments = srt_to_segments(input_srt)
        if not segments:
            raise RuntimeError("No subtitle segments found in input file.")

        translator_cls = available[args.translator]
        translator_inst = translator_cls.from_cli_args(args)
        opts = translator_cls.opts_from_cli(args)

        stats = {}
        try:
            translated = translator_inst.translate(
                segments,
                args.src_lang,
                args.dst_lang,
                stats=stats,
                **opts,
            )
            segments_to_srt(translated).save(str(output_srt))
            print(f"Translated SRT written to {output_srt}")
        finally:
            if stats:
                print("Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
        return 0


def main() -> int:
    """Entry point for subtranslator."""
    return SubTranslatorCLI().run()
