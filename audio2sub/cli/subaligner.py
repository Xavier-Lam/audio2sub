from __future__ import annotations

from pathlib import Path

from .. import aligners
from ..common import segments_to_srt, srt_to_segments
from .base import BaseCLI


class SubAlignerCLI(BaseCLI):
    """CLI for subaligner: align subtitle timing."""

    prog = "subaligner"
    description = (
        "Align subtitle timing to match a reference subtitle file "
        "using AI-powered alignment backends."
    )
    backend_arg_name = "aligner"
    backend_short_flag = "-a"
    default_backend_env = "DEFAULT_ALIGNER"
    default_backend = "gemini"
    backend_module = aligners
    backend_base_class = aligners.Base

    def _add_arguments(self, parser):
        parser.add_argument(
            "-i",
            "--input",
            required=True,
            dest="input",
            help="Path to input SRT subtitle file to align",
        )
        parser.add_argument(
            "-r",
            "--reference",
            required=True,
            help="Path to reference SRT subtitle file with correct timing",
        )
        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Output SRT file path",
        )
        parser.add_argument(
            "--src-lang",
            default=None,
            help="Language of the input subtitle (e.g., zh, en)",
        )
        parser.add_argument(
            "--ref-lang",
            default=None,
            help="Language of the reference subtitle (e.g., en, zh)",
        )

    def _run(self, args, available) -> int:
        input_srt = Path(args.input)
        reference_srt = Path(args.reference)
        output_srt = Path(args.output)

        if not input_srt.exists():
            raise FileNotFoundError(f"Input file not found: {input_srt}")
        if not reference_srt.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_srt}")

        segments = srt_to_segments(input_srt)
        reference = srt_to_segments(reference_srt)

        if not segments:
            raise RuntimeError("No subtitle segments found in input file.")
        if not reference:
            raise RuntimeError("No subtitle segments found in reference file.")

        aligner_cls = available[args.aligner]
        aligner_inst = aligner_cls.from_cli_args(args)
        opts = aligner_cls.opts_from_cli(args)

        stats = {}
        try:
            aligned = aligner_inst.align(
                segments,
                reference,
                src_lang=args.src_lang,
                ref_lang=args.ref_lang,
                stats=stats,
                **opts,
            )
            segments_to_srt(aligned).save(str(output_srt))
            print(f"Aligned SRT written to {output_srt}")
        finally:
            if stats:
                print("Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
        return 0


def main() -> int:
    """Entry point for subaligner."""
    return SubAlignerCLI().run()
