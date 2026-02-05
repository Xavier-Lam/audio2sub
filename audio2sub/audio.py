from pathlib import Path

import ffmpeg


def convert_media_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16_000,
    channels: int = 1,
    overwrite: bool = True,
):
    """Convert any media file to a WAV"""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    stream = ffmpeg.input(str(input_path)).output(
        str(output_path),
        ac=channels,
        ar=sample_rate,
        format="wav",
    )
    if overwrite:
        stream = stream.overwrite_output()
    else:
        stream = stream.global_args("-n")
    stream.run(quiet=True)


def cut_wav_segment(
    input_wav: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
):
    """Cut a WAV segment using ffmpeg"""

    input_wav = Path(input_wav)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(str(input_wav), ss=start, to=end).output(
        str(output_path), acodec="copy"
    )
    stream.overwrite_output().run(quiet=True)
