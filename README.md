# Audio2Sub

[![PyPI](https://img.shields.io/pypi/v/audio2sub.svg)](https://pypi.org/project/audio2sub/)
[![CI](https://github.com/Xavier-Lam/audio2sub/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/Xavier-Lam/audio2sub/actions/workflows/ci.yml)

**Audio2Sub** is a command-line tool that automatically transcribes audio from video or audio files and generates subtitles in the `.srt` format. It uses FFmpeg for media handling, [Silero VAD](https://github.com/snakers4/silero-vad) for precise voice activity detection, and supports multiple transcription backends to convert speech to text.

## Installation

Before installing, you must have [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system's PATH.

You can install Audio2Sub using `pip`. The default installation includes the `faster_whisper` backend.

```bash
pip install audio2sub[faster_whisper]
```

To install with a different backend, see the table in the [Backends](#Backends) section below.

## Usage
### Basic Example

```bash
audio2sub my_video.mp4 -o my_video.srt --lang en
```

This command will transcribe the audio from `my_video.mp4` into English and save the subtitles to `my_video.srt`.

**Notes:**
*   **First-Time Use**: The first time you run the program, it will download the necessary transcription models. This may take some time and require significant disk space.
*   **CUDA**: Performance significantly degraded without CUDA when using whisper-based local models. The program will raise a warning if CUDA is not available when it starts. If your system has a compatible GPU, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) first. If you are sure CUDA has been installed correctly and still get the warning, you may need to [reinstall a compatible PyTorch version manually](https://pytorch.org/get-started/locally/). The reinstallation of PyTorch may break other dependencies if you choose a different version than what you currently have. In this case, you may need to reinstall those according to the warnings shown.

### Using a Different Transcriber

Use the `-t` or `--transcriber` flag to select a different backend.

```bash
audio2sub my_audio.wav -o my_audio.srt --lang en -t whisper --model medium
```

Each transcriber has its own options. To see them, use `--help` with the transcriber specified.

```bash
audio2sub -t faster_whisper --help
```

## Backends

Audio2Sub supports the following transcription backends.

| Backend Name      | Description |
| --- | --- |
| `faster_whisper` | A faster reimplementation of Whisper using CTranslate2. See [Faster Whisper](https://github.com/guillaumekln/faster-whisper). This is the default backend. |
| `whisper`        | The original speech recognition model by OpenAI. See [OpenAI Whisper](https://github.com/openai/whisper). |
| `gemini`         | Google's Gemini model via their API. Requires a `GEMINI_API_KEY` environment variable or `--gemini-api-key` argument.|

You should use `pip install audio2sub[<backend>]` to install the desired backend support and use the corresponding transcriber with the `-t` flag.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the GitHub repository.
