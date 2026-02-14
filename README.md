# Audio2Sub

[![PyPI](https://img.shields.io/pypi/v/audio2sub.svg)](https://pypi.org/project/audio2sub/)
[![CI](https://github.com/Xavier-Lam/audio2sub/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/Xavier-Lam/audio2sub/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/xavierlam/audio2sub/latest?label=docker)](https://hub.docker.com/r/xavierlam/audio2sub)

**Audio2Sub** is an all-in-one subtitle toolkit that helps you automatically generate, translate, and synchronize subtitles using AI. Whether you need to transcribe audio to subtitles, translate subtitles into another language, or fix out-of-sync subtitle timing, Audio2Sub has you covered.

The toolkit includes three command-line tools:

- **audio2sub** — Automatically transcribe audio from video or audio files and generate `.srt` subtitles. Uses FFmpeg for media handling, [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection, and multiple transcription backends.
- **subtranslator** — Translate subtitle files from one language to another using AI. Supports translating between any language pair with backends like Google Gemini, Grok, and OpenAI.
- **subaligner** — Synchronize and align subtitle timing to match a reference subtitle. Ideal for fixing out-of-sync subtitles or aligning translated subtitles to the original timing, even when the subtitles are in different languages.

## Installation

Before installing, you must have [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system's PATH.

You can install Audio2Sub using `pip`. The default installation includes the `faster_whisper` backend.

```bash
pip install audio2sub[faster_whisper]
```

> **You have to choose a transcription backend to install**, or you will get an error about missing dependencies when you run the program.

To install with a specific transcription backend, see the table in the [Backends](#backends) section below.

To install with subtitle translation and alignment support:

```bash
# Install with subtitle translator support (Gemini + Grok + OpenAI)
pip install audio2sub[subtranslator]

# Install with subtitle aligner support (Gemini + Grok + OpenAI)
pip install audio2sub[subaligner]

# Install everything
pip install audio2sub[all]
```

## Usage

### Basic Examples

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

## Tools

### subtranslator — AI Subtitle Translator

Translate subtitle files between languages while preserving timing and formatting. Supports Gemini, Grok, and OpenAI backends.

To use the tool, you need to install the package by running `pip install audio2sub[subtranslator]` to get the necessary dependencies.

#### Basic Example

```bash
# Translate Chinese subtitles to English using the default backend (Gemini)
subtranslator my_video_zh.srt -s zh -d en -o my_video_en.srt

# Use OpenAI as the translation backend
subtranslator my_video_zh.srt -s zh -d en -o my_video_en.srt -t openai
```

### subaligner — AI Subtitle Synchronization & Alignment Tool

Automatically align subtitle timing to a reference subtitle or an audio transcription, even across different languages. Use this to fix out-of-sync subtitles or adapt timing between video versions.

Don't have a reference subtitle? No problem. Use **audio2sub** to transcribe your video and generate a reference subtitle file, then use subaligner to align your existing subtitle to match the generated timing.

To use the tool, you need to install the package by running `pip install audio2sub[subaligner]` to get the necessary dependencies.

#### Basic Example

```bash
# Align Chinese subtitles using English reference timing
subaligner -i chinese.srt -r english_reference.srt -o output.srt --src-lang zh --ref-lang en

# Use OpenAI backend
subaligner -i chinese.srt -r english_reference.srt -o output.srt -a openai
```

### Pipelining Tools Together

Audio2Sub's three tools can be chained together to create powerful subtitle workflows. Here are some common pipeline patterns:

#### Transcribe and Translate

Generate subtitles from a video, then translate them to another language:

```bash
audio2sub my_video.mp4 -o my_video_en.srt --lang en && subtranslator my_video_en.srt -s en -d es -o my_video_es.srt
```

#### Generate Reference for Alignment

Create a reference subtitle from your video, then align an existing subtitle to match:

```bash
audio2sub my_video_bluray.mp4 -o reference.srt --lang en && subaligner -i existing_subtitle.srt -r reference.srt -o aligned.srt --src-lang zh --ref-lang en
```

## Docker

Audio2Sub provides official Docker images for easy deployment without managing dependencies.

### Quick Start

```bash
# audio2sub: Transcribe with GPU support (recommended)
docker run --rm --gpus all -v "$(pwd):/media" xavierlam/audio2sub \
  my_video.mp4 -o my_video.srt --lang en

# audio2sub: Without GPU support, whisper backend
docker run --rm -v "$(pwd):/media" xavierlam/audio2sub:whisper \
  my_video.mp4 -o my_video.srt --lang en

# subtranslator: Translate subtitles
docker run --rm -v "$(pwd):/media" \
  -e GEMINI_API_KEY=your_api_key_here \
  xavierlam/subtranslator \
  my_video_zh.srt -s zh -d en -o my_video_en.srt

# subaligner: Align subtitle timing
docker run --rm -v "$(pwd):/media" \
  -e GEMINI_API_KEY=your_api_key_here \
  xavierlam/subaligner \
  -i chinese.srt -r english_ref.srt -o output.srt --src-lang zh --ref-lang en
```

Use `--gpus all` to enable GPU support, use different tags to select different backends.

### Pipeline Examples

The same workflows work seamlessly in Docker:

```bash
# Transcribe and translate in Docker
docker run --rm --gpus all -v "$(pwd):/media" xavierlam/audio2sub \
  my_video.mp4 -o en.srt --lang en && \
docker run --rm -v "$(pwd):/media" -e GEMINI_API_KEY=$GEMINI_API_KEY xavierlam/subtranslator \
  en.srt -s en -d zh -o zh.srt

# Generate reference and align
docker run --rm --gpus all -v "$(pwd):/media" xavierlam/audio2sub \
  my_video.mp4 -o reference.srt --lang en && \
docker run --rm -v "$(pwd):/media" -e GEMINI_API_KEY=$GEMINI_API_KEY xavierlam/subaligner \
  -i my_subtitle.srt -r reference.srt -o aligned.srt --src-lang zh --ref-lang en
```

### Available Images

#### audio2sub (Speech-to-Text Transcription)

| Image Tag | Backend | Description |
|-----------|---------|-------------|
| `xavierlam/audio2sub:latest` | faster-whisper | Recommended (same as faster-whisper) |
| `xavierlam/audio2sub:faster-whisper` | faster-whisper | Fast CTranslate2-based Whisper |
| `xavierlam/audio2sub:whisper` | whisper | Original OpenAI Whisper |
| `xavierlam/audio2sub:gemini` | gemini | Google Gemini API |

#### subtranslator (Subtitle Translation)

| Image Tag | Description |
|-----------|-------------|
| `xavierlam/subtranslator:latest` | AI subtitle translator with Gemini, Grok and OpenAI support |

#### subaligner (Subtitle Alignment)

| Image Tag | Description |
|-----------|-------------|
| `xavierlam/subaligner:latest` | AI subtitle aligner with Gemini, Grok and OpenAI support |

For detailed Docker documentation, GPU setup, and troubleshooting, see [docker/README.md](docker/README.md).

## Backends

### Transcription Backends (audio2sub)

Audio2Sub supports the following transcription backends.

| Backend Name      | Description |
| --- | --- |
| `faster_whisper` | A faster reimplementation of Whisper using CTranslate2. See [Faster Whisper](https://github.com/guillaumekln/faster-whisper). This is the default backend. |
| `whisper`        | The original speech recognition model by OpenAI. See [OpenAI Whisper](https://github.com/openai/whisper). |
| `gemini`         | Google's Gemini model via their API. Requires a `GEMINI_API_KEY` environment variable or `--api-key` argument.|

You should use `pip install audio2sub[<backend>]` to install the desired backend support and use the corresponding transcriber with the `-t` flag.

### Translation & Alignment Backends (subtranslator & subaligner)

Both the subtitle translator and subtitle aligner share the same set of AI backends.

| Backend Name | Description | API Key Env Var |
| --- | --- | --- |
| `gemini` | Google's Gemini model. Default backend. | `GEMINI_API_KEY` |
| `grok` | xAI's Grok model (OpenAI-compatible API). | `GROK_API_KEY` |
| `openai` | OpenAI's GPT models. | `OPENAI_API_KEY` |

Install with `pip install audio2sub[<backend>]` to get specific backend support. Or you can install with `pip install audio2sub[subtranslator]` or `pip install audio2sub[subaligner]` to get all supported backends.

> **Note**: I am not able to get an *OpenAI* API key to test the *OpenAI* backend, I don't know if it works as expected.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/Xavier-Lam/audio2sub).
