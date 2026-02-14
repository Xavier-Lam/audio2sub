# Audio2Sub Docker Images

Audio2Sub is an all-in-one subtitle toolkit. Official Docker images provide easy deployment for all three tools: **audio2sub** (speech-to-text transcription), **subtranslator** (AI subtitle translation), and **subaligner** (AI subtitle timing synchronization).

## Quick Start

### audio2sub — Transcribe Audio to Subtitles

Significantly faster transcription with GPU:

```bash
# Using faster-whisper (default, recommended)
docker run --rm --gpus all \
  -v "$(pwd):/media" \
  xavierlam/audio2sub \
  video.mp4 -o video.srt --lang en

# Using OpenAI Whisper
docker run --rm --gpus all \
  -v "$(pwd):/media" \
  xavierlam/audio2sub:whisper \
  video.mp4 -o video.srt --lang en

# Using faster-whisper without GPU
docker run --rm -v "$(pwd):/media" xavierlam/audio2sub \
  video.mp4 -o video.srt --lang en

# Using Gemini API (no GPU needed)
docker run --rm -v "$(pwd):/media" \
  -e GEMINI_API_KEY=your_api_key_here \
  xavierlam/audio2sub:gemini \
  video.mp4 -o video.srt --lang en
```

### subtranslator — Translate Subtitles

```bash
# Translate Chinese subtitles to English (Gemini, default)
docker run --rm -v "$(pwd):/media" \
  -e GEMINI_API_KEY=your_api_key_here \
  xavierlam/subtranslator \
  video_zh.srt -s zh -d en -o video_en.srt

# Use Grok backend
docker run --rm -v "$(pwd):/media" \
  -e GROK_API_KEY=your_api_key_here \
  xavierlam/subtranslator \
  video_zh.srt -s zh -d en -o video_en.srt -t grok

# Use OpenAI backend
docker run --rm -v "$(pwd):/media" \
  -e OPENAI_API_KEY=your_api_key_here \
  xavierlam/subtranslator \
  video_zh.srt -s zh -d en -o video_en.srt -t openai
```

### subaligner — Align Subtitle Timing

```bash
# Align Chinese subtitles using English reference timing
docker run --rm -v "$(pwd):/media" \
  -e GEMINI_API_KEY=your_api_key_here \
  xavierlam/subaligner \
  -i chinese.srt -r english_reference.srt -o output.srt --src-lang zh --ref-lang en

# Use Grok backend
docker run --rm -v "$(pwd):/media" \
  -e GROK_API_KEY=your_api_key_here \
  xavierlam/subaligner \
  -i chinese.srt -r english_reference.srt -o output.srt -a grok
```

## Available Images

All images are available on Docker Hub.

### audio2sub

| Tag | Backend | Description |
|-----|---------|-------------|
| `xavierlam/audio2sub:latest` | faster-whisper | Fast CTranslate2-based Whisper (Recommended) |
| `xavierlam/audio2sub:faster-whisper` | faster-whisper | Same as latest |
| `xavierlam/audio2sub:whisper` | whisper | Original OpenAI Whisper |
| `xavierlam/audio2sub:gemini` | gemini | Google Gemini API |

### subtranslator

| Tag | Description |
|-----|-------------|
| `xavierlam/subtranslator:latest` | AI subtitle translator with Gemini, Grok, and OpenAI support |

### subaligner

| Tag | Description |
|-----|-------------|
| `xavierlam/subaligner:latest` | AI subtitle aligner with Gemini, Grok, and OpenAI support |

## Troubleshooting

### GPU Not Working

If GPU acceleration isn't working, verify your setup:

1. **Check NVIDIA drivers**: Run `nvidia-smi` on your host machine. You should see your GPU listed.
2. **Verify Docker GPU access**: Run `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`. This should show the same GPU information inside Docker.
3. **Install NVIDIA Container Toolkit**: If step 2 fails, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
4. **Check Docker version**: Ensure you have Docker 19.03+ which supports the `--gpus` flag.

## Resources

- [Project Repository](https://github.com/Xavier-Lam/audio2sub)
- [Docker Hub](https://hub.docker.com/repository/docker/xavierlam/audio2sub)
- [Issue Tracker](https://github.com/Xavier-Lam/audio2sub/issues)
