# Audio2Sub Docker Images

Audio2Sub automatically transcribes audio from video or audio files and generates subtitles. Official Docker images provide easy deployment with multiple transcription backends including faster-whisper, OpenAI Whisper, and Google Gemini.

## Quick Start

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

## Available Images

All images are available on Docker Hub: `xavierlam/audio2sub:<tag>`

| Tag | Backend | Description |
|-----|---------|-------------|
| `latest` | faster-whisper | Fast CTranslate2-based Whisper (Recommended) |
| `faster-whisper` | faster-whisper | Same as latest |
| `whisper` | whisper | Original OpenAI Whisper |
| `gemini` | gemini | Google Gemini API |

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
