"""Audio2Sub package: convert media to subtitles."""

from .common import Segment
from .transcribe import transcribe


__title__ = "audio2sub"
__description__ = "Transcribe media files to SRT subtitles."
__url__ = "https://github.com/Xavier-Lam/audio2sub"
__version__ = "0.1.2"
__author__ = "Xavier-Lam"
__author_email__ = "xavierlam7@hotmail.com"


__all__ = [
    "__version__",
    "Segment",
    "transcribe",
]
