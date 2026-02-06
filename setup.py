from pathlib import Path
from setuptools import find_packages, setup
import os
import re

this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

package = dict()

with open(os.path.join("audio2sub", "__init__.py"), "r") as f:
    lines = f.readlines()
    for line in lines:
        match = re.match(r"(__\w+?__)\s*=\s*(.+)$", line)
        if match:
            package[match.group(1)] = eval(match.group(2))

setup(
    name=package["__title__"],
    version=package["__version__"],
    author=package["__author__"],
    author_email=package["__author_email__"],
    url=package["__url__"],
    description=package["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "ffmpeg-python>=0.2.0",
        "pysrt>=1.1.2",
        "tqdm",
        "onnxruntime>=1.14,<2",
    ],
    extras_require={
        "faster_whisper": ["faster-whisper>=1.0.1"],
        "whisper": ["openai-whisper>=20231117"],
        "gemini": ["google-genai>=1.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "openai-whisper>=20231117",
            "faster-whisper>=1.0.1",
            "google-genai>=1.0.0",
        ],
        "all": [
            "openai-whisper>=20231117",
            "faster-whisper>=1.0.1",
            "google-genai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio2sub=audio2sub.cli:main",
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Text Processing :: Linguistic",
    ],
)
