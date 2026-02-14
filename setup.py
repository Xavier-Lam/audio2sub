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
        if match and match.group(1) != "__all__":
            package[match.group(1)] = eval(match.group(2))

transcriber_requirements = {
    "ffmpeg-python>=0.2.0",
    "torch>=2.1.0",
    "torchaudio>=2.1.0",
    "onnxruntime>=1.14,<2",
}

gemini_requirements = {
    "google-genai>=1.0.0",
}

openai_requirements = {
    "openai>=1.0.0",
}

faster_whisper_requirements = {
    "faster-whisper>=1.0.1",
}

whisper_requirements = {
    "openai-whisper>=20231117",
}

all_requirements = (
    transcriber_requirements
    | gemini_requirements
    | openai_requirements
    | faster_whisper_requirements
    | whisper_requirements
)

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
        "pysrt>=1.1.2",
        "tqdm",
    ],
    extras_require={
        "faster_whisper": transcriber_requirements | faster_whisper_requirements,
        "whisper": transcriber_requirements | whisper_requirements,
        "gemini": transcriber_requirements | gemini_requirements,
        "subtranslator": gemini_requirements | openai_requirements,
        "subaligner": gemini_requirements | openai_requirements,
        "dev": all_requirements | {"pytest>=7.4.0"},
        "all": all_requirements,
    },
    entry_points={
        "console_scripts": [
            "audio2sub=audio2sub.cli.audio2sub:main",
            "subtranslator=audio2sub.cli.subtranslator:main",
            "subaligner=audio2sub.cli.subaligner:main",
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
