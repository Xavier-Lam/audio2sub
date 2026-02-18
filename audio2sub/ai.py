"""AI backend base class and mixins for Gemini, OpenAI and Grok integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import json
import logging
import os
import time
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

from .common import MissingDependencyException, Usage

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AIBackendBase(ABC):
    """Common base for all AI-powered backends."""

    name: str = "base"
    default_model: str = ""
    default_chunk: int = 2000
    default_retries: int = 3
    api_key_env_var: Optional[str] = None

    def __init__(self, model="", api_key=None) -> None:
        self.model = model or self.default_model
        self.api_key = api_key

    @classmethod
    def contribute_to_cli(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default=cls.default_model or None,
            help=(
                "Model name to use"
                + (f" (default: {cls.default_model})" if cls.default_model else "")
            ),
        )
        parser.add_argument(
            "--api-key",
            dest="api_key",
            required=False,
            help=(
                "API key (optional; env "
                f"{cls.api_key_env_var or 'API key env var'} "
                "is used if not provided)"
            ),
        )
        parser.add_argument(
            "--chunk",
            type=int,
            default=cls.default_chunk,
            help=("Items per API request " f"(default: {cls.default_chunk})"),
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            required=False,
            help="Additional instructions",
        )
        parser.add_argument(
            "--retries",
            type=int,
            default=cls.default_retries,
            help=(
                "Max retries per API request on transient failures "
                f"(default: {cls.default_retries})"
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "AIBackendBase":
        return cls(model=args.model, api_key=args.api_key)

    @classmethod
    def opts_from_cli(cls, args: argparse.Namespace) -> dict:
        return {"chunk": args.chunk, "prompt": args.prompt, "retries": args.retries}

    def _ensure_client(self):
        if getattr(self, "_client", None):
            return self._client
        self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self):
        """Instantiate the API client."""

    def _iter_chunks(self, items: list, size: int) -> Iterable[list]:
        if size <= 0:
            size = len(items)
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _resolve_api_key(self) -> str:
        api_key = self.api_key
        if not api_key and self.api_key_env_var:
            api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            env_hint = self.api_key_env_var or "API key"
            raise RuntimeError(f"{env_hint} is required for {self.name} backend.")
        return api_key

    _retry_delay: float = 0.5

    def _retry(
        self,
        fn: Callable[..., T],
        *args,
        retries: Optional[int] = None,
        **kwargs,
    ) -> T:
        max_retries = retries if retries is not None else self.default_retries
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "%s request failed (attempt %d/%d): %s  " "— retrying in %gs",
                        self.name,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        self._retry_delay,
                    )
                    time.sleep(self._retry_delay)
        raise last_exc  # type: ignore[misc]


class GeminiMixin:
    name = "gemini"
    default_model = "gemini-2.5-flash"
    api_key_env_var = "GEMINI_API_KEY"

    def _create_client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise MissingDependencyException(self) from exc
        api_key = self._resolve_api_key()
        return genai.Client(api_key=api_key)

    def _call(self, client, contents, retries=None) -> Tuple[str, Usage]:
        """Make a raw Gemini API call and return (text, Usage)."""

        def _do_call():
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
            )
            raw_text = response.text.strip() if hasattr(response, "text") else ""
            usage = Usage(
                tokens_in=getattr(response.usage_metadata, "prompt_token_count", 0),
                tokens_out=getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
            )
            return raw_text, usage

        return self._retry(_do_call, retries=retries)

    def _call_text(
        self, client, prompt: List[str], data, retries=None
    ) -> Tuple[str, Usage]:
        """Send a text+JSON request to Gemini."""
        parts = [
            {"text": "\n\n".join(prompt)},
            {"text": json.dumps(data, ensure_ascii=False)},
        ]
        contents = [{"role": "user", "parts": parts}]
        return self._call(client, contents, retries=retries)


class OpenAIMixin:
    name = "openai"
    default_model = "gpt-4o-mini"
    api_key_env_var = "OPENAI_API_KEY"
    base_url: Optional[str] = None

    def _create_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyException(self) from exc
        api_key = self._resolve_api_key()
        kwargs: dict = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _call(self, client, messages, retries=None) -> Tuple[str, Usage]:
        """Make an OpenAI-compatible chat call and return (text, Usage)."""

        def _do_call():
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            raw_text = response.choices[0].message.content.strip()
            usage = Usage(
                tokens_in=getattr(response.usage, "prompt_tokens", 0),
                tokens_out=getattr(response.usage, "completion_tokens", 0),
            )
            return raw_text, usage

        return self._retry(_do_call, retries=retries)

    def _call_text(
        self, client, prompt: List[str], data, retries=None
    ) -> Tuple[str, Usage]:
        """Send a text+JSON request via OpenAI-compatible chat."""
        messages = [
            {"role": "system", "content": "\n\n".join(prompt)},
            {
                "role": "user",
                "content": json.dumps(data, ensure_ascii=False),
            },
        ]
        return self._call(client, messages, retries=retries)


class GrokMixin(OpenAIMixin):
    name = "grok"
    default_model = "grok-3-mini"
    api_key_env_var = "GROK_API_KEY"
    base_url = "https://api.x.ai/v1"
