from __future__ import annotations

from .openai import OpenAI


class Grok(OpenAI):
    """Translator using Grok API (xAI, OpenAI-compatible).

    Inherits from OpenAI translator with different API endpoint.
    """

    name = "grok"
    default_model = "grok-3-mini"
    api_key_env_var = "GROK_API_KEY"
    base_url = "https://api.x.ai/v1"
