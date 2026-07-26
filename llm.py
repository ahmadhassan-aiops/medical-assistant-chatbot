"""Thin wrapper around the OpenAI-compatible Chat Completions API.

Works with Groq, xAI Grok, OpenAI and any other compatible endpoint — the
provider only decides the base URL, model and key (see ``config.py``).
"""

from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from config import Provider


def get_client(provider: Provider) -> OpenAI:
    key = provider.api_key()
    if not key:
        raise RuntimeError(
            f"No API key found for {provider.label}. "
            f"Set {provider.env_var} in your .env file or Streamlit secrets."
        )
    return OpenAI(api_key=key, base_url=provider.base_url)


def stream_chat(
    provider: Provider,
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> Iterator[str]:
    """Yield response text chunks as they arrive."""
    client = get_client(provider)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
