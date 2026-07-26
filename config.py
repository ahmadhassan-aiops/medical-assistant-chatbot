"""Provider configuration for the medical assistant.

All supported providers expose an OpenAI-compatible Chat Completions API, so a
single client (see ``llm.py``) talks to any of them — only the base URL, model
name and API-key environment variable change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    key: str                      # internal id
    label: str                    # shown in the UI
    base_url: str
    env_var: str                  # env var / secret holding the API key
    models: list[str] = field(default_factory=list)
    signup_url: str = ""
    free: bool = False

    def api_key(self) -> str | None:
        # Prefer Streamlit secrets when present, fall back to environment.
        try:
            import streamlit as st

            if self.env_var in st.secrets:
                return str(st.secrets[self.env_var]).strip() or None
        except Exception:
            pass
        val = os.getenv(self.env_var)
        return val.strip() if val else None


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        key="groq",
        label="Groq  (free)",
        base_url="https://api.groq.com/openai/v1",
        env_var="GROQ_API_KEY",
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "deepseek-r1-distill-llama-70b",
        ],
        signup_url="https://console.groq.com/keys",
        free=True,
    ),
    "xai": Provider(
        key="xai",
        label="xAI Grok",
        base_url="https://api.x.ai/v1",
        env_var="XAI_API_KEY",
        models=["grok-2-latest", "grok-2-1212", "grok-beta"],
        signup_url="https://console.x.ai",
        free=False,
    ),
    "custom": Provider(
        key="custom",
        label="Custom (OpenAI-compatible)",
        base_url=os.getenv("CUSTOM_BASE_URL", "https://api.openai.com/v1"),
        env_var="CUSTOM_API_KEY",
        models=[m for m in [os.getenv("CUSTOM_MODEL", "")] if m] or ["gpt-4o-mini"],
        signup_url="",
        free=False,
    ),
}

DEFAULT_PROVIDER = "groq"
