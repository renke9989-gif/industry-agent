"""Validated runtime configuration with safe defaults for local development."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    max_concurrent_research: int
    session_rate_limit_per_minute: int
    max_message_chars: int
    context_token_budget: int


def load_settings() -> Settings:
    return Settings(
        max_concurrent_research=_positive_int("MAX_CONCURRENT_RESEARCH", 5),
        session_rate_limit_per_minute=_positive_int("SESSION_RATE_LIMIT_PER_MINUTE", 10),
        max_message_chars=_positive_int("MAX_MESSAGE_CHARS", 5000),
        context_token_budget=_positive_int("CONTEXT_TOKEN_BUDGET", 24000),
    )
