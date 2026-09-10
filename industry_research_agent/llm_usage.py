"""Small helpers for preserving provider token usage in graph state."""
from __future__ import annotations


def extract_usage(response) -> dict:
    metadata = getattr(response, "usage_metadata", None) or {}
    if not metadata:
        metadata = (getattr(response, "response_metadata", None) or {}).get("token_usage", {})
    return {
        "input_tokens": int(metadata.get("input_tokens", metadata.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(metadata.get("output_tokens", metadata.get("completion_tokens", 0)) or 0),
        "total_tokens": int(metadata.get("total_tokens", 0) or 0),
        "llm_calls": 1 if metadata else 0,
    }


def merge_usage(previous: dict, current: dict) -> dict:
    merged = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    for key in merged:
        merged[key] = int(previous.get(key, 0) or 0) + int(current.get(key, 0) or 0)
    if not merged["total_tokens"]:
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged
