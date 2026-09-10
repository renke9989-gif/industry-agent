"""Evidence-first context budgeting and deterministic compression."""
from __future__ import annotations

import re
from typing import Iterable


def estimate_tokens(text: str) -> int:
    # A conservative, dependency-free estimate suitable for budgeting.
    return max(0, (len(str(text or "")) + 3) // 4)


def compact_context(
    sections: Iterable[tuple[str, str]],
    *,
    budget_tokens: int = 24_000,
    trigger_ratio: float = 0.8,
) -> tuple[str, dict]:
    """Return a bounded context while preserving marked critical sections.

    Sections whose name contains ``user``, ``condition``, ``evidence``,
    ``claim`` or ``conflict`` are kept first. Other sections are trimmed from
    the tail. This is intentionally deterministic so it can be tested without
    an extra LLM call.
    """
    budget = max(100, int(budget_tokens))
    entries = []
    seen = set()
    for name, value in sections:
        entry = (str(name), str(value or ""))
        if not entry[1].strip() or entry in seen:
            continue
        seen.add(entry)
        entries.append(entry)
    raw = "\n\n".join(f"[{name}]\n{value}" for name, value in entries)
    before = estimate_tokens(raw)
    limit = int(budget * max(0.1, min(1.0, trigger_ratio)))
    if before <= limit:
        return raw, {"before_tokens": before, "after_tokens": before, "compressed": False, "overflow_protected": False}

    critical = []
    optional = []
    for entry in entries:
        (critical if re.search(r"user|condition|evidence|claim|conflict|question|用户|条件|证据|冲突|问题", entry[0], re.I) else optional).append(entry)
    selected: list[tuple[str, str]] = []
    used = 0
    for name, value in critical + optional:
        remaining = max(0, limit - used)
        if not remaining:
            break
        chars = remaining * 4
        clipped = value if len(value) <= chars else value[:max(80, chars)] + "…"
        selected.append((name, clipped))
        used += estimate_tokens(clipped) + estimate_tokens(name) + 2
    result = "\n\n".join(f"[{name}]\n{value}" for name, value in selected)
    return result, {
        "before_tokens": before,
        "after_tokens": estimate_tokens(result),
        "compressed": True,
        "overflow_protected": estimate_tokens(result) > budget,
    }


def merge_context_metrics(previous: dict | None, current: dict | None) -> dict:
    """Accumulate context observations without retaining raw prompts."""
    previous = dict(previous or {})
    current = dict(current or {})
    calls = list(previous.get("calls", [])) + [current] if current else list(previous.get("calls", []))
    calls = calls[-50:]
    before = sum(int(item.get("before_tokens", 0) or 0) for item in calls)
    after = sum(int(item.get("after_tokens", 0) or 0) for item in calls)
    return {
        "calls": calls,
        "call_count": len(calls),
        "before_tokens": before,
        "after_tokens": after,
        "compression_count": sum(bool(item.get("compressed")) for item in calls),
        "overflow_count": sum(bool(item.get("overflow_protected")) for item in calls),
        "compression_ratio": round(after / before, 4) if before else 1.0,
        "preserved_condition_count": max((int(item.get("preserved_condition_count", 0) or 0) for item in calls), default=0),
        "preserved_evidence_count": max((int(item.get("preserved_evidence_count", 0) or 0) for item in calls), default=0),
        "budget": max((int(item.get("budget", 0) or 0) for item in calls), default=0),
    }
