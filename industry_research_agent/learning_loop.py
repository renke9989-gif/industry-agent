"""Controlled learning log; review is required before promotion to rules."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent
LEARNINGS_DIR = PROJECT_ROOT / ".learnings"
_LOCK = threading.Lock()


def _append(record: Dict[str, Any]) -> None:
    # Learning logs are observability only. A read-only deployment directory,
    # MCP subprocess sandbox, or mounted volume must never turn a search tool
    # call into a non-JSON error response.
    try:
        LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        record["status"] = "unreviewed"
        with _LOCK, (LEARNINGS_DIR / "review_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return


def log_error(category: str, message: str, *, context: Dict[str, Any] | None = None) -> None:
    _append({"type": "error", "category": category, "message": message[:2000], "context": context or {}})


def log_learning(category: str, lesson: str, *, evidence: str = "") -> None:
    _append({"type": "learning", "category": category, "lesson": lesson[:2000], "evidence": evidence[:1000]})


def log_feature_request(request: str, *, source: str = "user") -> None:
    _append({"type": "feature_request", "request": request[:2000], "source": source})
