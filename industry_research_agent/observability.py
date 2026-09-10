"""Thread-safe JSONL runtime events for local diagnostics and evaluation."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


_lock = threading.Lock()


def log_runtime_event(event_type: str, **fields) -> None:
    path = Path(os.getenv("RUNTIME_LOG_PATH", ".data/runtime.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **fields,
    }
    # Never log common secret-bearing values even if a caller passes them.
    for key in list(record):
        if any(marker in key.lower() for marker in ("api_key", "secret", "password", "authorization")):
            record[key] = "[redacted]"
    with _lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
