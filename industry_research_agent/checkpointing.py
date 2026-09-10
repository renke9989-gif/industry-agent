"""Checkpoint backend selection for the web app and offline tests.

The web service uses a local SQLite checkpointer so sessions survive a process
restart.  Tests and one-off evaluation runs can explicitly use MemorySaver to
avoid writing state to the repository.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def build_checkpointer(*, persistent: bool = False):
    backend = os.getenv("CHECKPOINT_BACKEND", "sqlite" if persistent else "memory").lower()
    if backend in {"memory", "inmemory", "none"}:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver(), None

    if backend != "sqlite":
        raise ValueError(f"unsupported CHECKPOINT_BACKEND: {backend}")

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite checkpoint requires langgraph-checkpoint-sqlite; "
            "install project requirements or set CHECKPOINT_BACKEND=memory"
        ) from exc

    configured_path = os.getenv("CHECKPOINT_SQLITE_PATH", ".data/checkpoints.sqlite3")
    path = Path(configured_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=10000")
    saver = SqliteSaver(connection)
    saver.setup()
    return saver, connection
