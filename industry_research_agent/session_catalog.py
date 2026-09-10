"""Persistent catalog for user-visible research sessions.

LangGraph checkpoints persist graph state, but they are intentionally keyed by
an opaque thread id and do not provide a chat-history index for the web UI.
This small catalog stores only session metadata; message content continues to
live in the LangGraph checkpoint database.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path


class SessionCatalog:
    def __init__(self, path: str | None = None):
        configured = path or os.getenv("SESSION_CATALOG_SQLITE_PATH", ".data/sessions.sqlite3")
        db_path = Path(configured)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=10000")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    title TEXT NOT NULL DEFAULT '新调研',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._connection.commit()

    def register(
        self,
        session_id: str,
        *,
        user_id: str = "anonymous",
        title: str = "新调研",
        timestamp: float | None = None,
    ) -> None:
        now = float(timestamp or time.time())
        clean_title = self._clean_title(title)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO research_sessions(session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, user_id or "anonymous", clean_title, now, now),
            )
            self._connection.commit()

    def touch(self, session_id: str, *, user_id: str, first_message: str = "") -> None:
        now = time.time()
        self.register(session_id, user_id=user_id, title=first_message or "新调研", timestamp=now)
        with self._lock:
            row = self._connection.execute(
                "SELECT title FROM research_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            title = row["title"] if row else "新调研"
            if first_message and title in {"新调研", "历史调研"}:
                title = self._clean_title(first_message)
            self._connection.execute(
                """
                UPDATE research_sessions
                SET user_id = ?, title = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (user_id or "anonymous", title, now, session_id),
            )
            self._connection.commit()

    def list_recent(self, *, limit: int = 50, user_id: str | None = None) -> list[dict]:
        safe_limit = max(1, min(int(limit), 100))
        with self._lock:
            if user_id:
                rows = self._connection.execute(
                    """SELECT session_id, user_id, title, created_at, updated_at
                       FROM research_sessions WHERE user_id = ? OR user_id = 'anonymous'
                       ORDER BY updated_at DESC LIMIT ?""",
                    (user_id, safe_limit),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """SELECT session_id, user_id, title, created_at, updated_at
                       FROM research_sessions ORDER BY updated_at DESC LIMIT ?""", (safe_limit,)
                ).fetchall()
        return [dict(row) for row in rows]

    def import_checkpoint_threads(self, checkpoint_connection: sqlite3.Connection | None) -> int:
        """Make pre-catalog checkpoints discoverable in the local demo UI."""
        if checkpoint_connection is None:
            return 0
        try:
            rows = checkpoint_connection.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"
            ).fetchall()
        except sqlite3.Error:
            return 0
        before = len(self.list_recent(limit=100))
        for row in rows:
            thread_id = str(row[0] if not isinstance(row, sqlite3.Row) else row["thread_id"])
            if thread_id:
                self.register(thread_id, title="历史调研")
        after = len(self.list_recent(limit=100))
        return max(0, after - before)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _clean_title(value: str) -> str:
        title = " ".join(str(value or "").split()).strip()
        if not title:
            return "新调研"
        return title[:42] + ("…" if len(title) > 42 else "")
