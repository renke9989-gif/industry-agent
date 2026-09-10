"""Durable local Run Store used by the web worker.

The LangGraph checkpoint contains the graph state.  This store contains the
execution envelope around one user request, so a disconnected browser or a
restarted process can still display and recover the run lifecycle.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


RUN_STATES = {"queued", "running", "retrying", "succeeded", "failed", "timed_out", "cancelled"}
TERMINAL_STATES = {"succeeded", "failed", "timed_out", "cancelled"}


class RunStore:
    def __init__(self, path: str | None = None, *, lease_seconds: int = 45):
        configured = path or os.getenv("RUN_STORE_SQLITE_PATH", ".data/runs.sqlite3")
        db_path = Path(configured)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.lease_seconds = max(5, int(lease_seconds))
        with self.lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA busy_timeout=10000")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    request_hash TEXT NOT NULL,
                    request_text TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    owner_id TEXT,
                    lease_expires_at REAL,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    progress REAL NOT NULL DEFAULT 0,
                    current_node TEXT,
                    checkpoint_ref TEXT,
                    artifact_ref TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    heartbeat_at REAL,
                    finished_at REAL,
                    UNIQUE(session_id, request_hash)
                )
                """
            )
            columns = {row[1] for row in self.connection.execute("PRAGMA table_info(runs)").fetchall()}
            if "request_text" not in columns:
                self.connection.execute("ALTER TABLE runs ADD COLUMN request_text TEXT NOT NULL DEFAULT ''")
            self.connection.commit()

    @staticmethod
    def request_hash(message: str) -> str:
        return hashlib.sha256(" ".join(str(message).split()).encode("utf-8")).hexdigest()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row else None

    def create_or_get(self, session_id: str, *, user_id: str, message: str) -> dict:
        now = time.time()
        digest = self.request_hash(message)
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM runs WHERE session_id = ? AND request_hash = ?",
                (session_id, digest),
            ).fetchone()
            if row:
                existing = self._row(row) or {}
                # Deduplicate active requests, but allow the same user message
                # again after its prior run reached a terminal state.
                if existing.get("status") not in TERMINAL_STATES:
                    return existing
                digest = hashlib.sha256(f"{digest}:{now}:{uuid.uuid4()}".encode("utf-8")).hexdigest()
            run_id = str(uuid.uuid4())
            self.connection.execute(
                """
                INSERT INTO runs(run_id, session_id, user_id, request_hash, request_text, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?)
                """,
                (run_id, session_id, user_id or "anonymous", digest, str(message), now),
            )
            self.connection.commit()
            return self.get(run_id) or {}

    def get(self, run_id: str) -> dict | None:
        with self.lock:
            return self._row(self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone())

    def latest_for_session(self, session_id: str) -> dict | None:
        with self.lock:
            return self._row(self.connection.execute(
                "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)
            ).fetchone())

    def list_recent(self, *, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 100)),)
            ).fetchall()
        return [dict(row) for row in rows]

    def active_for_session(self, session_id: str) -> dict | None:
        with self.lock:
            return self._row(self.connection.execute(
                """
                SELECT * FROM runs WHERE session_id = ?
                AND status IN ('queued', 'running', 'retrying') AND cancel_requested = 0
                ORDER BY created_at DESC LIMIT 1
                """, (session_id,)
            ).fetchone())

    def list_queued(self, *, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM runs WHERE status = 'queued' AND cancel_requested = 0 ORDER BY created_at LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def acquire(self, run_id: str, *, owner_id: str | None = None) -> dict:
        owner = owner_id or f"worker-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self.lock:
            row = self.get(run_id)
            if not row:
                raise KeyError(f"run not found: {run_id}")
            if row["cancel_requested"]:
                self._update_locked(run_id, status="cancelled", finished_at=now, error_code="cancelled")
                return self.get(run_id) or row
            token = int(row.get("fencing_token") or 0) + 1
            self._update_locked(
                run_id, status="running", attempt=int(row.get("attempt") or 0) + 1,
                owner_id=owner, lease_expires_at=now + self.lease_seconds,
                heartbeat_at=now, fencing_token=token, started_at=row.get("started_at") or now,
            )
            return self.get(run_id) or row

    def heartbeat(self, run_id: str, *, fencing_token: int, progress: float | None = None,
                  current_node: str | None = None) -> bool:
        now = time.time()
        with self.lock:
            values: dict[str, Any] = {"heartbeat_at": now, "lease_expires_at": now + self.lease_seconds}
            if progress is not None:
                values["progress"] = max(0.0, min(1.0, float(progress)))
            if current_node is not None:
                values["current_node"] = current_node
            return self._conditional_update(run_id, fencing_token, **values)

    def update(self, run_id: str, *, fencing_token: int | None = None, **values: Any) -> bool:
        allowed = {"status", "progress", "current_node", "checkpoint_ref", "artifact_ref",
                   "error_code", "error_message", "cancel_requested", "finished_at", "heartbeat_at",
                   "lease_expires_at", "owner_id", "attempt"}
        values = {key: value for key, value in values.items() if key in allowed}
        if "status" in values and values["status"] not in RUN_STATES:
            raise ValueError(f"invalid run status: {values['status']}")
        with self.lock:
            if fencing_token is None:
                return self._update_locked(run_id, **values)
            return self._conditional_update(run_id, fencing_token, **values)

    def _update_locked(self, run_id: str, **values: Any) -> bool:
        if not values:
            return False
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.lock:
            cursor = self.connection.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?", (*values.values(), run_id)
            )
            self.connection.commit()
            return cursor.rowcount == 1

    def _conditional_update(self, run_id: str, fencing_token: int, **values: Any) -> bool:
        values["fencing_token"] = fencing_token
        assignments = ", ".join(f"{key} = ?" for key in values)
        cursor = self.connection.execute(
            f"UPDATE runs SET {assignments} WHERE run_id = ? AND fencing_token = ? AND status NOT IN ('succeeded','failed','timed_out','cancelled')",
            (*values.values(), run_id, fencing_token),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def request_cancel(self, run_id: str) -> dict | None:
        with self.lock:
            row = self.get(run_id)
            if not row:
                return None
            now = time.time()
            if row["status"] in TERMINAL_STATES:
                return row
            values = {"cancel_requested": 1, "error_code": "cancel_requested"}
            if row["status"] == "queued":
                values.update(status="cancelled", finished_at=now)
            self._update_locked(run_id, **values)
            return self.get(run_id)

    def recover_stale(self) -> dict[str, int]:
        now = time.time()
        recovered = failed = 0
        with self.lock:
            rows = self.connection.execute(
                "SELECT run_id, checkpoint_ref FROM runs WHERE status IN ('running','retrying') AND lease_expires_at < ?",
                (now,),
            ).fetchall()
            for row in rows:
                if row["checkpoint_ref"]:
                    self._update_locked(row["run_id"], status="queued", owner_id=None, lease_expires_at=None,
                                        error_code="worker_lease_expired", error_message="previous worker lease expired")
                    recovered += 1
                else:
                    self._update_locked(row["run_id"], status="failed", finished_at=now,
                                        error_code="stale_run_without_checkpoint", error_message="no checkpoint available")
                    failed += 1
        return {"recovered": recovered, "failed": failed}

    def close(self) -> None:
        with self.lock:
            self.connection.close()
