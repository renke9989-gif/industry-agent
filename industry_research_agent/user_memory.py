"""Structured user-profile memory, intentionally separate from checkpoints."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path


_lock = threading.Lock()


def _path() -> Path:
    configured = os.getenv("USER_MEMORY_SQLITE_PATH", ".data/user_memory.sqlite3")
    path = Path(configured)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect():
    connection = sqlite3.connect(str(_path()), check_same_thread=False)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            user_id TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            source TEXT NOT NULL,
            confirmed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, fact_key)
        )
    """)
    connection.commit()
    return connection


def get_facts(user_id: str) -> dict:
    with _lock:
        connection = _connect()
        rows = connection.execute(
            "SELECT fact_key, fact_value FROM user_facts WHERE user_id=? AND confirmed=1", (user_id,)
        ).fetchall()
        connection.close()
    return {key: json.loads(value) for key, value in rows}


def save_confirmed_fact(user_id: str, fact_key: str, fact_value, source: str = "user_confirmed") -> None:
    with _lock:
        connection = _connect()
        connection.execute(
            "INSERT OR REPLACE INTO user_facts(user_id,fact_key,fact_value,source,confirmed,updated_at) VALUES(?,?,?,?,1,CURRENT_TIMESTAMP)",
            (user_id, fact_key, json.dumps(fact_value, ensure_ascii=False), source),
        )
        connection.commit()
        connection.close()


def delete_facts(user_id: str) -> None:
    with _lock:
        connection = _connect()
        connection.execute("DELETE FROM user_facts WHERE user_id=?", (user_id,))
        connection.commit()
        connection.close()
