from __future__ import annotations
import os
from run_store import RunStore

def build_run_store():
    if os.getenv("RUN_STORE_BACKEND", "sqlite").lower() == "mysql":
        from mysql_run_store import MySQLRunStore
        return MySQLRunStore()
    return RunStore()
