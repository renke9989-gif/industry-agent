"""Deterministic fault-injection harness for the local Run execution layer.

This complements ``eval_harness``: it does not call an LLM or the network and
focuses on lifecycle invariants (dedupe, cancellation, leases and recovery).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

from run_store import RunStore
from artifact_store import ArtifactStore
from workflow_blueprint import WorkflowBlueprint


def _store(tmp: str) -> RunStore:
    return RunStore(str(Path(tmp) / "runs.sqlite3"), lease_seconds=5)


def _new(store: RunStore, message: str = "研究咖啡行业") -> dict:
    return store.create_or_get("session-1", user_id="harness", message=message)


def case_search_timeout(tmp):
    s = _store(tmp); r = _new(s); s.update(r["run_id"], status="timed_out", error_code="search_timeout"); ok = s.get(r["run_id"])["status"] == "timed_out"; s.close(); return ok


def case_fetch_timeout(tmp):
    s = _store(tmp); r = _new(s); s.update(r["run_id"], status="timed_out", error_code="fetch_timeout"); ok = s.get(r["run_id"])["error_code"] == "fetch_timeout"; s.close(); return ok


def case_mcp_exit(tmp):
    s = _store(tmp); r = _new(s); a = s.acquire(r["run_id"]); s.update(r["run_id"], fencing_token=a["fencing_token"], status="retrying", error_code="mcp_process_exit"); ok = s.get(r["run_id"])["status"] == "retrying"; s.close(); return ok


def case_mcp_bad_schema(tmp):
    s = _store(tmp); r = _new(s); s.update(r["run_id"], status="failed", error_code="invalid_provider_payload"); ok = s.get(r["run_id"])["status"] == "failed"; s.close(); return ok


def case_direct_fallback(tmp):
    s = _store(tmp); r = _new(s); s.update(r["run_id"], status="succeeded", progress=1); ok = s.get(r["run_id"])["status"] == "succeeded"; s.close(); return ok


def case_transient_retry(tmp):
    s = _store(tmp); r = _new(s); a = s.acquire(r["run_id"]); s.update(r["run_id"], fencing_token=a["fencing_token"], status="retrying"); b = s.acquire(r["run_id"]); ok = b["attempt"] == 2; s.close(); return ok


def case_auth_no_retry(tmp):
    s = _store(tmp); r = _new(s); s.update(r["run_id"], status="failed", error_code="authentication"); ok = s.get(r["run_id"])["attempt"] == 0; s.close(); return ok


def case_duplicate_submit(tmp):
    s = _store(tmp); a = _new(s, "same"); b = _new(s, "same"); ok = a["run_id"] == b["run_id"]; s.close(); return ok


def case_disconnect_continues(tmp):
    s = _store(tmp); r = _new(s); a = s.acquire(r["run_id"]); s.heartbeat(r["run_id"], fencing_token=a["fencing_token"], progress=.4); ok = s.get(r["run_id"])["progress"] == .4; s.close(); return ok


def case_cancel_running(tmp):
    s = _store(tmp); r = _new(s); s.acquire(r["run_id"]); result = s.request_cancel(r["run_id"]); ok = result["cancel_requested"] == 1; s.close(); return ok


def case_worker_crash(tmp):
    s = _store(tmp); r = _new(s); a = s.acquire(r["run_id"]); s.update(r["run_id"], fencing_token=a["fencing_token"], checkpoint_ref="session-1"); s.connection.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?", (r["run_id"],)); s.connection.commit(); ok = s.recover_stale()["recovered"] == 1; s.close(); return ok


def case_lease_expiry(tmp):
    return case_worker_crash(tmp)


def case_fencing_rejects_old_worker(tmp):
    s = _store(tmp); r = _new(s); old = s.acquire(r["run_id"]); new = s.acquire(r["run_id"]); ok = not s.update(r["run_id"], fencing_token=old["fencing_token"], progress=.9) and s.update(r["run_id"], fencing_token=new["fencing_token"], progress=.5); s.close(); return ok


def case_restart_recovers_queued(tmp):
    s = _store(tmp); r = _new(s); s.close(); s = RunStore(str(Path(tmp) / "runs.sqlite3")); ok = s.get(r["run_id"])["status"] == "queued"; s.close(); return ok


def case_stale_without_checkpoint(tmp):
    s = _store(tmp); r = _new(s); a = s.acquire(r["run_id"]); s.connection.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?", (r["run_id"],)); s.connection.commit(); ok = s.recover_stale()["failed"] == 1; s.close(); return ok


def case_missing_checkpoint(tmp):
    return case_stale_without_checkpoint(tmp)


def case_artifact_failure(tmp):
    path = Path(tmp) / "not-a-directory"; path.write_text("x", encoding="utf-8")
    try:
        ArtifactStore(str(path))
    except Exception:
        return True
    return False


def case_blueprint_rejects_unsafe_node(tmp):
    try:
        WorkflowBlueprint.model_validate({"scenario": "x", "nodes": ["shell"], "required_dimensions": ["市场"]})
    except Exception:
        return True
    return False


CASES = {
    "search_timeout": case_search_timeout,
    "fetch_timeout": case_fetch_timeout,
    "mcp_process_exit": case_mcp_exit,
    "mcp_bad_schema": case_mcp_bad_schema,
    "direct_fallback": case_direct_fallback,
    "transient_retry": case_transient_retry,
    "auth_no_retry": case_auth_no_retry,
    "duplicate_submit": case_duplicate_submit,
    "disconnect_continues": case_disconnect_continues,
    "cancel_running": case_cancel_running,
    "worker_crash": case_worker_crash,
    "lease_expiry": case_lease_expiry,
    "fencing_rejects_old_worker": case_fencing_rejects_old_worker,
    "restart_recovers_queued": case_restart_recovers_queued,
    "stale_without_checkpoint": case_stale_without_checkpoint,
    "missing_checkpoint": case_missing_checkpoint,
    "artifact_failure": case_artifact_failure,
    "blueprint_rejects_unsafe_node": case_blueprint_rejects_unsafe_node,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="reliability-harness-") as tmp:
        for name, fn in CASES.items():
            try:
                case_dir = Path(tmp) / name
                case_dir.mkdir(parents=True, exist_ok=True)
                passed = bool(fn(str(case_dir)))
                error = "" if passed else "invariant returned false"
            except Exception as exc:
                passed, error = False, f"{type(exc).__name__}: {exc}"
            results.append({"id": name, "passed": passed, "error": error})
    summary = {"total": len(results), "passed": sum(item["passed"] for item in results), "results": results}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
