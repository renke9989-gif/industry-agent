"""Starlette + SSE API for resumable industry-research sessions."""
from __future__ import annotations

import json
import os
import sys
import uuid
import atexit
import time
import threading
import asyncio
import re
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from graph import compile_graph
from checkpointing import build_checkpointer
from state import initial_state
from providers import get_default_provider
from followup import answer_followup
from output_format import to_plain_text
from config import load_settings
from observability import log_runtime_event
from user_memory import get_facts, save_confirmed_fact, delete_facts
from research_tools import independent_source_count
from session_catalog import SessionCatalog
from run_store_factory import build_run_store
from artifact_store import ArtifactStore
from queue_backend import RedisStreams
from workflow_blueprint import blueprint_from_state, compile_blueprint
from private_rag import PrivateDocumentStore


load_dotenv()
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHECKPOINT_WARNING = ""
try:
    CHECKPOINTER, CHECKPOINTER_CONNECTION = build_checkpointer(persistent=True)
except RuntimeError as exc:
    # Keep the demo startable when the optional SQLite integration has not yet
    # been installed; health exposes the fallback so it is never silent.
    CHECKPOINT_WARNING = str(exc)
    from langgraph.checkpoint.memory import MemorySaver
    CHECKPOINTER, CHECKPOINTER_CONNECTION = MemorySaver(), None
AGENT = compile_graph(checkpointer=CHECKPOINTER)
SESSION_CATALOG = SessionCatalog()
SESSION_CATALOG.import_checkpoint_threads(CHECKPOINTER_CONNECTION)
RUN_STORE = build_run_store()
RUN_RECOVERY = RUN_STORE.recover_stale()
ARTIFACT_STORE = ArtifactStore()
PRIVATE_RAG = PrivateDocumentStore(os.getenv("PRIVATE_RAG_SQLITE_PATH", ".data/private_rag.sqlite3"))
EXECUTION_BACKEND = os.getenv("EXECUTION_BACKEND", "local").lower()
QUEUE = RedisStreams() if EXECUTION_BACKEND == "redis" else None
if CHECKPOINTER_CONNECTION is not None:
    atexit.register(CHECKPOINTER_CONNECTION.close)
atexit.register(SESSION_CATALOG.close)
atexit.register(RUN_STORE.close)
if CHECKPOINT_WARNING:
    print(f"[checkpoint] WARNING: {CHECKPOINT_WARNING}", flush=True)
else:
    print("[checkpoint] SQLite persistence enabled", flush=True)
RESEARCH_PROVIDER = get_default_provider()
SESSIONS: Dict[str, dict] = {}
CANCEL_EVENTS: Dict[str, threading.Event] = {}
ACTIVE_SESSIONS: set[str] = set()
SETTINGS = load_settings()
MAX_CONCURRENT_RESEARCH = SETTINGS.max_concurrent_research
RESEARCH_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_RESEARCH)
SESSION_REQUESTS: Dict[str, list[float]] = {}
RECOVERY_TASKS: set[asyncio.Task] = set()

ROLE_INFO = {
    "supervisor": {"name": "协调监督者", "icon": "🤖", "color": "#6366f1"},
    "market_analyst": {"name": "市场趋势分析师", "icon": "📈", "color": "#0ea5e9"},
    "competition_analyst": {"name": "竞争风险分析师", "icon": "⚔️", "color": "#ef4444"},
    "business_analyst": {"name": "商业模式分析师", "icon": "💰", "color": "#f59e0b"},
    "report_generator": {"name": "行业研究报告", "icon": "📄", "color": "#10b981"},
    "direct_answer": {"name": "行业研究助手", "icon": "💬", "color": "#0ea5e9"},
    "report_confirmation": {"name": "研究助手", "icon": "💡", "color": "#f59e0b"},
}


def create_session_id(user_id: str = "anonymous") -> str:
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"created": True}
    CANCEL_EVENTS[session_id] = threading.Event()
    SESSION_CATALOG.register(session_id, user_id=user_id)
    return session_id


def session_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 30}


def _snapshot(session_id: str) -> dict:
    try:
        snapshot = AGENT.get_state(session_config(session_id))
        return dict(snapshot.values) if snapshot and snapshot.values else {}
    except Exception:
        return {}


def _session_history(state: dict) -> list[dict]:
    """Return user-visible history while hiding internal supervisor routing."""
    history = []
    pending_question = str(state.get("pending_question", "")).strip()
    for message in state.get("messages", [])[-100:]:
        content = to_plain_text(getattr(message, "content", "")).strip()
        if not content or content.startswith("[Supervisor 决策]"):
            continue
        if isinstance(message, HumanMessage):
            item = {"type": "user", "name": "你", "icon": "👤", "content": content}
        elif isinstance(message, AIMessage):
            # The current structured question is restored separately so its
            # answer buttons and interview progress remain available.
            if pending_question and content == pending_question and state.get("awaiting_user"):
                continue
            is_report = "\n来源\n" in content and bool(
                state.get("report_generated") or state.get("citation_status") == "insufficient"
            )
            item = {
                "type": "report" if is_report else "analyst",
                "name": "行业研究报告" if is_report else "研究过程",
                "icon": "📄" if is_report else "💡",
                "content": content,
            }
            if is_report:
                item["report_generated"] = bool(state.get("report_generated"))
                item["citation_status"] = state.get("citation_status", "pending")
        else:
            continue
        if not history or history[-1]["content"] != item["content"]:
            history.append(item)
    return history


def _message_usage(messages) -> dict:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    for message in messages or []:
        metadata = getattr(message, "usage_metadata", None) or {}
        if not metadata:
            metadata = (getattr(message, "response_metadata", None) or {}).get("token_usage", {})
        if not metadata:
            continue
        usage["input_tokens"] += int(metadata.get("input_tokens", metadata.get("prompt_tokens", 0)) or 0)
        usage["output_tokens"] += int(metadata.get("output_tokens", metadata.get("completion_tokens", 0)) or 0)
        usage["total_tokens"] += int(metadata.get("total_tokens", 0) or 0)
        usage["llm_calls"] += 1
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _estimate_cost(usage: dict) -> float | None:
    input_rate = os.getenv("LLM_INPUT_COST_PER_1K")
    output_rate = os.getenv("LLM_OUTPUT_COST_PER_1K")
    if not input_rate or not output_rate:
        return None
    return round(
        usage.get("input_tokens", 0) / 1000 * float(input_rate)
        + usage.get("output_tokens", 0) / 1000 * float(output_rate),
        6,
    )


def _is_cancelled(session_id: str) -> bool:
    return CANCEL_EVENTS.setdefault(session_id, threading.Event()).is_set()


async def _graph_events_async(graph_input: dict, config: dict, cancel_event: threading.Event,
                              run: dict | None = None, timeout_seconds: int | None = None):
    """Run the synchronous graph off the event loop while preserving SSE events."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for index, event in enumerate(AGENT.stream(graph_input, config), start=1):
                if cancel_event.is_set():
                    break
                if run:
                    node = next(iter(event), "")
                    RUN_STORE.heartbeat(run["run_id"], fencing_token=run["fencing_token"],
                                        progress=min(0.95, index / 12), current_node=node)
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            # The worker owns final lifecycle persistence. This still runs if
            # the browser disconnects and the async SSE consumer is cancelled.
            if run:
                if cancel_event.is_set() or (RUN_STORE.get(run["run_id"]) or {}).get("cancel_requested"):
                    _finish_run(run, status="cancelled", current_node="cancelled", error_code="cancelled")
                else:
                    detached_state = _snapshot(config["configurable"]["thread_id"])
                    try:
                        artifact_ref = ARTIFACT_STORE.save(run["run_id"], _artifact_payload(run, detached_state))
                        RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"], artifact_ref=artifact_ref)
                    except Exception as artifact_exc:
                        RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"],
                                         error_code="artifact_write_failed", error_message=str(artifact_exc)[:500])
                    _finish_run(run)
        except Exception as exc:
            if run:
                RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"], status="failed",
                                 finished_at=time.time(), error_code=type(exc).__name__, error_message=str(exc)[:500])
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    worker = threading.Thread(target=producer, name=f"research-{config['configurable']['thread_id'][:8]}", daemon=True)
    worker.start()
    deadline = time.monotonic() + (timeout_seconds or int(os.getenv("RUN_TIMEOUT_SECONDS", "300")))
    while True:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            cancel_event.set()
            if run:
                RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"], status="timed_out",
                                 finished_at=time.time(), error_code="run_timeout", error_message="overall run timeout")
            raise TimeoutError("research run timed out") from exc
        if kind == "event":
            yield payload
        elif kind == "error":
            raise payload
        else:
            break


def _finish_run(run: dict | None, *, status: str = "succeeded", current_node: str = "done",
                error_code: str | None = None, error_message: str | None = None) -> None:
    if not run:
        return
    RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"], status=status,
                     progress=1.0 if status == "succeeded" else float(run.get("progress") or 0),
                     current_node=current_node, finished_at=time.time(), error_code=error_code,
                     error_message=error_message)


def _artifact_payload(run: dict, state: dict) -> dict:
    report = ""
    if state.get("report_generated"):
        report = next((to_plain_text(getattr(item, "content", "")) for item in reversed(state.get("messages", []))
                       if to_plain_text(getattr(item, "content", "")).strip()), "")
    return {
        "run_id": run["run_id"],
        "session_id": run["session_id"],
        "report": report,
        "evidence": state.get("evidence", []),
        "trace_events": state.get("trace_events", []),
        "run_metrics": state.get("last_run_metrics", {}),
        "completion_reason": state.get("completion_reason", ""),
    }


def _input_for_message(session_id: str, message: str, user_id: str = "anonymous") -> dict:
    previous = _snapshot(session_id)
    if not previous:
        state = initial_state(HumanMessage(content=message))
        state["known_facts"] = get_facts(user_id)
        state["context_token_budget"] = SETTINGS.context_token_budget
        return state
    status = dict(previous.get("dimension_status", {}))
    if previous.get("awaiting_user") and any(
        str(item).startswith("business:") for item in previous.get("missing_information", [])
    ):
        status["商业模式"] = "pending"
    known_facts = {**get_facts(user_id), **dict(previous.get("known_facts", {}))}
    covered_topics = list(previous.get("covered_topics", []))
    if previous.get("awaiting_user") and previous.get("pending_question"):
        current_question = previous.get("current_interview_question", {})
        topic = current_question.get("topic") or f"interview_answer_{previous.get('interview_round', 0)}"
        known_facts[topic] = message
        if topic not in covered_topics:
            covered_topics.append(topic)
    decision_goal = message if previous.get("current_interview_question", {}).get("topic") == "decision_goal" else previous.get("decision_goal", "")
    return {
        "messages": [HumanMessage(content=message)],
        "awaiting_user": False,
        "interview_complete": previous.get("interview_complete", False),
        "pending_question": "",
        "missing_information": [],
        "dimension_status": status,
        "research_complete": False,
        "next_agent": "",
        "user_turn_count": previous.get("user_turn_count", 1) + 1,
        "interview_round": previous.get("interview_round", 0),
        "report_confirmation_pending": previous.get("report_confirmation_pending", False),
        "report_confirmation_answer": previous.get("report_confirmation_answer", ""),
        "known_facts": known_facts,
        "covered_topics": covered_topics,
        "current_interview_question": {},
        "decision_goal": decision_goal,
        "context_token_budget": SETTINGS.context_token_budget,
    }


async def event_generator(session_id: str, user_input: str, user_id: str = "anonymous", run_id: str | None = None,
                          force_local: bool = False):
    if EXECUTION_BACKEND == "redis" and not force_local:
        async for event in redis_event_generator(session_id, user_input, user_id, run_id):
            yield event
        return
    cancel_event = CANCEL_EVENTS.setdefault(session_id, threading.Event())
    cancel_event.clear()
    run = RUN_STORE.get(run_id) if run_id else None
    if run and run["status"] in {"succeeded", "failed", "timed_out", "cancelled"}:
        yield {"event": "init", "data": json.dumps({"type": "init", "session_id": session_id, "run_id": run_id}, ensure_ascii=False)}
        yield {"event": "done", "data": json.dumps({"type": "done", "session_id": session_id,
            "run_id": run_id, "run_status": run["status"], "replayed": True}, ensure_ascii=False)}
        return
    if run:
        run = RUN_STORE.acquire(run_id, owner_id=f"web-{threading.get_ident()}")
        RUN_STORE.update(run_id, fencing_token=run["fencing_token"], checkpoint_ref=session_id,
                         current_node="supervisor")
    yield {
        "event": "init",
        "data": json.dumps({"type": "init", "session_id": session_id, "run_id": run_id}, ensure_ascii=False),
    }
    config = session_config(session_id)
    existing = _snapshot(session_id)
    try:
        # The report confirmation gate is explicit and idempotent. Button labels
        # are converted to stable commands before entering the graph.
        if existing.get("report_confirmation_pending"):
            normalized = user_input.strip()
            if normalized in ("生成报告", "没有，生成报告", "generate", "no"):
                graph_input = _input_for_message(session_id, user_input, user_id)
                graph_input.update({"report_confirmation_pending": False, "report_confirmation_answer": "generate",
                                    "research_complete": False, "awaiting_user": False})
                async for event in _graph_events_async(graph_input, config, cancel_event, run):
                    for node_name, output in event.items():
                        info = ROLE_INFO.get(node_name, {"name": node_name, "icon": "📌", "color": "#64748b"})
                        if node_name == "report_generator":
                            for message in output.get("messages", []):
                                yield {"event": "message", "data": json.dumps({
                                    "type": "report", "node": node_name, **info,
                                    "content": str(getattr(message, "content", "")),
                                    "report_generated": output.get("report_generated", False),
                                    "citation_status": output.get("citation_status", "pending"),
                                }, ensure_ascii=False)}
                final = _snapshot(session_id)
                _finish_run(run, current_node="report_generator")
                yield {"event": "done", "data": json.dumps({"type": "done", "session_id": session_id,
                    "awaiting_user": False, "research_complete": True, "report_generated": final.get("report_generated", False),
                    "citation_status": final.get("citation_status", "pending"),
                    "industry": final.get("industry", "unknown"), "evidence_count": len(final.get("evidence", []))}, ensure_ascii=False)}
                return
            # Any other text is the user's real follow-up question. Answer it
            # narrowly, preserve all evidence, then show the confirmation gate
            # again instead of rerunning the complete research graph.
            AGENT.update_state(config, {"report_confirmation_pending": False, "report_confirmation_answer": "continue",
                                        "awaiting_user": False, "followup_mode": True})
            existing = _snapshot(session_id)
            result = answer_followup(existing, user_input)
            AGENT.update_state(config, {
                "messages": [HumanMessage(content=user_input), *result["messages"]],
                "evidence": result["evidence"],
                "tool_events": result["tool_events"],
                "region": result.get("region", existing.get("region", "全国")),
                "research_complete": True,
                "report_confirmation_pending": True,
                "report_confirmation_answer": "",
                "awaiting_user": True,
                "followup_mode": False,
                "user_turn_count": existing.get("user_turn_count", 1) + 1,
                "context_metrics": result.get("context_metrics", existing.get("context_metrics", {})),
                "selected_skill": "followup_answer",
            })
            info = ROLE_INFO["direct_answer"]
            yield {"event": "message", "data": json.dumps({
                "type": "followup", "node": "followup", **info,
                "content": result["messages"][0].content,
                "selected_skill": "followup_answer",
                "context_metrics": result.get("context_metrics", {}),
            }, ensure_ascii=False)}
            confirm = ROLE_INFO["report_confirmation"]
            yield {"event": "message", "data": json.dumps({
                "type": "report_confirmation", "node": "report_confirmation", **confirm,
                "content": "这个问题已经回答。您还想继续提问，还是现在生成正式报告？",
            }, ensure_ascii=False)}
            final = _snapshot(session_id)
            _finish_run(run, current_node="followup")
            yield {"event": "done", "data": json.dumps({
                "type": "done", "session_id": session_id, "awaiting_user": True,
                "research_complete": True, "report_confirmation_pending": True,
                "industry": final.get("industry", "unknown"),
                "evidence_count": len(final.get("evidence", [])), "followup": True,
            }, ensure_ascii=False)}
            return

        # A completed report switches to focused Q&A. It never routes back
        # through the report generator for a normal follow-up message.
        if existing.get("research_complete") and (existing.get("report_generated") or existing.get("citation_status") == "insufficient") and not existing.get("awaiting_user"):
            result = answer_followup(existing, user_input)
            AGENT.update_state(config, {
                "messages": [HumanMessage(content=user_input), *result["messages"]],
                "evidence": result["evidence"],
                "tool_events": result["tool_events"],
                "region": result.get("region", existing.get("region", "全国")),
                "research_complete": True,
                "context_metrics": result.get("context_metrics", existing.get("context_metrics", {})),
                "selected_skill": "followup_answer",
            })
            info = {"name": "行业研究助手", "icon": "💬", "color": "#0ea5e9"}
            yield {
                "event": "message",
                "data": json.dumps({"type": "followup", "node": "followup", **info,
                                    "content": result["messages"][0].content,
                                    "selected_skill": "followup_answer",
                                    "context_metrics": result.get("context_metrics", {})}, ensure_ascii=False),
            }
            final = _snapshot(session_id)
            _finish_run(run, current_node="followup")
            yield {
                "event": "done",
                "data": json.dumps({"type": "done", "session_id": session_id,
                                    "awaiting_user": False, "research_complete": True,
                                    "industry": final.get("industry", "unknown"),
                                    "scores": final.get("scores", {}),
                                    "status": final.get("dimension_status", {}),
                                    "evidence_count": len(final.get("evidence", [])),
                                    "claim_entailment_rate": final.get("claim_entailment_rate"),
                                    "claim_judge_status": final.get("claim_judge_status", "unavailable"),
                                    "user_turn_count": final.get("user_turn_count", 1) + 1,
                                    "agent_step_count": final.get("agent_step_count", 0),
                                    "followup": True}, ensure_ascii=False),
            }
            return

        graph_input = _input_for_message(session_id, user_input, user_id)
        usage_before = dict(existing.get("llm_usage", {}))
        run_started = time.perf_counter()
        stream_cursor = run_started
        trace_events = []
        run_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        async for event in _graph_events_async(graph_input, config, cancel_event, run):
            if cancel_event.is_set():
                AGENT.update_state(config, {"awaiting_user": True, "research_complete": False,
                                            "completion_reason": "用户取消了本次研究", "force_finish": True})
                yield {"event": "cancelled", "data": json.dumps({"type": "cancelled", "session_id": session_id,
                    "message": "研究已停止，已保留当前证据和进度。"}, ensure_ascii=False)}
                return
            node_finished = time.perf_counter()
            node_duration_ms = int((node_finished - stream_cursor) * 1000)
            stream_cursor = node_finished
            for node_name, output in event.items():
                info = ROLE_INFO.get(node_name, {"name": node_name, "icon": "📌", "color": "#64748b"})
                node_trace = {
                    "node": node_name,
                    "duration_ms": node_duration_ms,
                    "tool_event_count": len(output.get("tool_events", [])),
                    "success": True,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "selected_skill": output.get("selected_skill", final.get("selected_skill", "") if 'final' in locals() else ""),
                    "context_before_tokens": output.get("context_metrics", {}).get("before_tokens", 0),
                    "context_after_tokens": output.get("context_metrics", {}).get("after_tokens", 0),
                    "context_compressed": bool(output.get("context_metrics", {}).get("compression_count", 0)),
                }
                trace_events.append(node_trace)
                yield {"event": "node_trace", "data": json.dumps({"type": "node_trace", **node_trace}, ensure_ascii=False)}
                yield {
                    "event": "agent",
                    "data": json.dumps({
                        "type": "agent_active", "node": node_name, **info,
                    }, ensure_ascii=False),
                }
                if output.get("scores"):
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "type": "progress",
                            "scores": output.get("scores", {}),
                            "status": output.get("dimension_status", {}),
                        }, ensure_ascii=False),
                    }
                if output.get("supervisor_reason") and output.get("next_agent") not in ("WAIT_USER", "REPORT_CONFIRMATION"):
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "supervisor", "node": node_name, **info,
                            "content": output["supervisor_reason"],
                            "decision": output.get("next_agent", ""),
                            "scenario": output.get("scenario", "unknown"),
                            "decision_goal": output.get("decision_goal", ""),
                            "research_plan": output.get("research_plan", []),
                            "research_stage": output.get("research_stage", ""),
                            "selected_skill": output.get("selected_skill", ""),
                            "skill_confidence": output.get("skill_confidence", 0),
                            "skill_fallback_reason": output.get("skill_fallback_reason", ""),
                            "routing_classifier": output.get("routing_classifier", "rules"),
                        }, ensure_ascii=False),
                    }
                for message in output.get("messages", []):
                    content = to_plain_text(getattr(message, "content", ""))
                    if not content or content.startswith("[Supervisor 决策]"):
                        continue
                    message_type = "report" if node_name == "report_generator" else "analyst"
                    if output.get("answer_type") == "report_confirmation":
                        message_type = "report_confirmation"
                    elif output.get("awaiting_user") or output.get("next_agent") == "WAIT_USER":
                        message_type = "question"
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": message_type, "node": node_name, **info, "content": content,
                            "answer_type": output.get("answer_type", ""),
                            "question": output.get("current_interview_question", {}),
                            "interview_round": output.get("interview_round", 0),
                            "interview_target_questions": output.get("interview_target_questions", 0),
                            "interview_readiness_score": output.get("interview_readiness_score", 0),
                            "citation_status": output.get("citation_status", ""),
                            "citation_validation_error": output.get("citation_validation_error", ""),
                            "report_generated": output.get("report_generated", False),
                            "claim_entailment_rate": output.get("claim_entailment_rate"),
                            "claim_judge_status": output.get("claim_judge_status", "unavailable"),
                            "user_conditions": output.get("user_conditions", []),
                            "recommendation_traces": output.get("recommendation_traces", []),
                            "recommendation_trace_status": output.get("recommendation_trace_status", "pending"),
                        }, ensure_ascii=False),
                    }

                if output.get("tool_events"):
                    latest = output["tool_events"][-1]
                    yield {
                        "event": "trace",
                        "data": json.dumps({"type": "trace", "event": latest}, ensure_ascii=False),
                    }

        # If cancellation arrives while a long synchronous tool is running,
        # the producer stops before publishing another node event. Handle that
        # boundary here so the client still receives an explicit cancellation.
        if cancel_event.is_set():
            AGENT.update_state(config, {"awaiting_user": True, "research_complete": False,
                                        "completion_reason": "用户取消了本次研究", "force_finish": True})
            _finish_run(run, status="cancelled", current_node="cancelled", error_code="cancelled")
            yield {"event": "cancelled", "data": json.dumps({"type": "cancelled", "session_id": session_id,
                "message": "研究已停止，已保留当前证据和进度。"}, ensure_ascii=False)}
            return

        final = _snapshot(session_id)
        cumulative_usage = final.get("llm_usage", {})
        for key in run_usage:
            run_usage[key] = max(0, int(cumulative_usage.get(key, 0) or 0) - int(usage_before.get(key, 0) or 0))
        total_duration_ms = int((time.perf_counter() - run_started) * 1000)
        tool_events = list(final.get("tool_events", []))
        run_metrics = {
            "duration_ms": total_duration_ms,
            "node_count": len(trace_events),
            "tool_call_count": len(tool_events),
            "tool_failure_count": sum(not bool(item.get("success")) for item in tool_events),
            "search_call_count": sum(item.get("tool") == "search" for item in tool_events),
            "fetch_page_count": sum(item.get("tool") == "fetch_page" for item in tool_events),
            "mcp_failure_count": sum(item.get("tool_source") == "mcp" and not bool(item.get("success")) for item in tool_events),
            "retry_count": sum(int(item.get("retry_count", 0) or 0) for item in tool_events),
            "llm_calls": run_usage["llm_calls"],
            "input_tokens": run_usage["input_tokens"],
            "output_tokens": run_usage["output_tokens"],
            "total_tokens": run_usage["total_tokens"],
            "estimated_cost": _estimate_cost(run_usage),
            "claim_entailment_rate": final.get("claim_entailment_rate"),
            "claim_judge_status": final.get("claim_judge_status", "unavailable"),
            "recommendation_trace_status": final.get("recommendation_trace_status", "pending"),
            "context_before_tokens": final.get("context_metrics", {}).get("before_tokens", 0),
            "context_after_tokens": final.get("context_metrics", {}).get("after_tokens", 0),
            "context_budget": final.get("context_metrics", {}).get("budget", SETTINGS.context_token_budget),
            "context_compression_count": final.get("context_metrics", {}).get("compression_count", 0),
            "context_overflow_count": final.get("context_metrics", {}).get("overflow_count", 0),
            "context_compression_ratio": final.get("context_metrics", {}).get("compression_ratio", 1.0),
            "recommendation_trace_rate": (
                sum(item.get("trace_status") == "validated" for item in final.get("recommendation_traces", []))
                / len(final.get("recommendation_traces", [])) if final.get("recommendation_traces") else 0
            ),
            "retry_policy": "node_max_attempts_2",
        }
        active_dimensions = [name for name, value in final.get("dimension_status", {}).items()
                             if value in ("complete", "insufficient", "researching")]
        if active_dimensions:
            total_cost = run_metrics.get("estimated_cost")
            run_metrics["cost_by_dimension"] = ({name: round(total_cost / len(active_dimensions), 6)
                                                   for name in active_dimensions}
                                                  if total_cost is not None else None)
            run_metrics["dimension_cost_method"] = "equal_split_estimate"
        AGENT.update_state(config, {
            "trace_events": list(final.get("trace_events", [])) + trace_events,
            "last_run_metrics": run_metrics,
        })
        log_runtime_event(
            "research_run_completed",
            session_id=session_id,
            industry=final.get("industry", "unknown"),
            completion_reason=final.get("completion_reason", ""),
            metrics=run_metrics,
            provider=RESEARCH_PROVIDER.active_provider,
        )
        final = _snapshot(session_id)
        if run:
            try:
                artifact_ref = ARTIFACT_STORE.save(run["run_id"], _artifact_payload(run, final))
                RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"], artifact_ref=artifact_ref)
            except Exception as artifact_exc:
                RUN_STORE.update(run["run_id"], fencing_token=run["fencing_token"],
                                 error_code="artifact_write_failed", error_message=str(artifact_exc)[:500])
        if run and (RUN_STORE.get(run["run_id"]) or {}).get("status") not in {"succeeded", "failed", "timed_out", "cancelled"}:
            _finish_run(run)
        yield {
            "event": "done",
            "data": json.dumps({
                "type": "done",
                "session_id": session_id,
                "awaiting_user": final.get("awaiting_user", False),
                "research_complete": final.get("research_complete", False),
                "industry": final.get("industry", "unknown"),
                "scores": final.get("scores", {}),
                "status": final.get("dimension_status", {}),
                "evidence_count": len(final.get("evidence", [])),
                "user_turn_count": final.get("user_turn_count", 1),
                "agent_step_count": final.get("agent_step_count", 0),
                "run_metrics": final.get("last_run_metrics", {}),
                "claim_entailment_rate": final.get("claim_entailment_rate"),
                "claim_judge_status": final.get("claim_judge_status", "unavailable"),
            }, ensure_ascii=False),
        }
    except Exception as exc:
        _finish_run(run, status="failed", current_node="error", error_code=type(exc).__name__,
                    error_message=str(exc)[:500])
        log_runtime_event("research_run_failed", session_id=session_id,
                          error_type=type(exc).__name__, error=str(exc)[:500])
        yield {
            "event": "error",
            "data": json.dumps({
                "type": "error", "message": f"{type(exc).__name__}: {str(exc)}",
            }, ensure_ascii=False),
        }


async def index(request: Request):
    html_path = os.path.join(os.path.dirname(__file__), "web_ui", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as handle:
            return Response(handle.read(), media_type="text/html")
    except FileNotFoundError:
        return Response("<h1>web_ui/index.html missing</h1>", media_type="text/html", status_code=404)


async def create_session(request: Request):
    user_id = str(request.headers.get("x-user-id") or "anonymous")[:128]
    session_id = create_session_id(user_id)
    return JSONResponse({"session_id": session_id})


async def list_sessions(request: Request):
    user_id = str(request.headers.get("x-user-id") or "anonymous")[:128]
    sessions = SESSION_CATALOG.list_recent(limit=50, user_id=user_id)
    return JSONResponse({"sessions": sessions})


async def send_message(request: Request):
    session_id = request.path_params.get("session_id") or ""
    if not session_id:
        body = await request.json()
        session_id = body.get("session_id") or create_session_id()
    else:
        body = await request.json()
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"created": True}
    message = str(body.get("message", "")).strip()
    user_id = str(request.headers.get("x-user-id") or body.get("user_id") or "anonymous")[:128]
    if not message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)
    # A fresh user turn must explicitly choose research depth before any
    # network/tool execution. This guard lives at the API boundary so stale
    # browser state or an old worker cannot bypass the consent card.
    previous_state = _snapshot(session_id)
    session_meta = SESSIONS.get(session_id, {})
    if session_meta.get("depth_choice_request") and not previous_state.get("depth_choice_request"):
        previous_state["depth_choice_request"] = session_meta["depth_choice_request"]
        previous_state["interview_consent_pending"] = bool(session_meta.get("interview_consent_pending"))
    consent_commands = {"进入访谈（推荐）", "进入访谈", "直接开始调研", "跳过访谈"}
    if previous_state.get("interview_consent_pending") and message in consent_commands:
        original = str(previous_state.get("depth_choice_request") or "")
        if not original:
            return JSONResponse({"error": "深度选择已过期，请重新发送问题"}, status_code=409)
        if message in {"直接开始调研", "跳过访谈"}:
            message = original + " 直接开始调研"
        else:
            message = original + " 进入访谈"
        try:
            AGENT.update_state(session_config(session_id), {"interview_consent_pending": False,
                "awaiting_user": False, "pending_question": "", "current_interview_question": {}})
        except Exception:
            pass
        SESSIONS.setdefault(session_id, {})["interview_consent_pending"] = False
    if (not previous_state.get("interview_consent_pending")
            and not previous_state.get("awaiting_user")
            and message not in consent_commands):
        from triage import triage_request
        triage = triage_request(message)
        if triage.get("needs_interview"):
            return EventSourceResponse(_depth_choice_generator(session_id, message, user_id))
    SESSION_CATALOG.touch(session_id, user_id=user_id, first_message=message)
    if len(message) > SETTINGS.max_message_chars:
        return JSONResponse({"error": f"消息过长，最多 {SETTINGS.max_message_chars} 个字符"}, status_code=413)
    active = RUN_STORE.active_for_session(session_id)
    if active:
        return JSONResponse({"error": "该会话已有研究任务运行中", "run_id": active["run_id"],
                             "status": active["status"]}, status_code=409)
    run = RUN_STORE.create_or_get(session_id, user_id=user_id, message=message)
    now = time.time()
    recent = [value for value in SESSION_REQUESTS.get(session_id, []) if now - value < 60]
    if len(recent) >= SETTINGS.session_rate_limit_per_minute:
        return JSONResponse({"error": "请求过于频繁，请稍后再试"}, status_code=429)
    SESSION_REQUESTS[session_id] = recent + [now]
    if EXECUTION_BACKEND == "redis":
        try:
            QUEUE.enqueue({"run_id": run["run_id"], "session_id": session_id, "user_id": user_id,
                           "message": message, "created_at": now})
        except Exception as exc:
            RUN_STORE.update(run["run_id"], status="failed", finished_at=time.time(),
                             error_code="queue_unavailable", error_message=str(exc)[:500])
            return JSONResponse({"error": "任务队列暂不可用", "run_id": run["run_id"]}, status_code=503)
    return EventSourceResponse(limited_event_generator(session_id, message, user_id, run["run_id"]))


async def _depth_choice_generator(session_id: str, message: str, user_id: str):
    """Emit the depth-choice card without creating a research Run."""
    config = session_config(session_id)
    existing = _snapshot(session_id)
    state = _input_for_message(session_id, message, user_id)
    SESSIONS.setdefault(session_id, {})["depth_choice_request"] = message
    SESSIONS[session_id]["interview_consent_pending"] = True
    state.update({"interview_consent_pending": True, "awaiting_user": True,
                  "pending_question": "是否进入深度访谈？", "depth_choice_request": message, "current_interview_question": {
                      "id": "depth_choice", "topic": "depth_choice", "title": "研究方式确认",
                      "question": "是否进入深度访谈，再开始联网调研？",
                      "why": "深度访谈会先确认目标、地区、预算等影响结论的条件。",
                      "options": ["进入深度访谈（推荐）", "直接开始调研"], "allow_custom": False,
                  }})
    try:
        AGENT.update_state(config, state)
    except Exception:
        pass
    payload = {"type": "question", "node": "supervisor", "name": "协调监督者", "icon": "🤖", "color": "#6366f1",
               "content": "开始研究前，请选择研究深度。", "question": state["current_interview_question"],
               "interview_round": 0, "interview_target_questions": 6, "interview_readiness_score": 0}
    yield {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}
    yield {"event": "done", "data": json.dumps({"type": "done", "session_id": session_id,
                                                    "awaiting_user": True, "depth_choice_pending": True}, ensure_ascii=False)}


async def limited_event_generator(session_id: str, message: str, user_id: str = "anonymous", run_id: str | None = None):
    if session_id in ACTIVE_SESSIONS:
        yield {"event": "error", "data": json.dumps({
            "type": "error", "message": "该会话已有研究任务正在运行，请等待完成或先停止研究。"
        }, ensure_ascii=False)}
        return
    ACTIVE_SESSIONS.add(session_id)
    try:
        async with RESEARCH_SEMAPHORE:
            async for event in event_generator(session_id, message, user_id, run_id):
                yield event
    finally:
        ACTIVE_SESSIONS.discard(session_id)


async def redis_event_generator(session_id: str, message: str, user_id: str, run_id: str | None):
    yield {"event": "init", "data": json.dumps({"type": "init", "session_id": session_id, "run_id": run_id}, ensure_ascii=False)}
    after = "0-0"
    deadline = time.monotonic() + int(os.getenv("RUN_TIMEOUT_SECONDS", "300"))
    while time.monotonic() < deadline:
        try:
            entries = await asyncio.to_thread(QUEUE.events, run_id, after, 2000, 50)
        except Exception:
            await asyncio.sleep(1)
            run = RUN_STORE.get(run_id) if run_id else None
            if run and run.get("status") in {"succeeded", "failed", "timed_out", "cancelled"}:
                yield {"event": "done", "data": json.dumps({"type": "done", "run_status": run["status"], "run_id": run_id}, ensure_ascii=False)}
                return
            continue
        for stream_id, payload in entries:
            after = stream_id
            yield {"id": stream_id, "event": payload.get("type", "message"), "data": json.dumps(payload, ensure_ascii=False)}
            if payload.get("type") in {"done", "error", "cancelled"}:
                return
        run = RUN_STORE.get(run_id) if run_id else None
        if run and run.get("status") in {"succeeded", "failed", "timed_out", "cancelled"} and not entries:
            yield {"event": "done", "data": json.dumps({"type": "done", "run_status": run["status"], "run_id": run_id}, ensure_ascii=False)}
            return
    if run_id:
        RUN_STORE.update(run_id, status="timed_out", finished_at=time.time(), error_code="run_timeout")
    yield {"event": "error", "data": json.dumps({"type": "error", "message": "research run timed out", "run_id": run_id}, ensure_ascii=False)}


async def cancel_session(request: Request):
    session_id = request.path_params["session_id"]
    if session_id not in SESSIONS and not _snapshot(session_id):
        return JSONResponse({"error": "session not found"}, status_code=404)
    CANCEL_EVENTS.setdefault(session_id, threading.Event()).set()
    run = RUN_STORE.active_for_session(session_id)
    if run:
        RUN_STORE.request_cancel(run["run_id"])
    return JSONResponse({"session_id": session_id, "cancel_requested": True})


async def get_run(request: Request):
    run = RUN_STORE.get(request.path_params["run_id"])
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return JSONResponse(run)


async def list_runs(request: Request):
    return JSONResponse({"runs": RUN_STORE.list_recent(limit=50)})


async def private_documents(request: Request):
    """Index local private Markdown/TXT/PDF files or retrieve top chunks."""
    if request.method == "POST":
        body = await request.json()
        root = body.get("path") or body.get("directory")
        if not root:
            return JSONResponse({"error": "path is required"}, status_code=400)
        try:
            allowed = Path(os.getenv("PRIVATE_RAG_SOURCE_ROOT", ".data/private_docs")).resolve()
            requested = Path(root).resolve()
            if requested != allowed and allowed not in requested.parents:
                raise ValueError(f"path must be inside private document root: {allowed}")
            result = await asyncio.to_thread(PRIVATE_RAG.index_dir, requested, bool(body.get("recursive", True)))
        except Exception as exc:
            return JSONResponse({"error": str(exc)[:300]}, status_code=400)
        return JSONResponse({"indexed": result, "count": len(result)})
    query = request.query_params.get("q", "")
    hits = await asyncio.to_thread(PRIVATE_RAG.search, query, int(request.query_params.get("top_k", "5")))
    return JSONResponse({"query": query, "results": hits})


async def upload_private_document(request: Request):
    """Accept a browser multipart upload, save it inside the private root, and index it."""
    allowed = Path(os.getenv("PRIVATE_RAG_SOURCE_ROOT", ".data/private_docs")).resolve()
    allowed.mkdir(parents=True, exist_ok=True)
    try:
        async with request.form(max_files=1, max_fields=10) as form:
            upload = form.get("file")
            if upload is None or not hasattr(upload, "filename"):
                return JSONResponse({"error": "请选择一个文件"}, status_code=400)
            filename = Path(str(upload.filename or "")).name
            if not filename or filename in {".", ".."}:
                return JSONResponse({"error": "文件名无效"}, status_code=400)
            suffix = Path(filename).suffix.lower()
            if suffix not in {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".pdf"}:
                return JSONResponse({"error": "仅支持 Markdown、TXT、RST、CSV、JSON 和 PDF"}, status_code=415)
            # Keep uploads within the configured root and avoid path traversal.
            safe_name = re.sub(r"[^0-9A-Za-z一-龥._-]", "_", filename)[:160]
            target = allowed / safe_name
            content = await upload.read()
            max_bytes = int(os.getenv("PRIVATE_RAG_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
            if len(content) > max_bytes:
                return JSONResponse({"error": f"文件不能超过 {max_bytes // (1024 * 1024)} MB"}, status_code=413)
            target.write_bytes(content)
            indexed = await asyncio.to_thread(PRIVATE_RAG.index_file, target)
            # Uploading a document explicitly opts subsequent research turns
            # into private-context retrieval for this server process.
            os.environ["PRIVATE_RAG_ENABLED"] = "true"
            return JSONResponse({"uploaded": indexed, "private_rag_enabled": True,
                                 "message": "资料已索引，后续研究将自动参考该文件"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:300]}, status_code=400)


async def run_events(request: Request):
    run_id = request.path_params["run_id"]
    run = RUN_STORE.get(run_id)
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    if not QUEUE:
        return JSONResponse({"error": "event stream is disabled", "run": run}, status_code=409)
    after = request.query_params.get("after") or request.headers.get("last-event-id") or "0-0"
    try:
        entries = await asyncio.to_thread(QUEUE.events, run_id, after, 0, 100)
    except Exception as exc:
        return JSONResponse({"error": "event stream unavailable", "detail": str(exc)[:200], "run": run}, status_code=503)
    return JSONResponse({"run_id": run_id, "events": [{"id": sid, **payload} for sid, payload in entries], "run": run})


async def cancel_run(request: Request):
    run = RUN_STORE.request_cancel(request.path_params["run_id"])
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    CANCEL_EVENTS.setdefault(run["session_id"], threading.Event()).set()
    return JSONResponse(run)


async def memory_profile(request: Request):
    user_id = str(request.headers.get("x-user-id") or "anonymous")[:128]
    if request.method == "GET":
        return JSONResponse({"user_id": user_id, "facts": get_facts(user_id)})
    if request.method == "DELETE":
        delete_facts(user_id)
        return JSONResponse({"user_id": user_id, "deleted": True})
    body = await request.json()
    if body.get("confirmed") is not True:
        return JSONResponse({"error": "长期记忆只接受用户明确确认的事实"}, status_code=400)
    fact_key = str(body.get("key", "")).strip()[:100]
    if not fact_key:
        return JSONResponse({"error": "key 不能为空"}, status_code=400)
    save_confirmed_fact(user_id, fact_key, body.get("value"))
    return JSONResponse({"user_id": user_id, "saved": fact_key})


async def get_session(request: Request):
    session_id = request.path_params["session_id"]
    if session_id not in SESSIONS and not _snapshot(session_id):
        return JSONResponse({"error": "session not found"}, status_code=404)
    state = _snapshot(session_id)
    run = RUN_STORE.latest_for_session(session_id)
    try:
        blueprint = compile_blueprint(blueprint_from_state(state)) if state else None
    except Exception:
        blueprint = None
    evidence = [{
        "id": item.get("id", ""),
        "dimension": item.get("dimension", ""),
        "source_title": item.get("source_title", ""),
        "source_url": item.get("source_url", ""),
        "source_type": item.get("source_type", "web"),
        "published_at": item.get("published_at", ""),
        "retrieved_at": item.get("retrieved_at", ""),
        "excerpt": item.get("excerpt", "")[:500],
        "confidence": item.get("confidence", 0),
    } for item in state.get("evidence", [])]
    active_dimensions = list(dict.fromkeys(
        dimension
        for step in state.get("research_plan", [])
        for dimension in step.get("dimensions", [])
    ))
    return JSONResponse({
        "session_id": session_id,
        "run": run,
        "context_metrics": state.get("context_metrics", {}),
        "selected_skill": state.get("selected_skill", ""),
        "initial_skill": state.get("initial_skill", ""),
        "skill_candidates": state.get("skill_candidates", []),
        "skill_confidence": state.get("skill_confidence", 0),
        "skill_fallback_reason": state.get("skill_fallback_reason", ""),
        "skill_trace": state.get("skill_trace", [])[-20:],
        "routing_classifier": state.get("routing_classifier", "rules"),
        "workflow_blueprint": blueprint,
        "industry": state.get("industry", "unknown"),
        "target_freshness_year": state.get("target_freshness_year"),
        "awaiting_user": state.get("awaiting_user", False),
        "research_complete": state.get("research_complete", False),
        "scores": state.get("scores", {}),
        "dimension_status": state.get("dimension_status", {}),
        "conflicts": state.get("conflicts", []),
        "evidence_count": len(state.get("evidence", [])),
        "private_retrieval_count": state.get("private_retrieval_count", 0),
        "private_context": state.get("private_context", [])[-20:],
        "independent_source_count": independent_source_count(state.get("evidence", [])),
        "evidence": evidence,
        "scenario": state.get("scenario", "unknown"),
        "decision_goal": state.get("decision_goal", ""),
        "research_stage": state.get("research_stage", "understand"),
        "active_dimensions": active_dimensions,
        "report_confirmation_pending": state.get("report_confirmation_pending", False),
        "report_generated": state.get("report_generated", False),
        "citation_status": state.get("citation_status", "pending"),
        "citation_validation_error": state.get("citation_validation_error", ""),
        "report_claims": state.get("report_claims", []),
        "claim_judgments": state.get("claim_judgments", []),
        "claim_entailment_rate": state.get("claim_entailment_rate"),
        "claim_judge_status": state.get("claim_judge_status", "unavailable"),
        "user_conditions": state.get("user_conditions", []),
        "recommendation_traces": state.get("recommendation_traces", []),
        "recommendation_trace_status": state.get("recommendation_trace_status", "pending"),
        "run_metrics": state.get("last_run_metrics", {}),
        "trace_events": state.get("trace_events", [])[-50:],
        "llm_usage": state.get("llm_usage", {}),
        "history": _session_history(state),
        "pending_question": state.get("pending_question", ""),
        "current_interview_question": state.get("current_interview_question", {}),
        "interview_round": state.get("interview_round", 0),
        "interview_target_questions": state.get("interview_target_questions", 0),
        "interview_readiness_score": state.get("interview_readiness_score", 0),
    })


async def reset_session(request: Request):
    session_id = request.path_params.get("session_id")
    user_id = str(request.headers.get("x-user-id") or "anonymous")[:128]
    return JSONResponse({"session_id": create_session_id(user_id), "replaced": session_id})


async def health(request: Request):
    mysql_ok = None
    redis_ok = None
    if os.getenv("RUN_STORE_BACKEND", "sqlite").lower() == "mysql":
        mysql_ok = await asyncio.to_thread(getattr(RUN_STORE, "ping", lambda: False))
    if EXECUTION_BACKEND == "redis":
        redis_ok = await asyncio.to_thread(QUEUE.ping)
    return JSONResponse({
        "status": "ok",
        "service": "industry-web-research-agent",
        "version": "2.1",
        "research_provider": RESEARCH_PROVIDER.active_provider,
        "provider_preference": "mcp" if RESEARCH_PROVIDER.prefer_mcp else "direct",
        "checkpoint_backend": "sqlite" if CHECKPOINTER_CONNECTION is not None else "memory_fallback",
        "checkpoint_warning": CHECKPOINT_WARNING,
        "cost_tracking": bool(os.getenv("LLM_INPUT_COST_PER_1K") and os.getenv("LLM_OUTPUT_COST_PER_1K")),
        "claim_judge_configured": bool(os.getenv("EVAL_JUDGE_API_KEY") and os.getenv("EVAL_JUDGE_MODEL")),
        "run_store": os.getenv("RUN_STORE_BACKEND", "sqlite").lower(),
        "execution_backend": EXECUTION_BACKEND,
        "redis_configured": QUEUE is not None,
        "mysql_reachable": mysql_ok,
        "redis_reachable": redis_ok,
        "stale_run_recovery": RUN_RECOVERY,
        "run_timeout_seconds": int(os.getenv("RUN_TIMEOUT_SECONDS", "300")),
    })


async def ready(request: Request):
    checkpoint_ready = CHECKPOINTER is not None
    mysql_ready = True if os.getenv("RUN_STORE_BACKEND", "sqlite").lower() != "mysql" else await asyncio.to_thread(getattr(RUN_STORE, "ping", lambda: False))
    redis_ready = True if EXECUTION_BACKEND != "redis" else await asyncio.to_thread(QUEUE.ping)
    status_code = 200 if checkpoint_ready and mysql_ready and redis_ready else 503
    return JSONResponse({
        "ready": checkpoint_ready,
        "checkpoint_backend": "sqlite" if CHECKPOINTER_CONNECTION is not None else "memory_fallback",
        "run_store_backend": os.getenv("RUN_STORE_BACKEND", "sqlite").lower(),
        "execution_backend": EXECUTION_BACKEND,
        "mysql_ready": mysql_ready,
        "redis_ready": redis_ready,
        "active_research_tasks": len(ACTIVE_SESSIONS),
        "max_concurrent_research": MAX_CONCURRENT_RESEARCH,
    }, status_code=status_code)


async def _resume_queued_run(run: dict) -> None:
    """Drain a recovered run without requiring a connected browser."""
    message = str(run.get("request_text") or "").strip()
    if not message:
        RUN_STORE.update(run["run_id"], status="failed", finished_at=time.time(),
                         error_code="missing_request_payload", error_message="cannot resume without request text")
        return
    try:
        async with RESEARCH_SEMAPHORE:
            async for _ in event_generator(run["session_id"], message, run.get("user_id") or "anonymous", run["run_id"]):
                pass
    finally:
        ACTIVE_SESSIONS.discard(run["session_id"])


async def startup_recover_runs() -> None:
    if EXECUTION_BACKEND == "redis":
        # Redis Worker owns queued-message recovery in production mode. The API
        # must never execute the same Run during startup.
        return
    for run in RUN_STORE.list_queued(limit=100):
        if not run.get("request_text") or run["session_id"] in ACTIVE_SESSIONS:
            continue
        ACTIVE_SESSIONS.add(run["session_id"])
        task = asyncio.create_task(_resume_queued_run(run), name=f"recover-{run['run_id'][:8]}")
        RECOVERY_TASKS.add(task)
        task.add_done_callback(RECOVERY_TASKS.discard)


app = Starlette(
    debug=False,
    on_startup=[startup_recover_runs],
    routes=[
        Route("/", index),
        Route("/health", health),
        Route("/ready", ready),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/sessions", list_sessions, methods=["GET"]),
        Route("/api/sessions/{session_id}", get_session, methods=["GET"]),
        Route("/api/sessions/{session_id}/messages", send_message, methods=["POST"]),
        Route("/api/sessions/{session_id}/cancel", cancel_session, methods=["POST"]),
        Route("/api/runs/{run_id}", get_run, methods=["GET"]),
        Route("/api/runs", list_runs, methods=["GET"]),
        Route("/api/runs/{run_id}/events", run_events, methods=["GET"]),
        Route("/api/private-documents", private_documents, methods=["GET", "POST"]),
        Route("/api/private-documents/upload", upload_private_document, methods=["POST"]),
        Route("/api/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        Route("/api/memory/profile", memory_profile, methods=["GET", "POST", "DELETE"]),
        Route("/api/sessions/{session_id}/reset", reset_session, methods=["POST"]),
        Route("/chat", send_message, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
