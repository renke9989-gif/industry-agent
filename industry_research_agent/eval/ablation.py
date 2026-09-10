"""Compare single-agent and multi-agent topologies under equal inputs.

This is opt-in and online because both variants use the configured LLM/search
provider. It reports trade-offs; it does not assume multi-agent is superior.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.single_agent import single_research_node
from graph import compile_graph
from report_generator import report_node
from state import ResearchState, initial_state
from eval.online_eval import _citation_coverage


def build_single_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("single_agent", single_research_node)
    workflow.add_node("report_generator", report_node)
    workflow.set_entry_point("single_agent")
    workflow.add_edge("single_agent", "report_generator")
    workflow.add_edge("report_generator", END)
    return workflow.compile(checkpointer=MemorySaver())


def run_variant(name: str, user_input: str, runs: int, interview_profile: dict | None = None) -> dict:
    is_single = name.startswith("single_agent")
    evidence_enabled = name.endswith("evidence")
    interview_enabled = "interview" in name
    attempts = []
    for index in range(runs):
        graph = build_single_graph() if is_single else compile_graph()
        config = {"configurable": {"thread_id": f"ablation-{name}-{index}-{time.time_ns()}"}, "recursion_limit": 30}
        started = time.perf_counter()
        try:
            graph_input = initial_state(HumanMessage(content=user_input))
            graph_input["evidence_enabled"] = evidence_enabled
            graph_input["auto_report"] = True
            graph_input["known_facts"] = dict(interview_profile or {}) if interview_enabled else {}
            graph_input["interview_ablation_enabled"] = interview_enabled
            if not is_single:
                graph_input["interview_complete"] = True
                graph_input["user_requested_start"] = True
            for _ in graph.stream(graph_input, config):
                pass
            state = dict(graph.get_state(config).values)
            evidence = state.get("evidence", [])
            events = state.get("tool_events", [])
            report = "\n".join(str(getattr(message, "content", "")) for message in state.get("messages", []))
            usage = state.get("llm_usage", {})
            attempts.append({
                "success": bool(state.get("research_complete")),
                "duration_seconds": round(time.perf_counter() - started, 2),
                "evidence_count": len(evidence),
                "dimension_completion_rate": sum(value == "complete" for value in state.get("dimension_status", {}).values()) / 6,
                "citation_count": sum(1 for message in state.get("messages", []) if "[E" in str(getattr(message, "content", ""))),
                "citation_coverage": _citation_coverage(report, evidence) if evidence_enabled else None,
                "evidence_enabled": evidence_enabled,
                "interview_enabled": interview_enabled,
                "interview_fact_count": len(graph_input.get("known_facts", {})),
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "llm_calls": int(usage.get("llm_calls", 0) or 0),
                "tool_success_rate": sum(bool(event.get("success")) for event in events) / len(events) if events else 0,
                "agent_steps": state.get("agent_step_count", 0),
                "recommendation_trace_rate": (
                    sum(item.get("trace_status") == "validated" for item in state.get("recommendation_traces", []))
                    / len(state.get("recommendation_traces", [])) if state.get("recommendation_traces") else 0
                ),
                "user_condition_reference_count": report.count("[U"),
            })
        except Exception as exc:
            attempts.append({"success": False, "duration_seconds": round(time.perf_counter() - started, 2), "error": str(exc)})
    successful = [item for item in attempts if item.get("success")]
    return {
        "variant": name,
        "runs": attempts,
        "summary": {
            "task_success_rate": len(successful) / runs,
            "average_latency": statistics.mean(item["duration_seconds"] for item in successful) if successful else None,
            "average_evidence_count": statistics.mean(item["evidence_count"] for item in successful) if successful else 0,
            "average_dimension_completion": statistics.mean(item["dimension_completion_rate"] for item in successful) if successful else 0,
            "average_tool_success_rate": statistics.mean(item["tool_success_rate"] for item in successful) if successful else 0,
            "average_agent_steps": statistics.mean(item["agent_steps"] for item in successful) if successful else 0,
            "average_citation_coverage": statistics.mean(item["citation_coverage"] for item in successful if item.get("citation_coverage") is not None) if successful and any(item.get("citation_coverage") is not None for item in successful) else None,
            "average_input_tokens": statistics.mean(item["input_tokens"] for item in successful) if successful else 0,
            "average_output_tokens": statistics.mean(item["output_tokens"] for item in successful) if successful else 0,
            "average_total_tokens": statistics.mean(item["total_tokens"] for item in successful) if successful else 0,
            "average_llm_calls": statistics.mean(item["llm_calls"] for item in successful) if successful else 0,
            "average_recommendation_trace_rate": statistics.mean(item["recommendation_trace_rate"] for item in successful) if successful else 0,
            "average_user_condition_references": statistics.mean(item["user_condition_reference_count"] for item in successful) if successful else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="想在杭州开咖啡店，预算50万")
    parser.add_argument("--case", default="coffee_hangzhou", help="profile id used for the interview ablation")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--variant", action="append", choices=[
        "single_agent", "single_agent_evidence", "multi_agent", "multi_agent_evidence",
        "multi_agent_interview", "multi_agent_interview_evidence",
    ], help="run only selected variant; repeat this flag to compare a subset")
    args = parser.parse_args()
    profiles_path = Path(__file__).with_name("interview_profiles.json")
    profiles = json.loads(profiles_path.read_text(encoding="utf-8")) if profiles_path.exists() else {}
    interview_profile = profiles.get(args.case, {})
    variants = args.variant or [
        "single_agent", "single_agent_evidence", "multi_agent", "multi_agent_evidence",
        "multi_agent_interview", "multi_agent_interview_evidence",
    ]
    output = {
        "input": args.input,
        "runs": args.runs,
        "interview_profile_id": args.case,
        "interview_profile": interview_profile,
        "variants": [run_variant(name, args.input, args.runs, interview_profile) for name in variants],
    }
    directory = Path(__file__).with_name("results")
    directory.mkdir(exist_ok=True)
    path = directory / f"{date.today().isoformat()}-ablation.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
