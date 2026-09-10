"""Opt-in online quality evaluation. It consumes real LLM/search credits."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage
from report_generator import _EXTERNAL_NUMBER
from routing import classify_request
from triage import triage_request

ROOT = Path(__file__).resolve().parents[1]


def _estimated_cost(usage: dict) -> float | None:
    input_rate = os.getenv("LLM_INPUT_COST_PER_1K")
    output_rate = os.getenv("LLM_OUTPUT_COST_PER_1K")
    if not input_rate or not output_rate:
        return None
    return round(
        int(usage.get("input_tokens", 0) or 0) / 1000 * float(input_rate)
        + int(usage.get("output_tokens", 0) or 0) / 1000 * float(output_rate), 6,
    )


def _citation_coverage(report: str, evidence: list[dict]) -> float:
    valid_ids = {item.get("id") for item in evidence}
    # Count externally meaningful numeric claims, not section numbering,
    # dates in the report header, or the user's budget/assumption fields.
    context_labels = ("预算", "报告生成日期", "生成日期", "报告日期", "日期", "情景", "步骤", "假设", "投入=", "用户输入", "未来", "行动", "验证标准", "验收标准", "用户条件")
    claim_lines = [line.strip() for line in report.splitlines()
                   if line.strip() and _EXTERNAL_NUMBER.search(line)
                   and not any(label in line for label in context_labels)
                   and not re.search(r"(?:第一|第二|第三|第四|第五)[、.：:]", line)]
    if not claim_lines:
        return 1.0
    supported = 0
    for line in claim_lines:
        cited = set(re.findall(r"\[(E\d+)\]", line))
        supported += bool(cited and cited.issubset(valid_ids))
    return supported / len(claim_lines)


def _dimension_completion(case: dict, status: dict) -> float:
    required = case.get("required_dimensions", [])
    if not required:
        return 1.0
    return sum(status.get(name) in ("complete", "insufficient") for name in required) / len(required)


def _acceptance_failures(case: dict, result: dict) -> list[str]:
    failures = []
    if not result.get("success"):
        failures.append("task_failed")
    if result.get("valid_url_rate", 0) < 1.0:
        failures.append("invalid_citation_url")
    if result.get("citation_recall", 0) < 0.9:
        failures.append("citation_recall_below_0.90")
    if result.get("tool_success_rate", 0) < 0.5:
        failures.append("tool_success_rate_below_0.50")
    if result.get("citation_status") == "insufficient":
        failures.append("report_citation_validation_failed")
    if result.get("dimension_completion_rate", 0) < 1.0:
        failures.append("required_dimension_incomplete")
    if result.get("duration_seconds", 0) > case.get("max_latency_seconds", 90):
        failures.append("latency_budget_exceeded")
    cost = result.get("estimated_cost")
    if cost is not None and cost > case.get("max_cost_usd", 0.2):
        failures.append("cost_budget_exceeded")
    return failures


def evaluate_case(case: dict, runs: int = 1, interview_profile: dict | None = None) -> dict:
    from graph import compile_graph
    from state import initial_state

    attempts = []
    expected_skill = classify_request(case["input"]).selected_skill
    for run in range(runs):
        print(f"[{case['id']}] run {run + 1}/{runs}: 初始化 Agent...", flush=True)
        started = time.perf_counter()
        graph = compile_graph()
        config = {"configurable": {"thread_id": f"online-{case['id']}-{run}-{int(time.time())}"}, "recursion_limit": 30}
        final = {}
        graph_input = initial_state(HumanMessage(content=case["input"]))
        graph_input["auto_report"] = True
        graph_input["interview_complete"] = True
        graph_input["user_requested_start"] = True
        # Evaluation must exercise the same full research path for every case;
        # otherwise narrowly phrased cases are intentionally routed to direct
        # answer and cannot measure the required dimensions.
        graph_input["needs_full_report"] = True
        graph_input["scenario"] = "idea"
        graph_input["industry"] = case.get("industry", "unknown")
        graph_input["evaluation_required_dimensions"] = list(case.get("required_dimensions", []))
        graph_input["target_freshness_year"] = int(case.get("freshness_year", date.today().year))
        graph_input["known_facts"] = dict(interview_profile or {})
        try:
            print(f"[{case['id']}] run {run + 1}/{runs}: 开始联网研究（MCP 可能需要几十秒初始化）...", flush=True)
            for event in graph.stream(graph_input, config):
                node_names = ",".join(event.keys())
                print(f"[{case['id']}] run {run + 1}/{runs}: 完成节点 {node_names}", flush=True)
                for output in event.values():
                    final.update(output)
            snapshot = graph.get_state(config).values
            final = dict(snapshot or final)
            report = "\n".join(str(getattr(m, "content", "")) for m in final.get("messages", []))
            evidence = final.get("evidence", [])
            urls = [item.get("source_url", "") for item in evidence]
            numeric_lines = [line for line in report.splitlines() if re.search(r"\d", line)]
            citations = set(re.findall(r"\[(E\d+)\]", report))
            years = sum(str(case["freshness_year"]) in str(item.get("period", "")) for item in evidence)
            usage = final.get("llm_usage", {})
            attempt = {
                "success": bool(final.get("research_complete")),
                "duration_seconds": round(time.perf_counter() - started, 2),
                "evidence_count": len(evidence),
                "valid_url_rate": sum(url.startswith(("http://", "https://")) for url in urls) / len(urls) if urls else 0,
                "citation_recall": _citation_coverage(report, evidence),
                "freshness_rate": years / len(evidence) if evidence else 0,
                "official_source_rate": sum(item.get("source_type") == "official" for item in evidence) / len(evidence) if evidence else 0,
                "tool_success_rate": sum(bool(item.get("success")) for item in final.get("tool_events", [])) / len(final.get("tool_events", [])) if final.get("tool_events") else 0,
                "tool_failure_count": sum(not bool(item.get("success")) for item in final.get("tool_events", [])),
                "search_call_count": sum(item.get("tool") == "search" for item in final.get("tool_events", [])),
                "fetch_page_count": sum(item.get("tool") == "fetch_page" for item in final.get("tool_events", [])),
                "mcp_failure_count": sum(item.get("tool_source") == "mcp" and not bool(item.get("success")) for item in final.get("tool_events", [])),
                "retry_count": sum(int(item.get("retry_count", 0) or 0) for item in final.get("tool_events", [])),
                "tool_error_codes": sorted({str(item.get("error_code")) for item in final.get("tool_events", []) if item.get("error_code")}),
                "agent_steps": final.get("agent_step_count", 0),
                "provider_sources": sorted({item.get("tool_source", "unknown") for item in final.get("tool_events", [])}),
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "llm_calls": int(usage.get("llm_calls", 0) or 0),
                "estimated_cost": _estimated_cost(usage),
                "dimension_completion_rate": _dimension_completion(case, final.get("dimension_status", {})),
                "conflict_count": len(final.get("conflicts", [])),
                "citation_status": final.get("citation_status", "pending"),
                "citation_validation_error": final.get("citation_validation_error", ""),
                "claim_entailment_rate": final.get("claim_entailment_rate"),
                "claim_judge_status": final.get("claim_judge_status", "unavailable"),
                "recommendation_trace_status": final.get("recommendation_trace_status", "pending"),
                "recommendation_trace_rate": (
                    sum(item.get("trace_status") == "validated" for item in final.get("recommendation_traces", []))
                    / len(final.get("recommendation_traces", []))
                    if final.get("recommendation_traces") else 0
                ),
                "selected_skill": final.get("selected_skill", ""),
                "initial_skill": final.get("initial_skill", ""),
                "expected_skill": expected_skill,
                "skill_selection_accuracy": 1.0 if final.get("initial_skill", final.get("selected_skill", "")) == expected_skill else 0.0,
                "routing_confidence": float(final.get("skill_confidence", 0) or 0),
                "clarification_rate": 1.0 if triage_request(case["input"]).get("needs_clarification") else 0.0,
                "context_before_tokens": int(final.get("context_metrics", {}).get("before_tokens", 0) or 0),
                "context_after_tokens": int(final.get("context_metrics", {}).get("after_tokens", 0) or 0),
                "context_compression_rate": 1.0 if final.get("context_metrics", {}).get("compression_count", 0) else 0.0,
                "context_overflow_rate": 1.0 if final.get("context_metrics", {}).get("overflow_count", 0) else 0.0,
                "context_compression_ratio": float(final.get("context_metrics", {}).get("compression_ratio", 1.0) or 1.0),
                "preserved_condition_count": int(final.get("context_metrics", {}).get("preserved_condition_count", 0) or 0),
                "preserved_evidence_count": int(final.get("context_metrics", {}).get("preserved_evidence_count", 0) or 0),
                "reprint_count": sum(bool(item.get("is_reprint")) for item in evidence),
                "independent_source_count": len({
                    str(item.get("source_url", "")).split('/')[2].lower()
                    for item in evidence
                    if item.get("source_url") and not item.get("is_reprint") and len(str(item.get("source_url", "")).split('/')) > 2
                }),
            }
            attempt["acceptance_failures"] = _acceptance_failures(case, attempt)
            attempt["accepted"] = not attempt["acceptance_failures"]
            attempts.append(attempt)
            print(f"[{case['id']}] run {run + 1}/{runs}: 完成，耗时 {attempts[-1]['duration_seconds']}s，证据 {len(evidence)} 条", flush=True)
        except Exception as exc:
            attempts.append({"success": False, "duration_seconds": round(time.perf_counter() - started, 2), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{case['id']}] run {run + 1}/{runs}: 失败：{attempts[-1]['error']}", flush=True)
    successful = [item for item in attempts if item.get("success")]
    numeric_metrics = ["valid_url_rate", "citation_recall", "freshness_rate", "official_source_rate", "tool_success_rate", "duration_seconds", "dimension_completion_rate", "independent_source_count", "reprint_count", "input_tokens", "output_tokens", "total_tokens", "llm_calls", "search_call_count", "fetch_page_count", "mcp_failure_count", "retry_count", "recommendation_trace_rate", "skill_selection_accuracy", "routing_confidence", "clarification_rate", "context_before_tokens", "context_after_tokens", "context_compression_rate", "context_overflow_rate", "context_compression_ratio", "preserved_condition_count", "preserved_evidence_count"]
    durations = [item["duration_seconds"] for item in successful]
    summary = {key: round(statistics.mean(item[key] for item in successful), 3) for key in numeric_metrics if successful and key in successful[0]}
    summary["p95_latency"] = round(sorted(durations)[max(0, int(len(durations) * 0.95) - 1)], 3) if durations else None
    costs = [item["estimated_cost"] for item in successful if item.get("estimated_cost") is not None]
    claim_rates = [item["claim_entailment_rate"] for item in successful if item.get("claim_entailment_rate") is not None]
    judge_statuses = sorted({item.get("claim_judge_status", "unavailable") for item in attempts})
    summary.update({"estimated_cost": round(statistics.mean(costs), 6) if costs else None,
                    "claim_entailment_rate": round(statistics.mean(claim_rates), 3) if claim_rates else None,
                    "claim_judge_status": judge_statuses[0] if len(judge_statuses) == 1 else judge_statuses,
                    "claim_entailment_metric": "available" if claim_rates else "unavailable",
                    "repeatability": len(successful) / runs,
                    "acceptance_rate": sum(item.get("accepted", False) for item in attempts) / runs})
    return {"case": case, "runs": attempts, "summary": {"task_success_rate": len(successful) / runs, **summary}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="case id; default runs all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--provider", choices=["mcp", "direct"], default=None,
                        help="研究 Provider；direct 用于绕过 MCP 排查 LLM/搜索链路")
    args = parser.parse_args()
    if args.provider:
        os.environ["RESEARCH_PROVIDER"] = args.provider
    cases = json.loads((Path(__file__).with_name("online_cases.json")).read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
    results = [evaluate_case(case, max(1, args.runs)) for case in cases]
    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    stamp = date.today().isoformat()
    output_name = f"{stamp}-{args.case}.json" if args.case else f"{stamp}-summary.json"
    (output_dir / output_name).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
