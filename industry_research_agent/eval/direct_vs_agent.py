"""Same-model direct answer vs full research Agent benchmark.

This benchmark is opt-in and consumes real model/search credits. The direct
baseline deliberately has no tools or hidden evidence so the comparison does
not claim capabilities it did not use.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage

from agents.web_research import get_llm
from eval.online_eval import _estimated_cost, evaluate_case
from llm_usage import extract_usage
from output_format import to_plain_text

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_direct(case: dict, runs: int, interview_profile: dict | None = None) -> dict:
    attempts = []
    for index in range(runs):
        started = time.perf_counter()
        prompt = (
            "请像普通通用AI一样直接回答用户，不调用工具，不假装已经联网，不虚构来源。"
            "如果缺少信息可以说明假设。控制在800字以内。\n"
            f"用户补充条件：{interview_profile or {}}\n用户问题：" + case["input"]
        )
        try:
            response = get_llm().invoke([HumanMessage(content=prompt)])
            usage = extract_usage(response)
            answer = to_plain_text(response.content)
            attempts.append({
                "success": bool(answer), "duration_seconds": round(time.perf_counter() - started, 2),
                "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"], "llm_calls": usage["llm_calls"],
                "estimated_cost": _estimated_cost(usage), "evidence_count": 0,
                "citation_recall": 0.0, "valid_url_rate": 0.0,
                "recommendation_trace_rate": 0.0,
                "answer_excerpt": answer[:500],
            })
        except Exception as exc:
            attempts.append({"success": False, "duration_seconds": round(time.perf_counter() - started, 2),
                             "error": f"{type(exc).__name__}: {exc}"})
    ok = [item for item in attempts if item.get("success")]
    metrics = ("duration_seconds", "input_tokens", "output_tokens", "total_tokens", "llm_calls",
               "evidence_count", "citation_recall", "valid_url_rate", "recommendation_trace_rate")
    return {"variant": "same_model_direct", "runs": attempts, "summary": {
        "task_success_rate": len(ok) / runs,
        **{key: round(statistics.mean(item[key] for item in ok), 3) if ok else None for key in metrics},
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="coffee_hangzhou")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("online_cases.json").read_text(encoding="utf-8"))
    case = next((item for item in cases if item["id"] == args.case), None)
    if not case:
        raise SystemExit(f"unknown case: {args.case}")
    profiles_path = Path(__file__).with_name("interview_profiles.json")
    profiles = json.loads(profiles_path.read_text(encoding="utf-8")) if profiles_path.exists() else {}
    interview_profile = profiles.get(args.case, {})
    output = {
        "case": case, "same_model": True,
        "interview_profile": interview_profile,
        "direct": run_direct(case, max(1, args.runs), interview_profile),
        "agent": evaluate_case(case, max(1, args.runs), interview_profile),
        "interpretation": "Compare quality, traceability, cost and latency; do not assume the Agent wins every metric.",
    }
    directory = Path(__file__).with_name("results")
    directory.mkdir(exist_ok=True)
    path = directory / f"{date.today().isoformat()}-{case['id']}-direct-vs-agent.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
