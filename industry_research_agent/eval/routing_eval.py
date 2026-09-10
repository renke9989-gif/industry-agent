"""Compare deterministic routing with the optional LLM second stage."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from routing import RuleIntentClassifier, classify_request


CASES = [
    ("coffee market", "杭州咖啡店市场规模和趋势怎么样", "market_research"),
    ("company", "帮我研究华为的芯片供应链", "company_research"),
    ("factory", "工厂现有CNC设备，想增加机器人焊接产线", "capability_extension"),
    ("diagnosis", "我的店已经运营但最近亏损，帮我诊断", "business_diagnosis"),
    ("idea", "我想开一家宠物店", "market_research"),
    ("vague", "帮我看看这个", None),
]


def evaluate(*, mode: str) -> dict:
    classifier = RuleIntentClassifier() if mode == "rules" else None
    rows = []
    for case_id, text, expected in CASES:
        started = time.perf_counter()
        decision = classify_request(text, classifier=classifier)
        rows.append({
            "id": case_id,
            "expected_skill": expected,
            "selected_skill": decision.selected_skill,
            "confidence": decision.confidence,
            "classifier": decision.classifier,
            "needs_clarification": decision.selected_skill is None,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        })
    correct = [row for row in rows if row["selected_skill"] == next(item[2] for item in CASES if item[0] == row["id"])]
    return {
        "mode": mode,
        "accuracy": round(len(correct) / len(rows), 3),
        "clarification_rate": round(sum(row["needs_clarification"] for row in rows) / len(rows), 3),
        "average_latency_ms": round(statistics.mean(row["duration_ms"] for row in rows), 3),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rules", "rules_llm", "both"], default="both")
    args = parser.parse_args()
    modes = ["rules", "rules_llm"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        if mode == "rules_llm" and os.getenv("ROUTING_LLM_ENABLED", "false").lower() not in {"1", "true", "yes"}:
            results.append({"mode": mode, "status": "unavailable", "reason": "set ROUTING_LLM_ENABLED=true"})
        else:
            results.append(evaluate(mode="rules" if mode == "rules" else "rules_llm"))
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
