"""Offline Agent Harness.

This harness is deterministic and does not spend API credits. It validates the
decisions and contracts that must remain stable; online quality evaluation is a
separate opt-in command because web results and LLM responses are non-deterministic.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.business_analyst import _missing_terms
from agents.supervisor import identify_industry
from research_tools import evidence_is_sufficient, sanitize_web_text
from triage import triage_request


IMPLEMENTED_ASSERTIONS = {
    "industry", "required_dimensions", "evidence", "report", "tool_events",
    "expected_sufficient", "expected_missing_business", "injection_removed",
    "max_agent_steps", "expect_no_mock", "scenario", "answer_type", "needs_full_report",
}


def load_cases(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def citation_ids(report: str) -> set[str]:
    return set(re.findall(r"\[(E\d+)\]", report or ""))


def evaluate_case(case: dict) -> dict:
    unknown = set(case.get("expected", {})) - IMPLEMENTED_ASSERTIONS
    failures: List[str] = []
    if unknown:
        failures.append("unsupported assertions: " + ", ".join(sorted(unknown)))

    expected = case.get("expected", {})
    detected = identify_industry(case.get("input", ""))
    if "industry" in expected and detected != expected["industry"]:
        failures.append(f"industry expected={expected['industry']} actual={detected}")

    triage = triage_request(case.get("input", ""))
    for field in ("scenario", "answer_type", "needs_full_report"):
        if field in expected and triage[field] != expected[field]:
            failures.append(f"{field} expected={expected[field]} actual={triage[field]}")

    evidence = expected.get("evidence", [])
    evidence_ids = {item.get("id") for item in evidence}
    report = expected.get("report", "")
    citations = citation_ids(report)
    invalid_citations = citations - evidence_ids
    if invalid_citations:
        failures.append("invalid citations: " + ", ".join(sorted(invalid_citations)))

    valid_url_count = sum(
        1 for item in evidence
        if urlparse(item.get("source_url", "")).scheme in ("http", "https")
        and bool(urlparse(item.get("source_url", "")).netloc)
    )
    key_claim_count = len(re.findall(r"\d+(?:\.\d+)?(?:%|％|亿|万|元)", report))
    citation_coverage = 1.0 if key_claim_count == 0 else min(1.0, len(citations) / key_claim_count)

    if "expected_sufficient" in expected:
        actual = evidence_is_sufficient(evidence)
        if actual != expected["expected_sufficient"]:
            failures.append(f"evidence sufficiency expected={expected['expected_sufficient']} actual={actual}")

    if "expected_missing_business" in expected:
        actual_missing = _missing_terms(case.get("input", ""))
        if actual_missing != expected["expected_missing_business"]:
            failures.append(f"business missing expected={expected['expected_missing_business']} actual={actual_missing}")

    if expected.get("injection_removed"):
        sanitized = sanitize_web_text(case.get("web_text", ""))
        if "[untrusted instruction removed]" not in sanitized:
            failures.append("prompt injection was not neutralized")

    events = expected.get("tool_events", [])
    tool_success_rate = sum(bool(item.get("success")) for item in events) / len(events) if events else 1.0
    source_hosts = {urlparse(item.get("source_url", "")).netloc for item in evidence if item.get("source_url")}
    authoritative_rate = (
        sum(item.get("source_type") == "official" for item in evidence) / len(evidence) if evidence else 0.0
    )

    if expected.get("expect_no_mock"):
        combined = json.dumps(case, ensure_ascii=False).lower()
        if any(marker in combined for marker in ("random mock", "模拟市场数据", "query_market_data")):
            failures.append("mock data marker found")

    return {
        "id": case["id"],
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "citation_coverage": citation_coverage,
            "valid_url_rate": valid_url_count / len(evidence) if evidence else 0.0,
            "independent_sources": len(source_hosts),
            "authoritative_rate": authoritative_rate,
            "tool_success_rate": tool_success_rate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("eval_cases.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [evaluate_case(case) for case in load_cases(args.cases)]
    passed = sum(item["passed"] for item in results)
    summary = {"total": len(results), "passed": passed, "failed": len(results) - passed, "results": results}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
