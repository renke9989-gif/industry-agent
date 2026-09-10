"""Deterministic Skill selection and authorization harness."""
from __future__ import annotations

import json

from routing import LLMIntentClassifier, RuleIntentClassifier
from skill_registry import validate_skill


class _Response:
    def __init__(self, content): self.content = content


class _LLM:
    def __init__(self, content): self.content = content; self.calls = 0
    def invoke(self, _prompt): self.calls += 1; return _Response(self.content)


def run_cases() -> list[dict]:
    rule = RuleIntentClassifier()
    cases = []

    def check(name, condition, detail=""):
        cases.append({"id": name, "passed": bool(condition), "detail": detail})

    market = rule.classify("2026年咖啡市场规模是多少？")
    check("specific_market_direct", market.selected_skill == "market_research" and not market.needs_full_report)
    vague = rule.classify("帮我看看这个")
    check("vague_requires_clarification", vague.selected_skill is None and vague.confidence < .5)
    check("idea_selects_market_research", rule.classify("我想开一家宠物店").selected_skill == "market_research")
    check("company_selects_company_research", rule.classify("帮我研究华为的供应链").selected_skill == "company_research")
    check("extension_selects_capability", rule.classify("工厂现有CNC设备，准备增加机器人焊接产线").selected_skill == "capability_extension")
    check("diagnosis_selects_business", rule.classify("我的店已经运营，但最近亏损需要经营诊断").selected_skill == "business_diagnosis")
    followup = rule.classify("那上海呢？", state={"report_confirmation_pending": True})
    check("report_followup_selects_followup", followup.selected_skill == "followup_answer")
    review = rule.classify("检查报告引用", state={"citation_status": "insufficient"})
    check("citation_failure_selects_review", review.selected_skill == "citation_review")

    llm = _LLM(json.dumps({"selected_skill": "company_research", "confidence": .82, "reason": "企业对象明确"}, ensure_ascii=False))
    selected = LLMIntentClassifier(llm).classify("需要分析一个公司")
    check("llm_second_stage_allowlist", selected.selected_skill == "company_research" and llm.calls == 1 and selected.classifier == "rules+llm")
    check("low_confidence_does_not_search", vague.needs_full_report is False and vague.selected_skill is None)
    try:
        validate_skill("shell_executor")
        unknown_rejected = False
    except ValueError:
        unknown_rejected = True
    check("unknown_skill_rejected", unknown_rejected)
    try:
        validate_skill("company_research", node="business_analyst")
        unauthorized_rejected = False
    except ValueError:
        unauthorized_rejected = True
    check("unauthorized_node_rejected", unauthorized_rejected)
    return cases


def main() -> int:
    results = run_cases()
    output = {"total": len(results), "passed": sum(item["passed"] for item in results), "results": results}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] == output["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
