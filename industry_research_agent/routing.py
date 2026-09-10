"""Layered rule/LLM routing for allow-listed Skills."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from skill_registry import SKILLS, get_skill
from triage import triage_request


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    intent: str
    selected_skill: str | None = None
    confidence: float = Field(ge=0, le=1)
    needs_interview: bool
    needs_full_report: bool
    reason: str
    candidates: list[dict] = Field(default_factory=list)
    classifier: str = "rules"
    fallback_reason: str = ""


class IntentClassifier(Protocol):
    def classify(self, text: str, *, state: dict | None = None) -> RoutingDecision: ...


def _skill_for_scenario(scenario: str, *, state: dict | None = None) -> str | None:
    if state and state.get("citation_status") == "insufficient":
        return "citation_review"
    if state and state.get("report_confirmation_pending"):
        return "followup_answer"
    mapping = {
        "company_research": "company_research",
        "operating_diagnosis": "business_diagnosis",
        "existing_product_extension": "capability_extension",
        "idea": "market_research",
        "planning": "market_research",
        "market_question": "market_research",
    }
    return mapping.get(scenario)


class RuleIntentClassifier:
    """Zero-dependency default classifier using the existing triage contract."""

    def classify(self, text: str, *, state: dict | None = None) -> RoutingDecision:
        triage = triage_request(text)
        scenario = triage["scenario"]
        chosen = _skill_for_scenario(scenario, state=state)
        if triage.get("needs_clarification") and not (state and (state.get("citation_status") == "insufficient" or state.get("report_confirmation_pending"))):
            confidence = float(triage.get("routing_confidence", 0.35))
            chosen = None
        else:
            confidence = float(triage.get("routing_confidence", 0.5))
        candidates = []
        for skill in SKILLS.values():
            if not skill.enabled or scenario not in skill.scenarios:
                continue
            score = min(1.0, 0.55 + skill.priority / 200)
            if chosen == skill.id:
                score = max(score, confidence)
            candidates.append({"skill_id": skill.id, "score": round(score, 3), "reason": f"scenario={scenario}"})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return RoutingDecision(
            scenario=scenario,
            intent=triage.get("decision_goal", ""),
            selected_skill=chosen,
            confidence=max(0.0, min(1.0, confidence)),
            needs_interview=bool(triage.get("needs_interview")),
            needs_full_report=bool(triage.get("needs_full_report")),
            reason=triage.get("routing_reason", "规则路由"),
            candidates=candidates,
        )


class LLMIntentClassifier:
    """Optional second-stage classifier; output is still validated by Registry."""

    def __init__(self, llm):
        self.llm = llm

    def classify(self, text: str, *, state: dict | None = None) -> RoutingDecision:
        base = RuleIntentClassifier().classify(text, state=state)
        prompt = (
            "从以下 allow-list 中选择一个 Skill，只返回 JSON："
            f"{list(SKILLS)}。不要生成代码。用户输入：{text}。"
            f"当前规则场景：{base.scenario}。"
            "字段：selected_skill、confidence、reason。"
        )
        try:
            response = self.llm.invoke(prompt)
            import json
            parsed = json.loads(str(getattr(response, "content", response)).strip().strip("`"))
            candidate = str(parsed.get("selected_skill", ""))
            if not get_skill(candidate):
                raise ValueError("unregistered skill")
            base.selected_skill = candidate
            base.confidence = max(0.0, min(1.0, float(parsed.get("confidence", base.confidence))))
            base.reason = str(parsed.get("reason", "LLM 二次路由"))
            base.classifier = "rules+llm"
            return base
        except Exception as exc:
            base.fallback_reason = f"LLM 路由失败，回退规则：{type(exc).__name__}"
            return base


class FakeIntentClassifier:
    def __init__(self, decision: RoutingDecision):
        self.decision = decision

    def classify(self, text: str, *, state: dict | None = None) -> RoutingDecision:
        return self.decision.model_copy(deep=True)


def classify_request(text: str, *, state: dict | None = None, classifier: IntentClassifier | None = None) -> RoutingDecision:
    if classifier is not None:
        return classifier.classify(text, state=state)
    rule = RuleIntentClassifier().classify(text, state=state)
    # The optional LLM stage is only used for genuinely ambiguous requests;
    # deterministic/high-confidence routes remain zero-cost.
    import os
    if os.getenv("ROUTING_LLM_ENABLED", "false").lower() in {"1", "true", "yes"} and .5 <= rule.confidence < .8:
        try:
            from agents.web_research import get_llm
            return LLMIntentClassifier(get_llm()).classify(text, state=state)
        except Exception as exc:
            rule.fallback_reason = f"LLM 路由不可用，回退规则：{type(exc).__name__}"
    return rule
