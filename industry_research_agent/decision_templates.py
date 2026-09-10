"""Structured decision templates: questions and dimensions, not local knowledge."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DecisionTemplate(BaseModel):
    id: str
    scenario: str
    purpose: str
    required_dimensions: list[str]
    required_topics: list[str]
    max_questions: int = Field(ge=0, le=10)
    completion_rule: Literal["all_topics_or_user_skip", "specific_answer"] = "all_topics_or_user_skip"


TEMPLATES = {
    "idea": DecisionTemplate(
        id="idea_feasibility_v1", scenario="idea", purpose="验证需求、资源匹配和最低成本进入路径",
        required_dimensions=["市场", "竞争", "商业模式", "风险", "机会"],
        required_topics=["decision_goal", "region", "budget", "business_form", "resources", "target_customer", "success"],
        max_questions=6,
    ),
    "planning": DecisionTemplate(
        id="planning_execution_v1", scenario="planning", purpose="比较落地方案、预算分配和启动顺序",
        required_dimensions=["竞争", "商业模式", "风险", "机会"],
        required_topics=["decision_goal", "customer", "options", "budget_allocation", "timeline", "success"],
        max_questions=6,
    ),
    "operating_diagnosis": DecisionTemplate(
        id="operating_diagnosis_v1", scenario="operating_diagnosis", purpose="定位经营问题并形成可验证假设",
        required_dimensions=["竞争", "商业模式", "风险"],
        required_topics=["problem", "timing", "metric", "actions", "constraints", "success"],
        max_questions=8,
    ),
    "existing_product_extension": DecisionTemplate(
        id="capability_extension_v1", scenario="existing_product_extension", purpose="评估现有能力扩展路径与实施风险",
        required_dimensions=["竞争", "商业模式", "风险", "机会"],
        required_topics=["current_capability", "bottleneck", "target", "production", "budget", "downtime", "team", "success"],
        max_questions=8,
    ),
    "company_research": DecisionTemplate(
        id="company_research_v1", scenario="company_research", purpose="限定企业研究范围和比较口径",
        required_dimensions=["市场", "竞争", "趋势", "机会", "风险"],
        required_topics=["scope", "focus", "period", "comparison"],
        max_questions=6,
    ),
    "option_comparison": DecisionTemplate(
        id="option_comparison_v1", scenario="option_comparison", purpose="使用统一权重比较可执行选项",
        required_dimensions=["市场", "竞争", "商业模式", "风险"],
        required_topics=["options", "priority", "constraint", "horizon"],
        max_questions=6,
    ),
    "market_question": DecisionTemplate(
        id="specific_answer_v1", scenario="market_question", purpose="直接回答范围明确的单点问题",
        required_dimensions=[], required_topics=[], max_questions=0, completion_rule="specific_answer",
    ),
}


def get_decision_template(scenario: str) -> DecisionTemplate:
    return TEMPLATES.get(scenario, TEMPLATES["market_question"])


def template_dimensions(scenario: str) -> list[str]:
    return list(get_decision_template(scenario).required_dimensions)
