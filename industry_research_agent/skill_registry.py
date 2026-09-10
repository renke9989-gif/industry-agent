"""Allow-listed Skill registry for deterministic Agent routing.

Skills describe executable capabilities; they are not arbitrary prompts or
model-generated code.  The registry is deliberately small so routing remains
auditable and easy to evaluate.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str = "1.0"
    name: str
    description: str
    scenarios: list[str]
    tags: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    allowed_nodes: list[str] = Field(default_factory=list)
    priority: int = Field(default=50, ge=1, le=100)
    enabled: bool = True


SKILLS: dict[str, SkillSpec] = {
    "market_research": SkillSpec(
        id="market_research", name="市场研究", version="1.0",
        description="联网检索市场规模、趋势、机会和政策证据",
        scenarios=["idea", "planning", "market_question"],
        tags=["市场", "趋势", "机会", "行业", "规模", "政策"],
        required_inputs=["industry"], allowed_nodes=["market_analyst", "competition_analyst", "business_analyst", "direct_answer", "report_generator"], priority=80,
    ),
    "company_research": SkillSpec(
        id="company_research", name="企业研究", version="1.0",
        description="围绕指定公司、产品、设备或业务线回答问题",
        scenarios=["company_research"], tags=["公司", "企业", "产品", "设备", "供应链"],
        required_inputs=["industry"], allowed_nodes=["market_analyst", "competition_analyst", "report_generator", "direct_answer"], priority=90,
    ),
    "business_diagnosis": SkillSpec(
        id="business_diagnosis", name="经营诊断", version="1.0",
        description="分析已运营业务的经营问题并提出验证动作",
        scenarios=["operating_diagnosis"], tags=["亏损", "复购", "成本", "经营", "诊断"],
        required_inputs=["industry"], allowed_nodes=["business_analyst", "competition_analyst", "report_generator"], priority=90,
    ),
    "capability_extension": SkillSpec(
        id="capability_extension", name="能力扩展评估", version="1.0",
        description="评估现有工厂、设备、产线或产品的扩展方案",
        scenarios=["existing_product_extension"], tags=["工厂", "设备", "产线", "改造", "扩展"],
        required_inputs=["industry"], allowed_nodes=["business_analyst", "competition_analyst", "report_generator"], priority=90,
    ),
    "followup_answer": SkillSpec(
        id="followup_answer", name="报告追问", version="1.0",
        description="基于当前会话证据回答报告完成后的具体追问",
        scenarios=["followup"], tags=["追问", "这类", "刚才", "薪资"],
        required_inputs=["session"], allowed_nodes=["followup"], priority=100,
    ),
    "citation_review": SkillSpec(
        id="citation_review", name="引用复核", version="1.0",
        description="复核 Claim 与 Evidence 的绑定、来源和支持关系",
        scenarios=["citation_review"], tags=["引用", "证据", "Claim", "校验"],
        required_inputs=["evidence"], allowed_nodes=["report_generator"], priority=100,
    ),
}


def get_skill(skill_id: str) -> SkillSpec | None:
    skill = SKILLS.get(skill_id)
    return skill if skill and skill.enabled else None


def list_skills() -> list[SkillSpec]:
    return [skill for skill in SKILLS.values() if skill.enabled]


def validate_skill(skill_id: str, *, node: str | None = None) -> SkillSpec:
    skill = get_skill(skill_id)
    if skill is None:
        raise ValueError(f"unknown or disabled skill: {skill_id}")
    if node and node not in skill.allowed_nodes:
        raise ValueError(f"skill {skill_id} cannot call node {node}")
    return skill
