"""Business model analyst with explicit assumptions and evidence-backed benchmarks."""
from __future__ import annotations
import re

from langchain_core.messages import AIMessage, HumanMessage

from state import ResearchState
from tools.roi_calculator import calculate_roi

from .web_research import research_dimensions


REQUIRED_BUSINESS_TERMS = {
    "客单价": ("客单价", "单价"),
    "月销量": ("月销量", "每月", "每天", "日均"),
    "成本或毛利率": ("成本", "毛利", "费用", "租金"),
}


def _latest_user_text(state: ResearchState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _missing_terms(text: str) -> list[str]:
    return [name for name, terms in REQUIRED_BUSINESS_TERMS.items() if not any(term in text for term in terms)]


def business_analyst_node(state: ResearchState) -> dict:
    user_text = " ".join(
        str(message.content) for message in state.get("messages", []) if isinstance(message, HumanMessage)
    )
    missing = _missing_terms(user_text)
    stage = state.get("business_stage", "unknown")
    intent = state.get("user_intent", "entry_feasibility")
    use_scenarios = state.get("assumption_mode", False) or stage in ("pre_launch", "planning")

    # Public benchmarks can be researched, but user-specific ROI is never fabricated.
    result = research_dimensions(
        state,
        analyst_name="商业模式分析师",
        dimensions=("商业模式",),
        role_instruction=(
            "你是商业模式分析师，负责公开可查的毛利率、成本结构、客单价和盈利方式。"
            "若用户处于筹备期，使用公开行业基准构建保守、中性、乐观三种情景；"
            "明确标注这些是假设，不得当作用户真实经营数据。"
            "只有用户明确要求计算已运营项目的实际 ROI 时，才使用用户经营参数。"
        ),
        max_tool_calls=4,
    )

    if use_scenarios:
        scenario_message = (
            "你目前处于筹备阶段，我不会要求你提供尚不存在的真实客单价、销量或毛利率。"
            "我会基于联网获取的行业基准，按保守、中性、乐观三种情景估算商业可行性；"
            "报告会把公开事实与测算假设分开标注。"
        )
        assumptions = list(state.get("assumptions", []))
        assumptions.extend([
            {"name": "客单价", "source": "行业公开基准", "type": "scenario_assumption"},
            {"name": "订单量", "source": "地区与店型基准", "type": "scenario_assumption"},
            {"name": "毛利率", "source": "行业公开基准", "type": "scenario_assumption"},
        ])
        result["messages"] = list(result.get("messages", [])) + [AIMessage(content=scenario_message)]
        result["assumptions"] = assumptions
        return result

    if intent == "roi_calculation" and stage == "operating" and missing:
        question = "为了测算你正在运营项目的实际 ROI，请补充：" + "、".join(missing) + "。"
        opinions = dict(result.get("analyst_opinions", {}))
        opinions["商业模式分析师"] = opinions.get("商业模式分析师", "") + "\n\n" + question
        missing_markers = [f"business:{name}" for name in missing]
        result.update({
            "messages": list(result.get("messages", [])) + [AIMessage(content=question)],
            "analyst_opinions": opinions,
            "awaiting_user": True,
            "pending_question": question,
            "missing_information": missing_markers,
        })
        status = dict(result.get("dimension_status", {}))
        status["商业模式"] = "insufficient"
        result["dimension_status"] = status
        return result

    # Only calculate when two explicit numeric inputs are discoverable. The report
    # keeps the inputs alongside the formula so assumptions stay reviewable.
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", user_text)]
    if intent == "roi_calculation" and stage == "operating" and len(numbers) >= 2 and state.get("budget") not in (None, "", "未知"):
        investment_match = re.search(r"(\d+(?:\.\d+)?)\s*万", str(state.get("budget")))
        investment = float(investment_match.group(1)) if investment_match else numbers[0]
        annual_benefit = numbers[-1]
        roi = calculate_roi.invoke({"investment": investment, "annual_benefit": annual_benefit})
        explanation = (
            f"\n\nROI 试算（仅基于用户参数）：投入={investment}万元，"
            f"年收益={annual_benefit}万元；ROI=年收益/投入={roi['roi']}，"
            f"预计回本={roi['payback_years']}年。"
        )
        opinions = dict(result.get("analyst_opinions", {}))
        opinions["商业模式分析师"] = opinions.get("商业模式分析师", "") + explanation
        result["analyst_opinions"] = opinions
        result["messages"] = list(result.get("messages", [])) + [AIMessage(content=explanation.strip())]
    return result
