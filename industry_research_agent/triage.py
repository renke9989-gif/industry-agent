"""Deterministic request triage used before expensive research starts."""
from __future__ import annotations

import re
from typing import Any, Dict
from decision_templates import get_decision_template, template_dimensions


START_NOW_TERMS = (
    "直接开始", "直接开始调研", "先开始", "按保守假设", "按默认假设",
    "不用再问", "跳过", "跳过访谈", "不进入访谈",
)

CITY_NAMES = (
    "北京", "上海", "天津", "重庆", "杭州", "深圳", "广州", "成都",
    "南京", "武汉", "西安", "苏州", "长沙", "郑州", "青岛", "厦门",
)


def detect_region(text: str) -> str | None:
    """Extract a region without collapsing combined questionnaire choices."""
    for city in CITY_NAMES:
        if city in text:
            return city
    if "新一线/二线城市" in text or "新一线或二线城市" in text:
        return "新一线/二线城市"
    if "新一线" in text and "二线" in text:
        return "新一线/二线城市"
    if "新一线" in text:
        return "新一线城市"
    match = re.search(r"([一二三四五六]线城市)", text)
    return match.group(1) if match else None


def user_requested_start(text: str) -> bool:
    return any(term in text for term in START_NOW_TERMS)


def detect_scenario(text: str) -> str:
    lowered = text.lower()
    if any(term in text for term in ("扩产", "改造", "增加产线", "增加功能", "新增设备", "现有设备", "工厂", "产能", "兼容")):
        return "existing_product_extension"
    if any(term in text for term in ("已经开", "正在运营", "销量下降", "成本上升", "亏损", "复购", "客户流失", "经营诊断")):
        return "operating_diagnosis"
    if any(term in text for term in ("对比", "比较", "哪个更好", "哪一个更好", "二选一")):
        return "option_comparison"
    if any(term in text for term in ("筹备", "选址", "准备开", "计划开", "准备做", "预算分配")):
        return "planning"
    # Creation intent must win over the generic word "公司". Otherwise
    # "想开一家初创公司" is incorrectly treated as research about an
    # existing company and skips the questionnaire entirely.
    if any(term in text for term in (
        "想开", "想做", "构思", "值不值得", "可不可行", "能不能做", "还没开始", "尚未开始",
        "初创公司", "创业公司", "成立公司", "创办公司", "注册公司", "开始创业",
    )) or bool(re.search(r"(?:想|准备|打算).{0,12}(?:开|成立|创办|注册).{0,8}(?:店|公司|企业|工厂|工作室)", text)):
        return "idea"
    known_company = any(name in lowered for name in ("华为", "小米", "苹果", "tesla", "英伟达", "nvidia"))
    explicit_company_research = any(term in text for term in ("业务线", "财报", "供应链", "竞争对手")) or bool(
        re.search(r"(?:研究|调研|分析|了解|查询|看看).{0,12}(?:公司|企业)", text)
        or re.search(r"(?:公司|企业).{0,10}(?:目前|财报|业务线|产品|设备|技术|竞争)", text)
    )
    if known_company or explicit_company_research:
        return "company_research"
    return "market_question"


def detect_decision_goal(text: str, scenario: str) -> str:
    if "报告" in text:
        return "生成正式研究报告"
    if scenario == "operating_diagnosis":
        return "诊断当前经营问题并提出验证方案"
    if scenario == "existing_product_extension":
        return "评估现有能力扩展方案"
    if scenario == "option_comparison":
        return "比较候选方案并辅助选择"
    if scenario in ("idea", "planning"):
        return "评估进入或落地可行性"
    if scenario == "company_research":
        return "回答企业、产品或竞争相关问题"
    return text.strip()[:120] or "回答当前问题"


def _explicit_research_request(text: str) -> bool:
    if "调研" in text or "研究" in text:
        return True
    return any(term in text for term in (
        "深入调研", "完整调研", "行业调研", "调研一下", "调研下", "研究一下", "研究下", "研究报告", "生成报告", "可行性分析",
        "全面分析", "方案论证", "商业计划", "帮我调研", "系统分析",
    ))


def _looks_like_specific_question(text: str) -> bool:
    return bool(re.search(r"[？?]$", text.strip())) or any(term in text for term in (
        "是什么", "有哪些", "多少", "为什么", "怎么样", "怎么做", "是否", "能否",
        "推荐", "目前", "最新", "政策", "规模", "趋势",
    ))


def triage_request(text: str) -> Dict[str, Any]:
    scenario = detect_scenario(text)
    explicit_research = _explicit_research_request(text)
    specific_question = _looks_like_specific_question(text)
    stripped = text.strip()
    needs_clarification = scenario == "market_question" and not explicit_research and not specific_question and (
        len(stripped) < 10 or any(term in stripped for term in ("这个", "那个", "帮我看看", "怎么样", "想了解"))
    )
    needs_full_report = explicit_research or scenario in (
        "idea", "planning", "operating_diagnosis", "existing_product_extension", "option_comparison", "company_research",
    )
    # A narrowly scoped company or market question should be answered directly,
    # even when it happens to contain words such as "分析".
    if specific_question and scenario in ("market_question", "company_research") and not explicit_research:
        needs_full_report = False

    # An explicit request to research a market/industry should begin with the
    # interview-consent card; only narrowly phrased factual questions bypass it.
    # Every new user request gets an explicit depth choice.  The user can still
    # choose “直接开始调研” to preserve the narrow direct-answer path.
    needs_interview = not user_requested_start(text)
    if needs_interview:
        needs_full_report = True
    answer_type = "research_plan" if needs_full_report else "direct_answer"
    complexity = "high" if scenario in ("existing_product_extension", "operating_diagnosis") else (
        "medium" if needs_full_report else "low"
    )
    hard_limit = 8 if complexity == "high" else (6 if complexity == "medium" else 0)
    if needs_full_report:
        routing_reason = "需要结合你的目标、资源和约束，先进行针对性访谈"
    elif specific_question:
        routing_reason = "问题范围已经明确，先直接联网回答，不启动完整报告流程"
    else:
        routing_reason = "当前请求适合先做简要回答；需要完整研究时会继续访谈"
    routing_confidence = 0.95 if scenario != "market_question" else (0.85 if specific_question else 0.55)
    if needs_clarification:
        routing_reason = "当前请求缺少可识别的研究对象或具体问题，需要先澄清范围"
        routing_confidence = 0.35
    return {
        "scenario": scenario,
        "decision_goal": detect_decision_goal(text, scenario),
        "answer_type": answer_type,
        "complexity": complexity,
        "needs_web": True,
        "needs_interview": needs_interview,
        "needs_full_report": needs_full_report,
        "hard_interview_limit": hard_limit,
        "user_requested_start": user_requested_start(text),
        "routing_reason": routing_reason,
        "decision_template_id": get_decision_template(scenario).id,
        "routing_confidence": routing_confidence,
        "needs_clarification": needs_clarification,
    }


def required_dimensions(scenario: str, needs_full_report: bool) -> list[str]:
    if not needs_full_report:
        return []
    return template_dimensions(scenario) or ["市场", "趋势", "机会"]
