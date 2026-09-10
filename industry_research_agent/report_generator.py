"""Generate a citation-grounded plain-text report tailored to the user scenario."""
from __future__ import annotations

import os
import re
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from context_manager import compact_context, merge_context_metrics
from langchain_openai import ChatOpenAI

from contracts import RecommendationTrace, ReportClaim, UserCondition
from output_format import to_plain_text
from state import ResearchState
from llm_usage import extract_usage, merge_usage
from claim_validation import validate_claims


load_dotenv()
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            openai_api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.1,
        )
    return _llm


def _source_list(evidence: list[dict]) -> str:
    if not evidence:
        return "本次未取得可引用的公开来源。"
    return "\n".join(
        f"[{item['id']}] {item.get('source_title') or item.get('source_url')}。"
        f"来源类型：{item.get('source_type', 'web')}；检索日期：{item.get('retrieved_at', '')[:10]}；"
        f"链接：{item.get('source_url')}"
        for item in evidence
    )


def _evidence_block(evidence: list[dict]) -> str:
    return "\n".join(
        f"[{item['id']}] 维度：{item.get('dimension')}；地区：{item.get('region')}；"
        f"时期：{item.get('period') or '未标明'}；摘录：{item.get('excerpt', '')[:700]}；"
        f"链接：{item.get('source_url')}"
        for item in evidence
    ) or "无可靠公开证据"


def build_user_conditions(state: ResearchState) -> list[UserCondition]:
    """Create stable IDs for user-provided decision constraints."""
    candidates: list[tuple[str, str, str]] = []
    excluded = {"interview_consent", "request_scope"}
    for key, value in dict(state.get("known_facts", {})).items():
        if key not in excluded and str(value).strip():
            candidates.append((str(key), str(value).strip(), "interview"))
    if state.get("region") not in (None, "", "全国", "未知"):
        candidates.append(("region", str(state["region"]), "request"))
    if state.get("budget") not in (None, "", "未知"):
        candidates.append(("budget", str(state["budget"]), "request"))
    if state.get("business_stage") not in (None, "", "unknown"):
        candidates.append(("business_stage", str(state["business_stage"]), "request"))
    if state.get("decision_goal"):
        candidates.append(("decision_goal", str(state["decision_goal"]), "request"))
    unique: list[tuple[str, str, str]] = []
    seen = set()
    for item in candidates:
        identity = (item[0], item[1])
        if identity not in seen:
            seen.add(identity)
            unique.append(item)
    return [UserCondition(id=f"U{index}", key=key, value=value, source=source)
            for index, (key, value, source) in enumerate(unique, 1)]


def _user_condition_block(items: list[UserCondition]) -> str:
    return "\n".join(f"[{item.id}] {item.key}：{item.value}" for item in items) or "没有已确认的用户条件"


def extract_recommendation_traces(
    report: str, evidence: list[dict], conditions: list[UserCondition],
) -> tuple[list[RecommendationTrace], list[str]]:
    """Parse action recommendations and validate both sides of the trace."""
    action_match = re.search(r"(?:^|\n)(?:行动建议|针对你的行动建议)\s*[：:]?\s*\n?([\s\S]*?)(?=\n(?:主要风险|未来\s*7\s*天|来源)\s*[：:]?|$)", report)
    if not action_match:
        return [], ["报告缺少可解析的行动建议章节"] if conditions else []
    block = action_match.group(1).strip()
    points = [item.strip() for item in re.split(r"(?=(?:第一|第二|第三|第四)[、，])", block) if item.strip()]
    valid_evidence = {str(item.get("id")) for item in evidence}
    valid_conditions = {item.id for item in conditions}
    traces: list[RecommendationTrace] = []
    warnings: list[str] = []
    for index, text in enumerate(points, 1):
        evidence_ids = [item for item in re.findall(r"\[(E\d+)\]", text) if item in valid_evidence]
        condition_ids = [item for item in re.findall(r"\[(U\d+)\]", text) if item in valid_conditions]
        missing = []
        if evidence and not evidence_ids:
            missing.append("Evidence")
        if conditions and not condition_ids:
            missing.append("用户条件")
        if missing:
            warnings.append(f"第{index}条行动建议缺少{'和'.join(missing)}绑定")
        metric_match = re.search(r"(?:验证标准|验收标准|目标)[：:]?([^。；]+)", text)
        traces.append(RecommendationTrace(
            id=f"R{index}", text=text, user_condition_ids=condition_ids,
            evidence_ids=evidence_ids,
            validation_metric=metric_match.group(1).strip() if metric_match else None,
            priority=index, trace_status="insufficient" if missing else "validated",
        ))
    return traces, warnings


_plain_text = to_plain_text

# Numbers with a unit are much more likely to be external claims than dates
# in a heading. They must carry an Evidence ID unless the line explicitly
# describes user input, an assumption, or an action plan.
_EXTERNAL_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|万亿元|亿元|万元|元|万人|家|次|个月|天)")


def _citation_warnings(report: str, evidence: list[dict]) -> list[str]:
    valid_ids = {item.get("id") for item in evidence}
    used_ids = set(re.findall(r"\[(E\d+)\]", report))
    warnings = []
    unknown = sorted(used_ids - valid_ids)
    if unknown:
        warnings.append("报告包含未知证据编号：" + "、".join(unknown))
    for line in report.splitlines():
        if _EXTERNAL_NUMBER.search(line) and not re.search(r"\[E\d+\]", line):
            if not _is_non_external_number_line(line):
                warnings.append("存在未标注来源的数字表述：" + line[:80])
                break
    return warnings


def validate_report_claims(report: str, evidence: list[dict]) -> list[ReportClaim]:
    valid_ids = {item.get("id") for item in evidence}
    claims = []
    for line in report.splitlines():
        ids = re.findall(r"\[(E\d+)\]", line)
        if ids:
            claims.append(ReportClaim(
                text=line.strip(), evidence_ids=ids,
                claim_type="calculation" if "ROI" in line else "fact",
            ))
    if any(item_id not in valid_ids for claim in claims for item_id in claim.evidence_ids):
        raise ValueError("report contains an unknown Evidence ID")
    for line in report.splitlines():
        if _EXTERNAL_NUMBER.search(line) and not re.search(r"\[E\d+\]", line):
            if not _is_non_external_number_line(line):
                raise ValueError("report contains an external numeric claim without an Evidence ID")
    return claims


def _is_non_external_number_line(line: str) -> bool:
    """Return true for numbers supplied by the user or describing the plan.

    This keeps the citation gate strict for market facts while avoiding false
    positives such as “未来 7 天”, recommendation ordering and user budget.
    """
    labels = ("预算", "日期", "生成", "情景", "步骤", "假设", "投入=", "用户", "未来", "行动", "验证标准", "验收标准")
    if any(label in line for label in labels):
        return True
    if re.search(r"(?:第一|第二|第三|第四|第五)[、.：:]", line):
        return True
    return False


def _repair_report_citations(report: str, evidence: list[dict], context_metrics: dict | None = None) -> tuple[str, dict]:
    """Ask the model once to remove or ground unsupported numeric claims.

    This is a repair pass, not a fact-generation pass: the model may only use
    the existing Evidence IDs and must delete any unsupported number.
    """
    repair_context, metrics = compact_context([("原报告", report[:9000]), ("Evidence", _evidence_block(evidence))], budget_tokens=24000)
    if context_metrics is not None:
        context_metrics["_last_observation"] = metrics
    prompt = f"""请修复下面报告的引用问题，只能使用已提供的 Evidence ID。
规则：每个外部数字和事实后面必须有真实的 [E#]；如果证据中没有支持，就删除该数字或改写为“证据不足”；不得新增数字、URL、来源或事实。保留普通中文文本，不使用 Markdown。
上下文：
{repair_context}
只返回修复后的报告正文。"""
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        return to_plain_text(response.content).strip(), extract_usage(response)
    except Exception:
        return report, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}


def _repair_recommendation_traces(
    report: str, evidence: list[dict], conditions: list[UserCondition], context_metrics: dict | None = None,
) -> tuple[str, dict]:
    """One bounded repair for action-to-condition/evidence bindings."""
    repair_context, metrics = compact_context([("用户条件", _user_condition_block(conditions)),
                                               ("外部证据", _evidence_block(evidence)),
                                               ("原报告", report[:9000])], budget_tokens=24000)
    if context_metrics is not None:
        context_metrics["_last_observation"] = metrics
    prompt = f"""只修复报告“行动建议”章节的追溯编号，其他章节和事实不得改动。
每条行动建议必须：
1. 绑定至少一个真实用户条件 [U#]；
2. 绑定至少一个确实支持该建议的外部证据 [E#]；
3. 写明“验证标准：...”且可执行；
4. 不得创造不存在的编号、数字、事实或 URL。没有支持时把建议改成“先补充证据”，不能强行引用。

上下文：
{repair_context}

只返回修复后的完整报告正文，保持普通中文纯文本。"""
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        return to_plain_text(response.content).strip(), extract_usage(response)
    except Exception:
        return report, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}


def _remove_unsupported_numeric_precision(report: str) -> str:
    """Remove unsupported numeric precision after the bounded repair pass.

    This is intentionally conservative: only numbers with units on lines
    without an Evidence ID are generalized. The surrounding recommendation
    remains, but an unsupported exact figure cannot leak into a formal report.
    """
    lines = []
    for line in report.splitlines():
        if _EXTERNAL_NUMBER.search(line) and not re.search(r"\[E\d+\]", line):
            if not _is_non_external_number_line(line):
                line = _EXTERNAL_NUMBER.sub("相关指标", line)
        lines.append(line)
    return "\n".join(lines)


def _remove_unsupported_claim_lines(report: str, judgments: list) -> str:
    """Remove fact lines that failed the local Claim→Evidence judge.

    It is safer to omit an unsupported sentence than to present it as a fact.
    Recommendation and calculation lines are retained because their user
    condition/evidence bindings are validated separately.
    """
    rejected = {
        str(item.claim_text).strip()
        for item in judgments
        if item.verdict in {"unknown", "contradicted"}
        and getattr(item, "claim_text", "").strip()
    }
    if not rejected:
        return report
    return "\n".join(line for line in report.splitlines() if line.strip() not in rejected)


def _scenario_layout(scenario: str) -> str:
    return {
        "idea": "重点判断需求是否真实、预算是否匹配，以及如何用最低成本验证；不要提前展开完整开店或公司建设方案",
        "planning": "重点比较已确认的候选方案、预算分配和落地顺序",
        "operating_diagnosis": "重点定位经营问题、可验证原因和短周期行动实验",
        "existing_product_extension": "重点比较保留、局部改造和重建的成本收益与实施风险",
        "company_research": "重点回答用户指定的公司、产品、设备或业务链问题，不做泛化公司介绍",
        "option_comparison": "重点给出选择建议、适用条件和导致推荐变化的关键变量",
    }.get(scenario, "重点回答用户当前的具体决策，不扩展无关行业背景")


def report_node(state: ResearchState) -> dict:
    evidence = list(state.get("evidence", []))
    user_conditions = build_user_conditions(state)
    evidence_enabled = state.get("evidence_enabled", True)
    opinions = state.get("analyst_opinions", {})
    scenario = state.get("scenario", "market_question")
    opinion_text, initial_context_metrics = compact_context(
        [("用户条件", str(state.get("known_facts", {}))),
         ("分析师结论", str(opinions)),
         ("Evidence", _evidence_block(evidence)),
         ("冲突", str(state.get("conflicts", [])))],
        budget_tokens=int(state.get("context_token_budget", 24000) or 24000),
    )
    initial_context_metrics["budget"] = int(state.get("context_token_budget", 24000) or 24000)
    initial_context_metrics["preserved_condition_count"] = len(state.get("known_facts", {}) or {})
    initial_context_metrics["preserved_evidence_count"] = len(evidence)
    context_metrics = merge_context_metrics(state.get("context_metrics"), initial_context_metrics)
    prompt = f"""你是行业研究报告编辑。请生成与用户当前场景匹配的中文纯文本正式报告。

用户要解决的问题：{state.get('decision_goal', '')}
场景：{scenario}
行业或对象：{state.get('industry', 'unknown')}
地区：{state.get('region', '全国')}
预算：{state.get('budget', '未知')}
阶段：{state.get('business_stage', 'unknown')}
生成日期：{date.today().isoformat()}
用户已确认信息：{state.get('known_facts', {})}
用户条件编号：
{_user_condition_block(user_conditions)}
明确假设：{state.get('assumptions', [])}
证据冲突：{state.get('conflicts', [])}

本场景写作重点：{_scenario_layout(scenario)}。
不得补充无关的固定六维章节。第一段必须直接回答用户要解决的问题。

输出约束：
1. 只输出普通中文文本。不得使用 Markdown 标题、星号、反引号、表格竖线、代码块或 JSON。
2. {"每个外部数字和关键事实必须紧跟真实 Evidence ID [E#]；没有证据时写‘证据不足’。" if evidence_enabled else "本实验版本关闭结构化 Evidence，不要求 Evidence ID；不得补造数字，也不要虚构引用。"}
3. 不得虚构 URL、年份、规模、份额、价格或利润率。
4. 区分公开事实、用户输入和测算假设，不混用年份、地区和统计口径。
5. 筹备或构思场景可以做保守／中性／乐观情景，但必须展示假设，不能把假设写成事实。
6. 证据不足时同时给出已知事实、缺口、暂时建议和最低成本验证动作。
7. 这是一份辅助决策的精简报告，不是行业百科，也不是三个分析师意见的拼接。正文控制在 700 至 1000 个中文字符。
8. 严格使用以下结构，并让标题单独占一行：
决策结论：120 字以内，明确回答是否值得做、优先做什么、暂时不要做什么。
关键依据：最多 3 条，每条只保留一个最影响决策的事实或判断；相同数字只能出现一次。
行动建议：最多 4 条，每条包含具体动作、目的和验证标准，并按优先级排序。
主要风险：最多 3 条，只写可能改变结论的风险；一般性套话不要写。
未来 7 天：列出 3 个可以立即执行的动作。
“关键依据”“行动建议”“主要风险”“未来 7 天”必须各自单独占一行。每个章节内的条目必须分别另起一行，并依次以“第一、”“第二、”“第三、”“第四、”开头，绝不能把多个条目写在同一段。
9. 删除行业常识、空泛背景、重复结论、礼貌性开场、证据原文复述和与用户当前条件无关的数据。不得为了凑结构增加内容。
10. 每条关键依据最多使用两个 Evidence ID。来源列表由系统追加，不要在正文重复标题、摘录或 URL。
11. 如果证据不足，用一句话说明缺口及其对结论的影响，然后给出最低成本验证方法；不要围绕“证据不足”反复解释。
12. 如果存在证据冲突，必须在“主要风险”或“关键依据”中用一句话说明冲突的年份、地区或统计口径，并降低结论确定性；不得自行选择一条来源隐藏冲突。
13. {"每条‘行动建议’都必须同时标注至少一个适用的用户条件 [U#] 和一个支持它的外部证据 [E#]" if evidence_enabled else "每条‘行动建议’必须标注至少一个适用的用户条件 [U#]"}，并明确写出“验证标准：...”。不得使用不存在的编号。

上下文（已按预算压缩，必须保留用户条件和 Evidence ID）：
{opinion_text}
"""
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        body = to_plain_text(response.content)
        usage = extract_usage(response)
    except Exception as exc:
        body = (
            "行业研究建议\n\n"
            f"报告生成失败（{type(exc).__name__}）。以下为已验证证据，供人工复核。\n\n"
            + _evidence_block(evidence)
        )
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    citation_validation_error = ""
    try:
        validate_report_claims(body, evidence)
    except ValueError as exc:
        citation_validation_error = str(exc)
        body += f"\n\n证据校验提示\n• {exc}。当前结论仅供人工复核。"
    warnings = _citation_warnings(body, evidence)
    # One bounded repair pass prevents a single uncited model sentence from
    # downgrading an otherwise evidence-grounded report. It never invents
    # citations; failed repair remains an explicit insufficient result.
    if evidence and (citation_validation_error or warnings):
        repaired, repair_usage = _repair_report_citations(body, evidence, context_metrics)
        context_metrics = merge_context_metrics(context_metrics, context_metrics.pop("_last_observation", {}))
        if repaired and repaired != body:
            body = repaired
            usage = merge_usage(usage, repair_usage)
            citation_validation_error = ""
            try:
                validate_report_claims(body, evidence)
            except ValueError as exc:
                citation_validation_error = str(exc)
            warnings = _citation_warnings(body, evidence)
            if citation_validation_error or warnings:
                body = _remove_unsupported_numeric_precision(body)
                citation_validation_error = ""
                try:
                    validate_report_claims(body, evidence)
                except ValueError as exc:
                    citation_validation_error = str(exc)
                warnings = _citation_warnings(body, evidence)
    recommendation_traces, trace_warnings = extract_recommendation_traces(body, evidence, user_conditions)
    if trace_warnings and evidence and user_conditions:
        repaired, trace_usage = _repair_recommendation_traces(body, evidence, user_conditions, context_metrics)
        context_metrics = merge_context_metrics(context_metrics, context_metrics.pop("_last_observation", {}))
        if repaired and repaired != body:
            body = repaired
            usage = merge_usage(usage, trace_usage)
            citation_validation_error = ""
            try:
                validate_report_claims(body, evidence)
            except ValueError as exc:
                citation_validation_error = str(exc)
            warnings = _citation_warnings(body, evidence)
            recommendation_traces, trace_warnings = extract_recommendation_traces(body, evidence, user_conditions)
    recommendation_trace_status = (
        "validated" if recommendation_traces and not trace_warnings
        else "insufficient" if user_conditions else "not_applicable"
    )
    warning_block = ""
    all_warnings = list(warnings) + trace_warnings
    if all_warnings:
        warning_block = "\n\n自动追溯检查\n" + "\n".join(f"• {item}" for item in all_warnings)
    report = body.rstrip() + "\n\n来源\n" + _source_list(evidence) + warning_block
    report_claims, claim_judgments, entailment_rate, judge_status = validate_claims(body, evidence, context_metrics)
    context_metrics = merge_context_metrics(context_metrics, context_metrics.pop("_last_observation", {}))
    unsupported_claims = [item for item in claim_judgments if item.verdict != "entailed" and item.evidence_ids]
    if unsupported_claims and judge_status == "unavailable":
        body = _remove_unsupported_claim_lines(body, unsupported_claims)
        body = _remove_unsupported_numeric_precision(body)
        citation_validation_error = ""
        warnings = _citation_warnings(body, evidence)
        try:
            validate_report_claims(body, evidence)
        except ValueError as exc:
            citation_validation_error = str(exc)
        report_claims, claim_judgments, entailment_rate, judge_status = validate_claims(body, evidence, context_metrics)
        unsupported_claims = [item for item in claim_judgments if item.verdict != "entailed" and item.evidence_ids]
        recommendation_traces, trace_warnings = extract_recommendation_traces(body, evidence, user_conditions)
        all_warnings = list(warnings) + trace_warnings
        report = body.rstrip() + "\n\n来源\n" + _source_list(evidence) + (
            "\n\n自动追溯检查\n" + "\n".join(f"• {item}" for item in all_warnings) if all_warnings else ""
        )
    if unsupported_claims and not citation_validation_error:
        citation_validation_error = f"{len(unsupported_claims)} 条 Claim 未被对应摘录充分支持"
    formal_report = bool(evidence) and not citation_validation_error and not all_warnings
    if not evidence and not citation_validation_error:
        citation_validation_error = "no validated evidence was collected"
    return {
        "messages": [AIMessage(content=report)],
        "context_metrics": context_metrics,
        "research_complete": True,
        "report_generated": formal_report,
        "report_confirmation_pending": False,
        "report_confirmation_answer": "generate",
        "awaiting_user": False,
        "answer_type": "report" if formal_report else "evidence_insufficient",
        "completion_reason": state.get("completion_reason", "research_finished") if formal_report else "evidence_insufficient",
        "citation_status": "insufficient" if citation_validation_error or all_warnings else "validated",
        "citation_validation_error": citation_validation_error,
        "report_claims": [item.model_dump(mode="json") for item in report_claims],
        "claim_judgments": [item.model_dump(mode="json") for item in claim_judgments],
        "claim_entailment_rate": entailment_rate,
        "claim_judge_status": judge_status,
        "user_conditions": [item.model_dump(mode="json") for item in user_conditions],
        "recommendation_traces": [item.model_dump(mode="json") for item in recommendation_traces],
        "recommendation_trace_status": recommendation_trace_status,
        "llm_usage": usage,
        "selected_skill": "citation_review" if not formal_report else state.get("selected_skill", "market_research"),
    }
