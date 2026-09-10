"""Short, evidence-grounded answers after a research report is complete."""
from __future__ import annotations

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from agents.web_research import get_llm
from providers import get_default_provider
from research_tools import build_evidence
from output_format import to_plain_text
from llm_usage import extract_usage
from triage import detect_region
from context_manager import compact_context, merge_context_metrics


JOB_TERMS = ("岗位", "工作机会", "求职", "投递", "招聘", "薪资", "工资", "校招", "实习", "面试", "offer")
REFERENCE_TERMS = ("这个", "这类", "这种", "这些", "它", "该方向", "刚才", "中厂", "大厂", "小厂", "薪资", "待遇")


def _recent_user_questions(state: dict, limit: int = 2) -> list[str]:
    questions = []
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            text = str(message.content).strip()
            if text:
                questions.append(text)
        if len(questions) >= limit:
            break
    return list(reversed(questions))


def resolve_followup_context(state: dict, question: str) -> dict:
    """Resolve an elliptical follow-up without inventing a new topic."""
    previous_questions = _recent_user_questions(state)
    recent = previous_questions[-1] if previous_questions else ""
    combined = f"{recent} {question}".strip()
    needs_context = any(term in question for term in REFERENCE_TERMS) or len(question) <= 24
    job_context = any(term.lower() in combined.lower() for term in JOB_TERMS)
    normalized = question
    if job_context:
        normalized = normalized.replace("中厂", "中型科技公司").replace("大厂", "大型科技公司").replace("小厂", "小型科技公司")
    # The previous user question is safer than an old broad industry label for
    # pronoun/ellipsis resolution. It is used as context, not as a new fact.
    topic_context = recent if needs_context and recent else str(state.get("industry", ""))
    return {
        "previous_questions": previous_questions,
        "topic_context": topic_context,
        "normalized_question": normalized,
        "job_context": job_context,
    }


def _evidence_text(items: list[dict]) -> str:
    return "\n".join(
        f"[{item.get('id')}] {item.get('source_title', '')}: {item.get('excerpt', '')[:500]} ({item.get('source_url', '')})"
        for item in items
    ) or "暂无相关证据"


def answer_followup(state: dict, question: str) -> dict:
    """Research one narrow follow-up question and return an answer, never a new report."""
    provider = get_default_provider()
    evidence = list(state.get("evidence", []))
    events = list(state.get("tool_events", []))
    previous_region = state.get("region", "全国")
    current_region = detect_region(question) or previous_region
    resolved = resolve_followup_context(state, question)
    query = f"{resolved['topic_context']} {current_region} {resolved['normalized_question']} 招聘数据 {date.today().year}".strip() if resolved["job_context"] else (
        f"{resolved['topic_context']} {current_region} {resolved['normalized_question']} {date.today().year}".strip()
    )
    new_evidence = []
    try:
        response = provider.search(query, max_results=3, depth="advanced")
        events.append({
            "tool": "followup_search", "query": query, "success": bool(response.results),
            "result_count": len(response.results), "tool_source": response.provider,
        })
        for result in response.results[:2]:
            source = result.model_dump(mode="json")
            excerpt = source.get("snippet", "")
            if not excerpt:
                continue
            item = build_evidence(
                evidence_id=f"E{len(evidence) + len(new_evidence) + 1}",
                dimension="机会",
                claim=excerpt[:300],
                source=source,
                excerpt=excerpt,
                period=source.get("published_at") or "",
                region=current_region,
            )
            new_evidence.append(item)
    except Exception as exc:
        events.append({"tool": "followup_search", "query": query, "success": False, "error": f"{type(exc).__name__}: {exc}"})

    context = _evidence_text(new_evidence or evidence[-8:])
    context, context_metrics = compact_context(
        [("最近用户问题", "；".join(resolved['previous_questions'])),
         ("当前问题", question), ("证据", context), ("用户条件", str(state.get("known_facts", {})))],
        budget_tokens=int(state.get("context_token_budget", 24000) or 24000),
    )
    context_metrics["preserved_condition_count"] = len(state.get("known_facts", {}) or {})
    context_metrics["preserved_evidence_count"] = len(new_evidence or evidence[-8:])
    context_metrics["budget"] = int(state.get("context_token_budget", 24000) or 24000)
    prompt = f"""你是行业研究助手。用户已经收到完整报告，现在只是在追问一个具体问题。

问题：{question}
最近的用户问题：{'；'.join(resolved['previous_questions']) or '无'}。
承接后的当前主题：{resolved['topic_context'] or state.get('industry', '未知')}。
行业：{state.get('industry', '未知')}；本轮目标地区：{current_region}。
此前地区条件：{previous_region}。如果用户本轮明确了新城市或地区，以本轮条件为准，这是条件更新，不是概念矛盾。
城市层级是地区约束，不得称为“行业定位”。
如果当前问题省略了主语，例如“这类岗位”“中厂薪资”“那上海呢”，必须承接最近的用户问题，不得擅自切换行业。
求职语境中的“大厂、中厂、小厂”分别指大型、中型、小型科技公司，不得解释为制造工厂。

请直接回答问题，控制在 400 字以内，不要重新生成完整报告，不要复述六维度模板，不要使用 Markdown 标题、表格或星号。
只使用以下证据；关键事实后标注已有证据编号 [E#]。若证据不足，请明确说明，并说明还需要比较什么信息。

证据和上下文：
{context}
"""
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        answer = to_plain_text(response.content)
        usage = extract_usage(response)
    except Exception as exc:
        answer = f"暂时无法生成追问答案：{type(exc).__name__}。现有证据不足以可靠回答这个问题。"
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    return {
        "messages": [AIMessage(content=answer)],
        "evidence": evidence + new_evidence,
        "tool_events": events,
        "research_complete": True,
        "followup_mode": True,
        "region": current_region,
        "followup_context": resolved,
        "llm_usage": usage,
        "context_metrics": merge_context_metrics(state.get("context_metrics"), context_metrics),
        "selected_skill": "followup_answer",
    }
