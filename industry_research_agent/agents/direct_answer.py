"""Narrow answer path for questions that do not need a full report."""
from __future__ import annotations

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from agents.web_research import get_llm
from providers import get_default_provider
from research_tools import build_evidence
from output_format import to_plain_text
from llm_usage import extract_usage
from context_manager import compact_context, merge_context_metrics


def direct_answer_node(state: dict) -> dict:
    question = next(
        (str(message.content).strip() for message in reversed(state.get("messages", []))
         if isinstance(message, HumanMessage)),
        "",
    )
    provider = get_default_provider()
    evidence = list(state.get("evidence", []))
    events = list(state.get("tool_events", []))
    query = f"{state.get('industry', '')} {state.get('region', '')} {question} {date.today().year}".strip()
    new_evidence = []
    try:
        response = provider.search(query, max_results=4, depth="advanced")
        events.append({"tool": "direct_answer_search", "query": query, "success": bool(response.results),
                       "result_count": len(response.results), "tool_source": response.provider})
        for result in response.results[:3]:
            source = result.model_dump(mode="json")
            excerpt = source.get("snippet", "")
            if not excerpt:
                continue
            new_evidence.append(build_evidence(
                evidence_id=f"E{len(evidence) + len(new_evidence) + 1}",
                dimension="机会",
                claim=excerpt[:300],
                source=source,
                excerpt=excerpt,
                period=source.get("published_at") or "",
                region=state.get("region", "全国"),
            ))
    except Exception as exc:
        events.append({"tool": "direct_answer_search", "query": query, "success": False,
                       "error": f"{type(exc).__name__}: {exc}"})

    context = "\n".join(
        f"[{item.get('id')}] {item.get('source_title', '')}：{item.get('excerpt', '')[:600]}；链接：{item.get('source_url', '')}"
        for item in (new_evidence or evidence[-5:])
    ) or "暂无可靠公开证据"
    context, context_metrics = compact_context(
        [("用户问题", question), ("研究资料", context)],
        budget_tokens=int(state.get("context_token_budget", 24000) or 24000),
    )
    context_metrics["preserved_evidence_count"] = len(new_evidence or evidence[-5:])
    context_metrics["budget"] = int(state.get("context_token_budget", 24000) or 24000)
    prompt = f"""你是行业研究助手。用户只问了一个具体问题，不要生成完整行业报告，不要套用市场、竞争、商业模式六维模板。
请第一句话直接回应用户的问题，然后用简短段落说明依据、限制和下一步，控制在 400 字以内。只使用普通中文，不使用 Markdown 标题、表格、代码块或原始 JSON。外部事实后标注真实证据编号 [E#]；证据不足就明确说明。

用户问题：{question}
场景：{state.get('scenario', 'market_question')}
研究资料：
{context}
"""
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        answer = to_plain_text(response.content)
        usage = extract_usage(response)
    except Exception as exc:
        answer = f"暂时无法完成联网回答（{type(exc).__name__}）。目前没有足够证据支持可靠结论。"
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    return {
        "messages": [AIMessage(content=answer)],
        "evidence": evidence + new_evidence,
        "tool_events": events,
        "research_complete": True,
        "answer_type": "direct_answer",
        "completion_reason": "direct_question_answered",
        "next_agent": "DIRECT_ANSWER",
        "llm_usage": usage,
        "context_metrics": merge_context_metrics(state.get("context_metrics"), context_metrics),
        "selected_skill": "market_research",
    }
