"""Shared evidence-first web research loop for all analyst nodes."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from research_tools import build_evidence, evidence_is_sufficient
from context_manager import compact_context, merge_context_metrics
from providers import ResearchProvider, get_default_provider
from state import Evidence, ResearchState
from contracts import Evidence as EvidenceModel, EvidenceExtraction
from output_format import to_plain_text
from llm_usage import extract_usage
from private_rag import PrivateDocumentStore


load_dotenv()
_llm = None
_private_store = None


def _private_hits(query: str, top_k: int = 3) -> list[dict]:
    global _private_store
    if os.getenv("PRIVATE_RAG_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return []
    if _private_store is None:
        _private_store = PrivateDocumentStore(os.getenv("PRIVATE_RAG_SQLITE_PATH", ".data/private_rag.sqlite3"))
    return _private_store.search(query, top_k=top_k)


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

DIMENSION_QUERY_HINTS = {
    "市场": "市场规模 增长率 官方统计 政府协会 报告",
    "趋势": "发展趋势 政策 官方发布 政府协会 最新",
    "机会": "细分市场 需求增长 消费者 机会 行业协会",
    "竞争": "主要企业 市场份额 竞争格局 集中度 财报 年报",
    "风险": "进入壁垒 监管 政策 官方发布 风险 失败 原因",
    "商业模式": "毛利率 成本结构 客单价 盈利模式 财报 年报",
}


def planned_dimensions(state: ResearchState, owned_dimensions: Sequence[str]) -> List[str]:
    """Limit an analyst to dimensions selected by the current Research Plan."""
    planned = {
        dimension
        for step in state.get("research_plan", [])
        for dimension in step.get("dimensions", [])
    }
    if not planned:
        return list(owned_dimensions)
    return [dimension for dimension in owned_dimensions if dimension in planned]


def plan_queries(industry: str, region: str, dimensions: Sequence[str], target_year: int | None = None) -> List[str]:
    """Create focused, freshness-aware web queries without relying on an LLM."""
    current_year = int(target_year or date.today().year)
    year = f"{current_year - 1} {current_year}"
    scope = "" if not region or region == "全国" else region
    queries = []
    for dimension in dimensions:
        hint = DIMENSION_QUERY_HINTS[dimension]
        # Keep a freshness term in every query and make the preferred source
        # class explicit; the ranking layer still accepts non-official sources
        # when no first-party material is available.
        source_hint = ""
        if dimension in {"市场", "趋势", "风险"}:
            source_hint = " (site:gov.cn OR site:org.cn)"
        elif dimension in {"竞争", "商业模式"}:
            source_hint = " (财报 OR 年报 OR 公司公告)"
        queries.append(f"{industry} {scope} {hint} {year} 当前最新数据{source_hint}".strip())
    return queries


def _extract_value(text: str) -> tuple[str, str, str]:
    """Best-effort extraction for display only; the original excerpt stays authoritative."""
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(万亿元|亿元|万元|元)", "amount"),
        (r"(\d+(?:\.\d+)?)\s*(%|％)", "percent"),
    ]
    for pattern, _ in patterns:
        match = re.search(pattern, text)
        if match:
            period = ""
            year = re.search(r"(20\d{2})年?", text)
            if year:
                period = year.group(1)
            return match.group(1), match.group(2), period
    return "", "", ""


def _unique_sources(results: Iterable[dict], limit: int = 3, target_year: int | None = None) -> List[dict]:
    selected: List[dict] = []
    hosts = set()
    target_year = int(target_year or date.today().year)
    def rank(item: dict):
        published = str(item.get("published_at") or "")
        freshness = 1 if str(target_year) in published else (0.5 if str(target_year - 1) in published else 0)
        return (freshness, item.get("source_type") == "official", item.get("source_type") == "company", item.get("score", 0))
    for result in sorted(results, key=rank, reverse=True):
        host = re.sub(r"^www\.", "", re.sub(r"^https?://", "", result.get("url", "")).split("/")[0])
        if not host or host in hosts:
            continue
        hosts.add(host)
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected


def validate_evidence(item: dict) -> dict | None:
    """Validate and normalize a candidate before it enters ResearchState."""
    try:
        validated = EvidenceModel.model_validate(item).model_dump(mode="json")
        value = str(validated.get("value") or "").strip()
        excerpt = re.sub(r"[\s,，]", "", validated.get("excerpt", ""))
        if value and re.sub(r"[\s,，]", "", value) not in excerpt:
            # A model/regex value that cannot be found in the source excerpt is
            # not allowed to enter state as a grounded external number.
            validated["value"] = None
            validated["unit"] = None
            validated["confidence"] = max(0.0, round(validated["confidence"] - 0.2, 2))
        return validated
    except Exception:
        return None


def extract_structured_evidence(
    *, evidence_id: str, dimension: str, source: dict, excerpt: str, region: str,
) -> EvidenceExtraction:
    """Convert one search/page candidate into the shared validated envelope.

    This boundary deliberately sits between web content and ResearchState. A
    malformed URL, short excerpt, unsupported source type, or an ungrounded
    numeric value is rejected here rather than leaking raw model/tool output
    into the report. The extraction is deterministic first; an LLM may later
    rewrite the claim, but it cannot bypass this Pydantic gate.
    """
    value, unit, period = _extract_value(excerpt)
    item = build_evidence(
        evidence_id=evidence_id,
        dimension=dimension,
        claim=excerpt[:300],
        source=source,
        excerpt=excerpt,
        value=value,
        unit=unit,
        period=period or source.get("published_at", ""),
        region=region,
    )
    validated = validate_evidence(item)
    if not validated:
        return EvidenceExtraction(items=[], missing_fields=["valid_url_or_excerpt"])
    missing = []
    if not validated.get("published_at"):
        missing.append("published_at")
    if not validated.get("period"):
        missing.append("period")
    if not validated.get("value"):
        missing.append("value")
    return EvidenceExtraction(
        items=[EvidenceModel.model_validate(validated)],
        missing_fields=missing,
    )


def extract_evidence_with_retry(
    *, evidence_id: str, dimension: str, source: dict, excerpt: str, region: str,
    use_llm: bool = True,
) -> tuple[EvidenceExtraction, int]:
    """Extract evidence with a structured-output retry, then safe fallback.

    The deterministic extractor remains the safety net. When enabled, the LLM
    may normalize the claim/value/period, but source URL/title/publisher fields
    are always taken from the search result and passed through the same
    Pydantic validator before entering state.
    """
    baseline = extract_structured_evidence(
        evidence_id=evidence_id, dimension=dimension, source=source,
        excerpt=excerpt, region=region,
    )
    if not use_llm:
        return baseline, 0
    prompt = f"""从下面的网页摘录中抽取一条事实证据，只能使用摘录中明确出现的信息。
返回 EvidenceExtraction JSON，items 最多一条；如果没有可验证事实，items 为空并填写 missing_fields。
维度：{dimension}
地区：{region}
来源标题：{source.get('title', '')}
来源 URL：{source.get('url', '')}
网页摘录：{excerpt[:1800]}
"""
    for attempt in range(2):
        try:
            llm = get_llm()
            if hasattr(llm, "with_structured_output"):
                result = llm.with_structured_output(EvidenceExtraction).invoke([HumanMessage(content=prompt)])
            else:
                result = llm.invoke([HumanMessage(content=prompt)])
                content = result.content if hasattr(result, "content") else result
                content = str(content).strip().removeprefix("```").removesuffix("```").strip()
                result = EvidenceExtraction.model_validate(json.loads(content))
            if not isinstance(result, EvidenceExtraction):
                result = EvidenceExtraction.model_validate(result)
            if not result.items:
                continue
            base_item = baseline.items[0].model_dump(mode="json") if baseline.items else None
            if not base_item:
                continue
            candidate = result.items[0].model_dump(mode="json")
            # Trust web-tool metadata over model-generated source metadata.
            for key in ("source_title", "source_url", "source_type", "published_at", "retrieved_at", "is_primary_source", "is_reprint"):
                candidate[key] = base_item[key]
            candidate["id"] = evidence_id
            candidate["dimension"] = dimension
            candidate["region"] = region
            validated = validate_evidence(candidate)
            if validated:
                return EvidenceExtraction(items=[EvidenceModel.model_validate(validated)]), attempt + 1
        except Exception:
            continue
    return baseline, 2


def detect_evidence_conflicts(items: Sequence[dict]) -> List[dict]:
    conflicts = []
    for index, left in enumerate(items):
        if not left.get("value"):
            continue
        for right in items[index + 1:]:
            if left.get("dimension") != right.get("dimension") or not right.get("value"):
                continue
            if left.get("unit") != right.get("unit"):
                continue
            if left.get("period") and right.get("period") and left.get("period") != right.get("period"):
                conflicts.append({
                    "dimension": left.get("dimension"), "reason": "period_mismatch",
                    "evidence_ids": [left.get("id"), right.get("id")],
                    "details": [left.get("period"), right.get("period")],
                })
            elif left.get("region") and right.get("region") and left.get("region") != right.get("region"):
                conflicts.append({
                    "dimension": left.get("dimension"), "reason": "region_mismatch",
                    "evidence_ids": [left.get("id"), right.get("id")],
                    "details": [left.get("region"), right.get("region")],
                })
    return conflicts


def research_dimensions(
    state: ResearchState,
    *,
    analyst_name: str,
    dimensions: Sequence[str],
    role_instruction: str,
    max_tool_calls: int = 4,
    provider: ResearchProvider | None = None,
) -> dict:
    """Run search -> fetch -> evidence -> verify -> synthesize within a hard budget."""
    industry = state.get("industry", "unknown")
    region = state.get("region", "全国")
    evidence: List[Evidence] = list(state.get("evidence", []))
    events = list(state.get("tool_events", []))
    questions = list(state.get("research_questions", []))
    status = dict(state.get("dimension_status", {}))
    scores = dict(state.get("scores", {}))
    new_evidence: List[Evidence] = []
    raw_sources: List[dict] = []
    evidence_enabled = state.get("evidence_enabled", True)
    calls = 0
    provider = provider or get_default_provider()

    active_dimensions = planned_dimensions(state, dimensions)
    remaining_dimensions = [
        dimension for dimension in active_dimensions
        if status.get(dimension, "pending") in ("pending", "researching")
    ]
    target_year = int(state.get("target_freshness_year") or date.today().year)
    queries = plan_queries(industry, region, remaining_dimensions, target_year)
    private_context = list(state.get("private_context", []))
    private_retrieval_count = int(state.get("private_retrieval_count", 0) or 0)
    for dimension, query in zip(remaining_dimensions, queries):
        if calls >= max_tool_calls:
            break
        questions.append(query)
        private_hits = _private_hits(query)
        if private_hits:
            private_context.extend({**hit, "dimension": dimension} for hit in private_hits)
            private_retrieval_count += 1
            events.append({"tool": "private_retrieve", "query": query, "success": True,
                           "result_count": len(private_hits), "tool_source": "private_rag", "duration_ms": 0})
        status[dimension] = "researching"
        search_started = time.perf_counter()
        response = provider.search(query, max_results=5, depth="advanced")
        results = [item.model_dump(mode="json") for item in response.results]
        provider_event = dict(response.event or {})
        event = {
            "tool": "search", "query": query, "success": bool(results),
            "cached": response.cached, "result_count": len(results),
            "tool_source": response.provider,
            "duration_ms": int((time.perf_counter() - search_started) * 1000),
        }
        for key in ("error", "error_code", "retry_count"):
            if key in provider_event:
                event[key] = provider_event[key]
        events.append(event)
        calls += 1

        for source_index, source in enumerate(_unique_sources(results, limit=3, target_year=target_year)):
            # Search snippets are valid candidate evidence. Fetch only the
            # highest-ranked page while budget remains, so one dimension cannot
            # starve the analyst's other dimensions.
            if source_index == 0 and calls < max_tool_calls:
                fetch_started = time.perf_counter()
                page_response = provider.fetch_page(source["url"])
                page = page_response.model_dump(mode="json")
                provider_event = dict(page_response.event or {})
                fetch_event = {"tool": "fetch_page", "url": source["url"], "success": bool(page.get("text")),
                               "tool_source": page_response.provider,
                               "duration_ms": int((time.perf_counter() - fetch_started) * 1000)}
                for key in ("error", "error_code", "retry_count"):
                    if key in provider_event:
                        fetch_event[key] = provider_event[key]
                events.append(fetch_event)
                calls += 1
            else:
                page = {"text": ""}
            excerpt = page.get("text", "")[:1200] or source.get("snippet", "")
            if not excerpt:
                continue
            if not evidence_enabled:
                raw_sources.append({
                    "dimension": dimension, "source_title": source.get("title", ""),
                    "source_url": source.get("url", ""), "excerpt": excerpt,
                    "period": source.get("published_at", ""),
                })
                continue
            extraction, extraction_retries = extract_evidence_with_retry(
                evidence_id=f"E{len(evidence) + len(new_evidence) + 1}",
                dimension=dimension,
                source=source,
                excerpt=excerpt,
                region=region,
                use_llm=(provider.name != "fake" and os.getenv("EVIDENCE_LLM_EXTRACTION", "true").lower() not in ("0", "false", "no")),
            )
            if extraction_retries:
                events.append({"tool": "extract_evidence", "success": bool(extraction.items),
                               "tool_source": "llm_structured", "retry_count": max(0, extraction_retries - 1),
                               "duration_ms": 0})
            for extracted in extraction.items:
                validated = extracted.model_dump(mode="json")
                if not any(existing.get("source_url") == validated["source_url"] and existing.get("dimension") == dimension for existing in evidence + new_evidence):
                    new_evidence.append(validated)

        dimension_evidence = [item for item in evidence + new_evidence if item.get("dimension") == dimension]
        dimension_raw = [item for item in raw_sources if item.get("dimension") == dimension]
        if (evidence_enabled and evidence_is_sufficient(dimension_evidence)) or (not evidence_enabled and len(dimension_raw) >= 2):
            status[dimension] = "complete"
            scores[dimension] = max(scores.get(dimension, 0), 4)
        else:
            status[dimension] = "insufficient"
            scores[dimension] = max(scores.get(dimension, 0), 1 if (dimension_evidence or dimension_raw) else 0)

    evidence.extend(new_evidence)
    conflicts = list(state.get("conflicts", [])) + detect_evidence_conflicts(new_evidence)
    relevant = [item for item in evidence if item.get("dimension") in active_dimensions]
    private_text = "\n".join(
        f"[P{index}] {item.get('title', '')} / chunk {item.get('chunk_index', 0)}"
        f"\n{item.get('text', '')[:700]}"
        for index, item in enumerate(private_context[-12:], start=1)
    )
    if evidence_enabled:
        evidence_text = "\n".join(
            f"[{item['id']}] {item.get('source_title', '')} | {item.get('period', '')} | "
            f"{item.get('excerpt', '')[:500]} | {item.get('source_url', '')}"
            for item in relevant
        ) or "没有取得可靠公开证据"
        citation_instruction = "每个数字和关键事实后必须标注证据编号，例如 [E1]。"
    else:
        evidence_text = "\n".join(
            f"{item.get('source_title', '')} | {item.get('period', '')} | {item.get('excerpt', '')[:500]}"
            for item in raw_sources
        ) or "没有取得搜索资料"
        citation_instruction = "该实验版本不启用结构化 Evidence，不要求证据编号，但仍不得补造数字。"
    if private_text:
        evidence_text = evidence_text + "\n\n私域文档检索结果（P#，仅作内部上下文；外部事实仍必须绑定 E#）：\n" + private_text
    evidence_text, context_metrics = compact_context(
        [("用户条件", str(state.get("known_facts", {}))),
         ("研究问题", str(state.get("decision_goal", ""))),
         ("证据", evidence_text)],
        budget_tokens=int(state.get("context_token_budget", 24000) or 24000),
    )
    context_metrics["preserved_condition_count"] = len(state.get("known_facts", {}) or {})
    context_metrics["preserved_evidence_count"] = len(relevant)
    context_metrics["budget"] = int(state.get("context_token_budget", 24000) or 24000)
    prompt = f"""{role_instruction}

当前行业：{industry}；地区：{region}；预算：{state.get('budget', '未知')}。
用户场景：{state.get('scenario', 'unknown')}；用户要解决的问题：{state.get('decision_goal', '')}。
本节点只研究：{'、'.join(active_dimensions)}。请先回应用户要解决的问题，不要扩展无关的固定模板章节。
请只依据下面的资料输出中文分析，不得补造数字。{citation_instruction}
若来源口径、年份或地区不一致，要明确说明；证据不足时直接写“证据不足”。
只输出普通中文文本，不使用 Markdown 标题、星号、表格、代码块或 Markdown 链接。
严格使用以下简洁结构：
结论：进入 / 谨慎进入 / 观望 / 不建议
理由：用一到两句话说明与用户条件最相关的判断依据
关键发现一：一句话结论，最多附两个最直接的 Evidence ID
关键发现二：一句话结论，最多附两个最直接的 Evidence ID
关键发现三：一句话结论，最多附两个最直接的 Evidence ID
每个“关键发现”必须从新的一行开始，不能把两条发现写在同一段。
总长度控制在 280 至 360 个中文字符。不要复述原始网页摘录，不要堆砌相近数字，不要输出 URL；证据弱或与用户决策无关的信息直接省略。

证据：
{evidence_text}
"""
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        opinion = to_plain_text(response.content)
        usage = extract_usage(response)
    except Exception as exc:
        opinion = f"证据已收集，但分析生成失败：{type(exc).__name__}。\n\n{evidence_text[:1600]}"
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}

    opinions = dict(state.get("analyst_opinions", {}))
    opinions[analyst_name] = opinion
    return {
        "messages": [AIMessage(content=opinion)],
        "research_questions": questions,
        "evidence": evidence,
        "private_context": private_context[-20:],
        "private_retrieval_count": private_retrieval_count,
        "tool_events": events,
        "dimension_status": status,
        "scores": scores,
        "analyst_opinions": opinions,
        "context_metrics": merge_context_metrics(state.get("context_metrics"), context_metrics),
        "agent_step_count": state.get("agent_step_count", 0) + 1,
        "llm_usage": usage,
        "conflicts": conflicts,
    }
