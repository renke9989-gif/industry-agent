"""Claim-to-Evidence validation for citation grounding and evaluation."""
from __future__ import annotations

import json
import os
import re
from typing import Iterable

from contracts import ClaimJudgment, Evidence, ReportClaim
from context_manager import compact_context


EXTERNAL_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|万亿元|亿元|万元|元|万人|家|次|个月|天)")
EVIDENCE_ID = re.compile(r"\[(E\d+)\]")


def extract_claims(report: str, evidence: Iterable[Evidence]) -> list[ReportClaim]:
    """Extract report claims while preserving the existing plain-text UI."""
    valid_ids = {str(item.get("id")) for item in evidence}
    claims: list[ReportClaim] = []
    for line in report.splitlines():
        text = line.strip()
        if not text:
            continue
        if any(label in text for label in ("预算", "报告生成日期", "生成日期", "情景", "步骤", "假设", "投入=")):
            continue
        ids = [item for item in EVIDENCE_ID.findall(text) if item in valid_ids]
        if ids or EXTERNAL_NUMBER.search(text):
            if any(term in text for term in ("行动建议", "验证标准", "验收标准", "未来", "第一、", "第二、", "第三、", "第四、")):
                claim_type = "recommendation"
            elif any(term in text for term in ("ROI", "回本", "收益", "公式", "测算")):
                claim_type = "calculation"
            else:
                claim_type = "fact"
            claims.append(ReportClaim(text=text, evidence_ids=ids, claim_type=claim_type))
    return claims


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}|\d+(?:\.\d+)?", text.lower())
    return set(words)


def _number_supported(number: str, excerpt: str) -> bool:
    """Match numeric claims despite harmless spacing/punctuation/unit variants."""
    compact_claim = re.sub(r"[\s,，]", "", number)
    compact_excerpt = re.sub(r"[\s,，]", "", excerpt)
    if compact_claim in compact_excerpt:
        return True
    # Reports sometimes shorten 亿元/万元 to 亿/万. Keep the numeric value
    # and accept only that narrow unit abbreviation, never a different value.
    for full, short in (("亿元", "亿"), ("万元", "万"), ("万人", "万")):
        if full in compact_claim and compact_claim.replace(full, short) in compact_excerpt:
            return True
    return False


def _rule_judge(claim: ReportClaim, evidence_by_id: dict[str, Evidence]) -> ClaimJudgment:
    excerpts = [str(evidence_by_id[item].get("excerpt", "")) for item in claim.evidence_ids if item in evidence_by_id]
    joined = " ".join(excerpts)
    if not excerpts:
        return ClaimJudgment(claim_text=claim.text, evidence_ids=claim.evidence_ids, verdict="unknown",
                             rationale="没有匹配到有效 Evidence ID", judge_source="rules", confidence=0.0)
    if claim.claim_type in {"recommendation", "calculation"}:
        return ClaimJudgment(claim_text=claim.text, evidence_ids=claim.evidence_ids, verdict="entailed",
                             rationale="行动建议或计算结果不作为原始外部事实判定；其用户条件和证据绑定单独校验",
                             judge_source="rules", confidence=0.8)
    numbers = EXTERNAL_NUMBER.findall(claim.text)
    if numbers and all(_number_supported(number, joined) for number in numbers):
        return ClaimJudgment(claim_text=claim.text, evidence_ids=claim.evidence_ids, verdict="entailed",
                             rationale="外部数字均能在摘录中找到", judge_source="rules", confidence=0.95)
    overlap = len(_tokens(claim.text) & _tokens(joined)) / max(1, len(_tokens(claim.text)))
    if overlap >= 0.18:
        return ClaimJudgment(claim_text=claim.text, evidence_ids=claim.evidence_ids, verdict="entailed",
                             rationale=f"Claim 与摘录关键词重合率 {overlap:.2f}", judge_source="rules", confidence=round(min(0.85, overlap + 0.35), 2))
    return ClaimJudgment(claim_text=claim.text, evidence_ids=claim.evidence_ids, verdict="unknown",
                         rationale="摘录与 Claim 的数字或语义重合不足", judge_source="rules", confidence=round(overlap, 2))


def _llm_judge(claims: list[ReportClaim], evidence: list[Evidence], context_metrics: dict | None = None) -> list[ClaimJudgment] | None:
    api_key = os.getenv("EVAL_JUDGE_API_KEY")
    model = os.getenv("EVAL_JUDGE_MODEL")
    if not api_key or not model or not claims:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, openai_api_key=api_key, base_url=os.getenv("EVAL_JUDGE_BASE_URL") or None, temperature=0)
        payload = {"claims": [item.model_dump() for item in claims], "evidence": [dict(item) for item in evidence]}
        context, metrics = compact_context([("claims", json.dumps(payload["claims"], ensure_ascii=False)),
                                            ("evidence", json.dumps(payload["evidence"], ensure_ascii=False))], budget_tokens=24000)
        if context_metrics is not None:
            context_metrics.update(metrics)
            context_metrics["_last_observation"] = metrics
        prompt = (
            "判断每条 Claim 是否被对应 Evidence 摘录支持。只能返回 JSON 数组，每项包含 "
            "claim_text、evidence_ids、verdict(entailed/contradicted/unknown)、rationale、confidence。"
            "不得添加证据之外的事实。\n" + context
        )
        response = llm.invoke(prompt)
        raw = str(getattr(response, "content", response)).strip().removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        items = parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            return None
        return [ClaimJudgment.model_validate({**item, "judge_source": "llm"}) for item in items]
    except Exception:
        return None


def validate_claims(report: str, evidence: list[Evidence], context_metrics: dict | None = None) -> tuple[list[ReportClaim], list[ClaimJudgment], float | None, str]:
    claims = extract_claims(report, evidence)
    by_id = {str(item.get("id")): item for item in evidence}
    rules = [_rule_judge(claim, by_id) for claim in claims]
    llm = _llm_judge(claims, evidence, context_metrics)
    judgments = llm if llm is not None and len(llm) == len(claims) else rules
    if llm is None:
        return claims, judgments, None, "unavailable"
    supported = sum(item.verdict == "entailed" for item in judgments)
    return claims, judgments, supported / len(judgments) if judgments else 1.0, "llm"
