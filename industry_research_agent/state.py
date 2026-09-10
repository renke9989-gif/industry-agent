"""LangGraph state and typed contracts for the web research agent."""
import os
from typing import Dict, List, Literal, Optional, TypedDict

try:  # Python 3.8 compatibility for the offline harness.
    from typing import Annotated
except ImportError:  # pragma: no cover - used only on legacy interpreters
    from typing_extensions import Annotated

from langchain_core.messages import BaseMessage
from contracts import Evidence
try:
    from langgraph.graph.message import add_messages
except ImportError:  # allows static/offline checks without LangGraph installed
    def add_messages(left, right):
        return list(left or []) + list(right or [])


DimensionName = Literal["市场", "竞争", "商业模式", "机会", "风险", "趋势"]
DimensionState = Literal["pending", "researching", "complete", "insufficient"]


def add_llm_usage(left: Optional[dict], right: Optional[dict]) -> dict:
    keys = ("input_tokens", "output_tokens", "total_tokens", "llm_calls")
    merged = {key: int((left or {}).get(key, 0) or 0) + int((right or {}).get(key, 0) or 0) for key in keys}
    if not merged["total_tokens"]:
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged


class ToolEvent(TypedDict, total=False):
    tool: str
    query: str
    url: str
    success: bool
    cached: bool
    duration_ms: int
    result_count: int
    error: str
    timestamp: str
    tool_source: str
    retry_count: int
    error_code: str


class ResearchState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    industry: str
    region: str
    target_freshness_year: int
    budget: str
    profile_confirmed: bool
    user_intent: str
    business_stage: str
    assumption_mode: bool
    research_stage: str
    research_plan: List[dict]
    evaluation_required_dimensions: List[str]
    assumptions: List[dict]
    interview_complete: bool
    interview_questions: List[str]
    interview_round: int
    soft_interview_checkpoint: int
    hard_interview_limit: int
    interview_target_questions: int
    interview_readiness_score: int
    current_interview_question: dict
    covered_topics: List[str]
    known_facts: Dict[str, str]
    scenario: str
    decision_goal: str
    decision_template_id: str
    routing_confidence: float
    answer_type: str
    complexity: str
    needs_web: bool
    needs_interview: bool
    needs_full_report: bool
    evidence_enabled: bool
    auto_report: bool
    blocking_unknowns: List[str]
    user_requested_start: bool

    scores: Dict[str, int]
    dimension_status: Dict[str, str]
    research_questions: List[str]
    evidence: List[Evidence]
    private_context: List[dict]
    private_retrieval_count: int
    tool_events: List[ToolEvent]
    trace_events: List[dict]
    last_run_metrics: dict
    llm_usage: Annotated[Dict[str, int], add_llm_usage]
    analyst_opinions: Dict[str, str]
    analyst_questions: Dict[str, str]

    next_agent: str
    research_complete: bool
    awaiting_user: bool
    pending_question: str
    missing_information: List[str]
    completion_reason: str
    user_turn_count: int
    agent_step_count: int
    supervisor_reason: str
    conflicts: List[dict]
    error_count: int
    force_finish: bool
    report_confirmation_pending: bool
    report_confirmation_answer: str
    report_generated: bool
    citation_status: str
    citation_validation_error: str
    report_claims: List[dict]
    claim_judgments: List[dict]
    claim_entailment_rate: Optional[float]
    claim_judge_status: str
    user_conditions: List[dict]
    recommendation_traces: List[dict]
    recommendation_trace_status: str
    followup_mode: bool
    context_metrics: dict
    context_token_budget: int
    selected_skill: str
    initial_skill: str
    skill_candidates: List[dict]
    skill_confidence: float
    skill_fallback_reason: str
    skill_trace: List[dict]
    routing_classifier: str


DIMENSIONS: List[str] = ["市场", "竞争", "商业模式", "机会", "风险", "趋势"]


def initial_state(user_message: BaseMessage) -> ResearchState:
    """Return a fresh, serializable state for a new research session."""
    return {
        "messages": [user_message],
        "industry": "unknown",
        "target_freshness_year": __import__("datetime").date.today().year,
        "region": "全国",
        "budget": "未知",
        "profile_confirmed": False,
        "user_intent": "entry_feasibility",
        "business_stage": "unknown",
        "assumption_mode": False,
        "research_stage": "understand",
        "research_plan": [],
        "assumptions": [],
        "interview_complete": False,
        "interview_questions": [],
        "interview_round": 0,
        "soft_interview_checkpoint": 5,
        "hard_interview_limit": 8,
        "interview_target_questions": 5,
        "interview_readiness_score": 0,
        "current_interview_question": {},
        "covered_topics": [],
        "known_facts": {},
        "scenario": "unknown",
        "decision_goal": "",
        "decision_template_id": "",
        "routing_confidence": 0.0,
        "answer_type": "",
        "complexity": "medium",
        "needs_web": True,
        "needs_interview": False,
        "needs_full_report": False,
        "evidence_enabled": True,
        "auto_report": False,
        "blocking_unknowns": [],
        "user_requested_start": False,
        "scores": {dimension: 0 for dimension in DIMENSIONS},
        "dimension_status": {dimension: "pending" for dimension in DIMENSIONS},
        "research_questions": [],
        "evidence": [],
        "private_context": [],
        "private_retrieval_count": 0,
        "tool_events": [],
        "trace_events": [],
        "last_run_metrics": {},
        "llm_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0},
        "analyst_opinions": {},
        "analyst_questions": {},
        "next_agent": "",
        "research_complete": False,
        "awaiting_user": False,
        "pending_question": "",
        "missing_information": [],
        "completion_reason": "",
        "user_turn_count": 1,
        "agent_step_count": 0,
        "supervisor_reason": "",
        "conflicts": [],
        "error_count": 0,
        "force_finish": False,
        "report_confirmation_pending": False,
        "report_confirmation_answer": "",
        "report_generated": False,
        "citation_status": "pending",
        "citation_validation_error": "",
        "report_claims": [],
        "claim_judgments": [],
        "claim_entailment_rate": None,
        "claim_judge_status": "unavailable",
        "user_conditions": [],
        "recommendation_traces": [],
        "recommendation_trace_status": "pending",
        "followup_mode": False,
        "context_metrics": {},
        "context_token_budget": int(os.getenv("CONTEXT_TOKEN_BUDGET", "24000")),
        "selected_skill": "",
        "initial_skill": "",
        "skill_candidates": [],
        "skill_confidence": 0.0,
        "skill_fallback_reason": "",
        "skill_trace": [],
        "routing_classifier": "rules",
    }
