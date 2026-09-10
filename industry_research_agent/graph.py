"""LangGraph workflow for the evidence-first web research agent."""
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.business_analyst import business_analyst_node
from agents.competition_analyst import competition_analyst_node
from agents.direct_answer import direct_answer_node
from agents.market_analyst import market_analyst_node
from agents.supervisor import supervisor_node
from report_generator import report_node
from report_confirmation import report_confirmation_node
from state import ResearchState
from skill_registry import validate_skill

try:
    from langgraph.types import RetryPolicy
except ImportError:  # pragma: no cover - legacy/offline interpreter
    RetryPolicy = None


def _retryable_node_error(exc: Exception) -> bool:
    if isinstance(exc, (ValueError, TypeError, ImportError, KeyError, AssertionError)):
        return False
    message = str(exc).lower()
    return not any(marker in message for marker in ("cancel", "invalid api key", "authentication", "insufficient_quota"))


def route_after_supervisor(state: ResearchState) -> Literal[
    "market_analyst", "competition_analyst", "business_analyst", "direct_answer", "report_confirmation", "report_generator", "wait_user"
]:
    decision = state.get("next_agent", "FINISH")
    skill_id = state.get("selected_skill")
    if skill_id:
        try:
            validate_skill(skill_id, node=decision if decision in {
                "market_analyst", "competition_analyst", "business_analyst",
                "direct_answer", "report_generator", "followup",
            } else None)
        except ValueError:
            # A bad model/router output must never execute an unauthorized node.
            return "wait_user"
    if decision == "REPORT_CONFIRMATION" and state.get("auto_report"):
        return "report_generator"
    return {
        "market_analyst": "market_analyst",
        "competition_analyst": "competition_analyst",
        "business_analyst": "business_analyst",
        "DIRECT_ANSWER": "direct_answer",
        "REPORT_CONFIRMATION": "report_confirmation",
        "GENERATE_REPORT": "report_generator",
        "WAIT_USER": "wait_user",
    }.get(decision, "report_generator")


def build_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)
    retry_policy = RetryPolicy(max_attempts=2, initial_interval=0.5, jitter=False,
                               retry_on=_retryable_node_error) if RetryPolicy else None

    def add_retriable(name, node):
        if retry_policy is None:
            workflow.add_node(name, node)
        else:
            workflow.add_node(name, node, retry_policy=retry_policy)

    add_retriable("supervisor", supervisor_node)
    add_retriable("market_analyst", market_analyst_node)
    add_retriable("competition_analyst", competition_analyst_node)
    add_retriable("business_analyst", business_analyst_node)
    add_retriable("direct_answer", direct_answer_node)
    add_retriable("report_confirmation", report_confirmation_node)
    add_retriable("report_generator", report_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "market_analyst": "market_analyst",
            "competition_analyst": "competition_analyst",
            "business_analyst": "business_analyst",
            "direct_answer": "direct_answer",
            "report_confirmation": "report_confirmation",
            "report_generator": "report_generator",
            "wait_user": END,
        },
    )
    workflow.add_edge("market_analyst", "supervisor")
    workflow.add_edge("competition_analyst", "supervisor")
    workflow.add_edge("business_analyst", "supervisor")
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("report_confirmation", END)
    workflow.add_edge("report_generator", END)
    return workflow


def compile_graph(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer or MemorySaver())
