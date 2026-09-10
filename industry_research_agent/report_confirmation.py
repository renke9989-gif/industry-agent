"""Human confirmation gate before generating a formal report."""
from langchain_core.messages import AIMessage


def report_confirmation_node(state: dict) -> dict:
    return {
        "messages": [AIMessage(content="当前研究已经完成。请问您还想继续提问，还是现在生成正式报告？")],
        "report_confirmation_pending": True,
        "report_confirmation_answer": "",
        "research_complete": True,
        "awaiting_user": True,
        "answer_type": "report_confirmation",
        "next_agent": "REPORT_CONFIRMATION",
        "research_stage": "report_confirmation",
    }
