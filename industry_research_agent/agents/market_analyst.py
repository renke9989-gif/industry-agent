"""Market, opportunity and trend analyst backed only by current web evidence."""
from state import ResearchState

from .web_research import research_dimensions


def market_analyst_node(state: ResearchState) -> dict:
    return research_dimensions(
        state,
        analyst_name="市场分析师",
        dimensions=("市场", "趋势", "机会"),
        role_instruction=(
            "你是市场研究分析师，负责市场规模、增速、政策、需求驱动和细分机会。"
            "优先采纳政府、协会、上市公司披露等一手来源，并区分事实和判断。"
        ),
        max_tool_calls=4,
    )
