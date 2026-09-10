"""Competition and risk analyst using the shared real-web evidence loop."""
from state import ResearchState

from .web_research import research_dimensions


def competition_analyst_node(state: ResearchState) -> dict:
    return research_dimensions(
        state,
        analyst_name="竞争分析师",
        dimensions=("竞争", "风险"),
        role_instruction=(
            "你是竞争与风险分析师，负责主要玩家、市场份额、集中度、进入壁垒、替代威胁和监管风险。"
            "不得使用模拟数据；对看似乐观的市场结论主动寻找反证。"
        ),
        max_tool_calls=4,
    )
