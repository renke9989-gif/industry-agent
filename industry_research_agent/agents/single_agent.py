"""Single-agent baseline used by the ablation experiment."""
from state import ResearchState
from .web_research import research_dimensions


def single_research_node(state: ResearchState) -> dict:
    return research_dimensions(
        state,
        analyst_name="单一研究分析师",
        dimensions=("市场", "竞争", "商业模式", "机会", "风险", "趋势"),
        role_instruction=(
            "你是单一行业研究分析师，需要覆盖市场、竞争、商业模式、机会、风险和趋势。"
            "只依据证据，不得补造数字，并为关键事实添加 [E#]。"
        ),
        max_tool_calls=8,
    )
