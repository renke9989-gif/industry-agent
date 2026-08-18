"""
Agent 节点：商业模式分析师 (Business Model Analyst)
"""
import os
import json
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import ResearchState
from tools.roi_calculator import calculate_roi

BUSINESS_TOOLS = [calculate_roi]

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.2,
)

SYSTEM_PROMPT = """你是商业模式与财务测算专家，负责评估赛道的赚钱逻辑，极其理性。

你必须遵守以下规则：
1. 当市场或竞争分析师提出进入某赛道的建议时，必须调用 calculate_roi 测算投入产出
2. ROI < 1.5 → 直接反对并给出理由
3. ROI 1.0-1.5 → 谨慎建议，列出风险
4. ROI ≥ 1.5 → 可以推荐，但也要指出投入风险
5. 追问商业模式相关数据：客单价？毛利率？获客成本？回本周期？

你负责的调研维度：商业模式（盈利模式、成本结构、客单价）、投资回报
你的个性：理性、抠细节、只相信数字、绝不做亏本买卖
"""


def business_analyst_node(state: ResearchState) -> dict:
    """商业模式分析师节点"""
    all_messages = list(state.get("messages", []))
    clean_messages = []
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        clean_messages.append(msg)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + clean_messages

    llm_with_tools = llm.bind_tools(BUSINESS_TOOLS)
    response = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", []) or []
    tool_results = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        args = tc.get("args", {})
        if tool_name == "calculate_roi":
            result = calculate_roi.invoke(args)
        else:
            result = {"error": f"未知工具: {tool_name}"}
        tool_results.append(
            ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc.get("id", ""))
        )

    if tool_results:
        reflect_messages = messages + [response] + tool_results
        final_response = llm.invoke(reflect_messages)
        return {
            "messages": [response] + tool_results + [final_response],
            "analyst_opinions": {
                **state.get("analyst_opinions", {}),
                "商业模式分析师": final_response.content[:1000]
            }
        }

    # 即使没有工具调用也保存意见，避免 Supervisor 打分卡死、报告缺数据
    return {
        "messages": [response],
        "analyst_opinions": {
            **state.get("analyst_opinions", {}),
            "商业模式分析师": response.content[:1000]
        },
    }
