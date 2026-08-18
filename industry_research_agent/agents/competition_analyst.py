"""
Agent 节点：竞争格局分析师 (Competition & Risk Analyst)

工具加载方式：
- 优先从 MCP Server 动态加载（协议解耦，不改 Agent 代码即可扩展工具）
- Fallback: 本地硬编码 query_market_data（MCP Server 未启动时兜底）
"""
import os
import json
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import ResearchState
from tools.market_data_mock import query_market_data

# 尝试从 MCP Server 加载工具
_MCP_TOOLS = []
_MCP_ENABLED = False
try:
    from mcp_servers.mcp_client import load_mcp_tools_sync
    _mcp_server_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "mcp_servers", "market_data_server.py"
    )
    _MCP_TOOLS = load_mcp_tools_sync(_mcp_server_path)
    _MCP_ENABLED = True
    print(f"[MCP] 已从 MCP Server 加载 {len(_MCP_TOOLS)} 个工具")
except Exception as e:
    print(f"[MCP] MCP Server 未启动，使用本地 Fallback 工具 ({e})")

# 工具列表：MCP 工具优先，本地兜底
COMPETITION_TOOLS = _MCP_TOOLS if _MCP_TOOLS else [query_market_data]

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.3,
)

SYSTEM_PROMPT = """你是行业竞争分析专家，负责评估赛道竞争格局和进入风险。

你的专业领域：竞品分析、市场份额、进入壁垒、潜在风险。

你必须遵守以下规则：
1. 如果用户提到某个行业，可以调用 query_market_data 或 list_industries 查询市场数据
2. 竞争格局一定追问"当地已有几家？头部玩家是谁？你的差异化是什么？"
3. 专门找茬——任何"看起来很美"的赛道都要质疑它的坑
4. 评估维度：竞争集中度、进入壁垒、替代品威胁、政策风险
5. 信息收集完毕后，输出 "竞争维度信息充足，可以评分"

你负责的调研维度：竞争格局（集中度、竞品）、进入壁垒、风险
你的个性：犀利、喜欢唱反调、警惕性高
"""


def competition_analyst_node(state: ResearchState) -> dict:
    """竞争格局分析师节点"""
    all_messages = list(state.get("messages", []))
    clean_messages = []
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        clean_messages.append(msg)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + clean_messages

    llm_with_tools = llm.bind_tools(COMPETITION_TOOLS)
    response = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", []) or []
    tool_results = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        args = tc.get("args", {})

        matched_tool = None
        for tool in COMPETITION_TOOLS:
            if tool.name == tool_name:
                matched_tool = tool
                break

        if matched_tool:
            try:
                result = matched_tool.invoke(args)
            except Exception as e:
                result = {"error": str(e)}
        else:
            result = {"error": f"未知工具: {tool_name}"}

        tool_results.append(
            ToolMessage(content=json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result),
                       tool_call_id=tc.get("id", ""))
        )

    if tool_results:
        reflect_messages = messages + [response] + tool_results
        final_response = llm.invoke(reflect_messages)
        return {
            "messages": [response] + tool_results + [final_response],
            "analyst_opinions": {
                **state.get("analyst_opinions", {}),
                "竞争分析师": final_response.content[:1000]
            }
        }

    # 即使没有工具调用也保存意见，避免 Supervisor 打分卡死、报告缺数据
    return {
        "messages": [response],
        "analyst_opinions": {
            **state.get("analyst_opinions", {}),
            "竞争分析师": response.content[:1000]
        },
    }
