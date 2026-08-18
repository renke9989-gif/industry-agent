"""
Agent 节点：市场趋势分析师 (Market & Trend Analyst)

职责：
- 深挖市场规模、增长趋势、政策环境维度
- 绑定工具：联网搜索(MCP) → knowledge_upsert（自动入库）
- 搜索策略（面试可讲）：直接调用 MCP 联网搜索（Tavily 主力 + 必应兜底），
  不检索本地知识库；搜索后自动结构化入库，知识库"越用越聪明"。
"""
import os
import json
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import ResearchState
from tools.knowledge_retriever import web_search, knowledge_upsert

# ============================================================
# 工具加载：Tavily MCP（主力） + 必应 MCP（兜底） + 本地 ddgs（最后防线）
# ============================================================
# 设计说明（面试可讲）：
#   联网搜索通过 MCP 标准协议接入多个搜索源：
#   - tavily_search：Tavily 结构化 AI 搜索（主力，质量高、可直接入库）
#   - bing_search：必应中文搜索（免费兜底，无需 Key、国内可达）
#   同时保留本地 ddgs 库作为最后防线，三级降级链保证 Agent 永远有真实联网能力。
_mcp_search_tools = []
_MCP_ENABLED = False
_MCP_ERROR = ""

try:
    from mcp_servers.mcp_client import load_tavily_search_tools, load_bing_search_tools
    # 主力 Tavily + 兜底必应，一个失败不影响另一个
    _mcp_search_tools = load_tavily_search_tools() + load_bing_search_tools()
    if _mcp_search_tools:
        _MCP_ENABLED = True
        print(f"[MCP] 已接入搜索 MCP：{', '.join(t.name for t in _mcp_search_tools)}")
    else:
        _MCP_ERROR = "MCP Server 返回空工具列表"
except Exception as e:
    _MCP_ERROR = str(e)

if not _MCP_ENABLED:
    print(f"[搜索] MCP 搜索未启用（{_MCP_ERROR[:60]}），使用本地 ddgs 搜索兜底（真实联网）")

# 市场分析师工具链：联网搜索 + 知识入库。
# 注意：不暴露本地知识检索（retrieve_knowledge）给模型——按需求直接联网搜索，
# 搜索后自动入库，动态知识库仍作为"知识积累层"保留。
MARKET_TOOLS = [knowledge_upsert, web_search]
if _MCP_ENABLED:
    MARKET_TOOLS.extend(_mcp_search_tools)

# 搜索工具名集合（用于动态匹配，不硬编码具体工具名）
SEARCH_TOOL_NAMES = {"web_search", "search", "fetch_content", "tavily_search", "bing_search"}

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.3,
)

SYSTEM_PROMPT = """你是拥有15年经验的行业研究分析师，覆盖消费、科技、服务等多个赛道。
你的专业领域：市场规模测算、增长趋势、驱动因素、政策环境。

你必须遵守以下规则：
1. 需要市场数据时，直接调用 tavily_search 联网搜索获取最新真实数据
2. 联网搜索后，必须调用 knowledge_upsert 将结果结构化入库
3. 追问时要求具体数字（市场规模多少亿？年增速多少？客单价多少？）
4. 不要一次性问太多问题，每次 1-2 个精准追问
5. 如果所有能问的都问了，必须输出以下精确信号： "市场维度信息充足，可以评分"

你负责的调研维度：市场规模、增长趋势、政策环境
你的个性：严谨、数据驱动、不轻易下结论、绝不编造具体数字
"""


def _find_tool(tool_name: str):
    """在 MARKET_TOOLS 中按名字查找工具（兼容 MCP 工具和本地工具）"""
    for t in MARKET_TOOLS:
        if t.name == tool_name:
            return t
    return None


# 各搜索工具可接受的参数白名单（过滤多余字段，避免传给工具报错）
_SEARCH_ARGS_WHITELIST = {
    "tavily_search": ("query", "max_results", "search_depth"),
    "bing_search": ("query", "count", "offset"),
    "web_search": ("query", "industry"),
}


def _search_with_fallback(preferred: str, args: dict) -> str:
    """
    搜索降级链：Tavily(MCP) → 必应(MCP) → 本地 ddgs。

    先尝试 LLM 选择的工具，再按优先级逐级兜底，任一步拿到有效结果即返回。
    这一步是"鲁棒性"设计：单个搜索源挂掉不影响整个调研流程。
    """
    order = [preferred] + [n for n in ("tavily_search", "bing_search", "web_search") if n != preferred]
    errors = []
    for name in order:
        tool = _find_tool(name)
        if not tool:
            continue
        # 只传该工具认识的参数
        allowed = _SEARCH_ARGS_WHITELIST.get(name)
        call_args = {k: v for k, v in args.items() if k in allowed} if allowed else args
        if not call_args.get("query"):
            errors.append(f"{name}: 缺少 query 参数")
            continue
        try:
            result = tool.invoke(call_args)
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            if (not text or not text.strip()
                    or "未返回结果" in text or "MCP error" in text or "搜索失败" in text):
                errors.append(f"{name}: 无有效结果")
                continue
            return text
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")
            continue
    return json.dumps({"error": "所有搜索工具均不可用", "details": errors}, ensure_ascii=False)


def market_analyst_node(state: ResearchState) -> dict:
    """市场趋势分析师节点 —— LangGraph 节点函数

    流程（直接联网版）：
      ① 需要数据时，模型直接调用 tavily_search（MCP 联网搜索）
      ② 拿到真实数据 → 调用 knowledge_upsert 自动入库
      ③ 追问关键细节 / 输出"市场维度信息充足，可以评分"
    """
    industry = state.get("industry", "unknown")
    search_triggered = state.get("search_triggered", False)

    all_messages = list(state.get("messages", []))
    # 清理历史 tool 消息，避免污染
    clean_messages = []
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        clean_messages.append(msg)

    # 不检索本地知识库：直接联网搜索获取真实数据（见 SYSTEM_PROMPT 规则1）
    last_score = 0.0

    context = f"当前调研行业：{industry}。请决定下一步：是调用联网搜索获取市场数据，还是追问用户更多细节，还是信息已充足可以给出评分。"
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + clean_messages + [HumanMessage(content=context)]

    llm_with_tools = llm.bind_tools(MARKET_TOOLS)
    response = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", []) or []
    tool_results = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        args = tc.get("args", {})

        if tool_name in SEARCH_TOOL_NAMES:
            # 联网搜索：Tavily(MCP) → 必应(MCP) → 本地 ddgs 三级降级链
            if "industry" not in args or not args["industry"]:
                args["industry"] = industry
            result = _search_with_fallback(tool_name, args)
            search_triggered = True

        elif tool_name == "knowledge_upsert":
            if "industry" not in args or not args["industry"]:
                args["industry"] = industry
            result = knowledge_upsert.invoke(args)

        else:
            # 其他 MCP 工具（如 crawl_webpage 网页抓取）直接调用
            tool = _find_tool(tool_name)
            if tool:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = {"error": f"工具调用失败: {str(e)}"}
            else:
                result = {"error": f"未知工具: {tool_name}"}

        tool_results.append(
            ToolMessage(content=json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result,
                       tool_call_id=tc.get("id", ""))
        )

    if tool_results:
        reflect_messages = messages + [response] + tool_results
        final_response = llm.invoke(reflect_messages)
        return {
            "messages": [response] + tool_results + [final_response],
            "search_triggered": search_triggered,
            "last_retrieval_score": last_score,
            "analyst_opinions": {
                **state.get("analyst_opinions", {}),
                "市场分析师": final_response.content[:1000]
            }
        }

    # 即使没有工具调用也保存意见，避免 Supervisor 打分卡死、报告缺数据
    return {
        "messages": [response],
        "search_triggered": search_triggered,
        "last_retrieval_score": last_score,
        "analyst_opinions": {
            **state.get("analyst_opinions", {}),
            "市场分析师": response.content[:1000]
        },
    }
