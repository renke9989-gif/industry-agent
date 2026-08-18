"""
网页版聊天助手后端 —— 基于 Starlette + SSE 流式输出

启动：
    python web_server.py
    或双击 scripts/run_web.bat

访问：
    http://localhost:8000

架构：
    - GET  /            返回前端页面 (web_ui/index.html)
    - POST /chat        接收用户消息，SSE 流式返回 Agent 每一步
    - GET  /reset       重置会话
"""
import os
import sys
import json
import asyncio
import uuid

# 强制 UTF-8，避免 Windows 控制台 emoji 崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from state import ResearchState
from graph import compile_graph
from tools.knowledge_retriever import load_seed_knowledge

# ============================================================
# 全局 Agent 实例（启动时初始化一次）
# ============================================================
AGENT = None
SEED_COUNT = 0


def init_agent():
    global AGENT, SEED_COUNT
    if AGENT is None:
        SEED_COUNT = load_seed_knowledge()
        AGENT = compile_graph()
    return AGENT, SEED_COUNT


# ============================================================
# 角色 → 前端展示信息映射
# ============================================================
ROLE_INFO = {
    "supervisor": {"name": "协调监督者", "icon": "🤖", "color": "#6366f1"},
    "market_analyst": {"name": "市场趋势分析师", "icon": "📈", "color": "#0ea5e9"},
    "business_analyst": {"name": "商业模式分析师", "icon": "💰", "color": "#f59e0b"},
    "competition_analyst": {"name": "竞争格局分析师", "icon": "⚔️", "color": "#ef4444"},
    "conflict_resolver": {"name": "冲突裁决", "icon": "⚠️", "color": "#f97316"},
    "report_generator": {"name": "调研报告", "icon": "📄", "color": "#10b981"},
}


def build_initial_state(user_input: str) -> ResearchState:
    return {
        "messages": [HumanMessage(content=user_input)],
        "scores": {"市场": 0, "竞争": 0, "商业模式": 0, "机会": 0, "风险": 0, "趋势": 0},
        "industry": "unknown",
        "region": "未知",
        "budget": "未知",
        "analyst_opinions": {},
        "analyst_questions": {},
        "next_agent": "",
        "research_complete": False,
        "turn_count": 0,
        "supervisor_reason": "",
        "conflicts": [],
        "knowledge_cache": [],
        "search_triggered": False,
        "last_retrieval_score": 0.0,
        "error_count": 0,
        "force_finish": False,
    }


def _is_thinking_content(content: str) -> bool:
    """
    判断一段 LLM 输出是否为"思考过程"而非最终答复。

    DeepSeek 等模型在反射阶段有时会把工具调用"写"成纯文本
    （如 "<tool_calls>...<invoke name=web_search>..."），
    这类内容应折叠到"思考过程"中，而不是直接展示给用户。
    """
    if not content:
        return False

    # 工具调用标记（伪 function calling 痕迹）
    tool_markers = [
        "<tool_calls>", "<tool_call>", "</tool_calls>", "</tool_call>",
        "<invoke", "</invoke>", "<parameter", "</parameter>",
        "<web_search>", "</web_search>", "<retrieve_knowledge>",
        "<knowledge_upsert>", "<query>", "</query>",
    ]
    if any(marker in content for marker in tool_markers):
        return True

    # 中间推理/过程性文字特征
    thinking_markers = [
        "本地知识库相似度不足",
        "我需要联网搜索",
        "我需要调用",
        "我先检索",
        "检索结果如下",
        "正在检索",
        "正在搜索",
    ]
    if any(marker in content for marker in thinking_markers):
        return True

    return False


# ============================================================
# SSE 事件生成器：把 Agent 每一步实时推给前端
# ============================================================
async def event_generator(user_input: str):
    """生成 SSE 事件流"""
    agent, seed_count = init_agent()

    thread_id = str(uuid.uuid4())
    initial_state = build_initial_state(user_input)
    config = {"configurable": {"thread_id": thread_id}}

    # 事件 1：初始化信息
    yield {
        "event": "init",
        "data": json.dumps({
            "type": "init",
            "seed_count": seed_count,
            "thread_id": thread_id,
        }, ensure_ascii=False),
    }

    try:
        for event in agent.stream(initial_state, config):
            for node_name, node_output in event.items():
                info = ROLE_INFO.get(node_name, {"name": node_name, "icon": "📌", "color": "#64748b"})

                messages = node_output.get("messages", [])
                scores = node_output.get("scores", {})
                next_agent = node_output.get("next_agent", "")
                supervisor_reason = node_output.get("supervisor_reason", "")
                search_triggered = node_output.get("search_triggered", False)

                # 事件：当前发言的角色（专家切换展示）
                yield {
                    "event": "agent",
                    "data": json.dumps({
                        "type": "agent_active",
                        "node": node_name,
                        "name": info["name"],
                        "icon": info["icon"],
                        "color": info["color"],
                    }, ensure_ascii=False),
                }

                # 事件：进度条更新
                if scores:
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "type": "progress",
                            "scores": scores,
                        }, ensure_ascii=False),
                    }

                # 事件：Supervisor 决策
                if supervisor_reason:
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "supervisor",
                            "node": node_name,
                            "icon": info["icon"],
                            "name": info["name"],
                            "color": info["color"],
                            "content": supervisor_reason,
                            "decision": next_agent,
                        }, ensure_ascii=False),
                    }

                # 事件：消息内容
                for msg in messages:
                    content = getattr(msg, "content", "")
                    if not content:
                        continue

                    # 跳过工具调用消息和纯工具消息（不展示给用户）
                    if isinstance(msg, ToolMessage):
                        continue
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        continue

                    # Supervisor 的决策消息单独处理，避免重复
                    if content.startswith("[Supervisor 决策]"):
                        continue

                    # 判断是否为"思考过程"（工具调用痕迹 / 中间推理文字）
                    is_thinking = _is_thinking_content(content)

                    msg_type = "report" if node_name == "report_generator" else "analyst"
                    if is_thinking:
                        msg_type = "thinking"

                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": msg_type,
                            "node": node_name,
                            "icon": info["icon"],
                            "name": info["name"],
                            "color": info["color"],
                            "content": content,
                            "search_triggered": search_triggered,
                        }, ensure_ascii=False),
                    }

                # 报告完成后发送完成事件
                if node_output.get("research_complete"):
                    final_scores = node_output.get("scores", {})
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "type": "done",
                            "industry": node_output.get("industry", "unknown"),
                            "scores": final_scores,
                            "turn_count": node_output.get("turn_count", 0),
                        }, ensure_ascii=False),
                    }
                    return

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {
            "event": "error",
            "data": json.dumps({
                "type": "error",
                "message": str(e),
            }, ensure_ascii=False),
        }


# ============================================================
# 路由处理
# ============================================================

async def index(request: Request):
    """返回前端页面"""
    ui_dir = os.path.join(os.path.dirname(__file__), "web_ui")
    html_path = os.path.join(ui_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        return Response(html, media_type="text/html")
    return Response("<h1>前端页面缺失，请检查 web_ui/index.html</h1>", media_type="text/html")


async def chat(request: Request):
    """SSE 聊天接口"""
    body = await request.json()
    user_input = body.get("message", "").strip()
    if not user_input:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    return EventSourceResponse(event_generator(user_input))


async def reset(request: Request):
    """重置会话"""
    return JSONResponse({"status": "ok", "message": "会话已重置"})


async def health(request: Request):
    """健康检查"""
    return JSONResponse({"status": "ok", "service": "industry-research-agent"})


# ============================================================
# 应用入口
# ============================================================
app = Starlette(
    debug=False,
    routes=[
        Route("/", index),
        Route("/chat", chat, methods=["POST"]),
        Route("/reset", reset, methods=["GET"]),
        Route("/health", health),
    ],
)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🔍 行业调研智能体 · 网页版")
    print("=" * 60)
    print("\n初始化 Agent（加载知识库 + 编译图）...")
    agent, seed_count = init_agent()
    print(f"知识库就绪：{seed_count} 条种子知识")
    print("\n✅ 启动成功！请在浏览器打开：")
    print("   http://localhost:8000")
    print("\n按 Ctrl+C 停止服务\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
