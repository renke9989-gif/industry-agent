"""
MCP Client: Agent 启动时自动连接 MCP Server，动态加载工具

面试话术：
    "Agent 启动时通过 MCP Client 自动发现并注册外部工具。
    搜索工具不硬编码在 Agent 里，而是通过 MCP 标准协议从 DuckDuckGo MCP Server 动态获取。
    MCP（Model Context Protocol）是 Anthropic 提出的开放标准，
    让 AI 应用以统一协议接入外部工具和数据源。
    关键设计：MCP 连接是常驻的（贯穿整个 Agent 生命周期），
    工具加载后连接不关闭，否则调用工具时会报 ClosedResourceError。"

支持两种 MCP Server 启动方式：
1. 本地 Python 脚本（自研 MCP Server，如 market_data_server.py）
2. 第三方 PyPI 包（如 duckduckgo-mcp-server，通过 python -m 启动）

核心设计：
- 常驻连接：stdio 子进程在 Agent 运行期间一直存活
- 工具调用时复用同一个 ClientSession，避免连接关闭
"""
import asyncio
import os
import sys
import threading
from typing import List, Optional, Dict
from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# MCP 工具的通用 JSON Schema：允许任意键值对。
#
# 为什么必须是 dict 而不是 pydantic 模型：
# - langchain 根据 `**kwargs` 函数签名推断的 args_schema 是 {"kwargs": {...}} 嵌套结构，
#   导致 invoke 时实际参数被丢弃（MCP Server 收到空 arguments，报 'Required at ...'）。
# - 空 pydantic 模型 + extra='allow' 也不行：_to_args_and_kwargs 会把"无声明字段的模型"
#   当成无参工具，同样丢弃参数。
# - 传入 dict 类型 args_schema 时，_parse_input 原样返回输入 dict，_to_args_and_kwargs
#   再作为 kwargs 透传给 MCP Server——这才是正确路径。
_MCP_CATCH_ALL_SCHEMA: dict = {"type": "object", "additionalProperties": True}


class McpToolManager:
    """
    MCP 工具管理器：维护常驻连接，管理工具生命周期。

    关键点：MCP 连接在 Agent 运行期间保持常驻，
    工具调用复用同一个 session，避免 ClosedResourceError。
    """

    def __init__(self):
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._loop = None
        self._tools: List[BaseTool] = []
        self._lock = threading.Lock()
        # 保持对 stdio_client / ClientSession 上下文管理器的引用。
        # stdio_client 是 @asynccontextmanager 生成器，若被 GC 会触发 finally
        # 关闭所有流，导致后续 call_tool 报 ClosedResourceError。
        self._stdio_ctx = None
        self._session_ctx = None

    def connect(
        self,
        server_script_path: Optional[str] = None,
        module_name: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[dict] = None,
    ):
        """
        启动 MCP Server 子进程并建立常驻连接（阻塞，直到工具加载完成）。

        必须在 Agent 运行前调用一次，之后连接保持常驻。

        三种启动方式：
        1. Python 模块：module_name="xxx.server" → python -m xxx.server
        2. Python 脚本：server_script_path="xxx.py" → python xxx.py
        3. 任意命令：command="npx", args=["-y","bing-cn-mcp"] → npx -y bing-cn-mcp
        """
        python_path = os.getenv("MCP_PYTHON_PATH", "python")

        if command:
            # 通用命令启动（如 npx）
            final_command = command
            final_args = args or []
        elif module_name:
            final_command = python_path
            final_args = ["-m", module_name]
        elif server_script_path:
            final_command = python_path
            final_args = [server_script_path]
        else:
            raise ValueError("必须提供 command / module_name / server_script_path 之一")

        server_params = StdioServerParameters(command=final_command, args=final_args, env=env or {})

        # 在独立线程中运行 asyncio 事件循环，保持连接常驻
        self._loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._connect_async(server_params))
            # 关键修复：连接建立后事件循环必须保持常驻（run_forever），
            # 否则后续通过 run_coroutine_threadsafe 调度的工具调用永远不会被执行，
            # 表现为"连接成功但调用超时（20s TimeoutError）"。
            self._loop.run_forever()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=30)  # 等待连接建立（最多 30 秒）

        if not self._session:
            raise RuntimeError("MCP 连接建立超时")

    async def _connect_async(self, server_params):
        """建立 stdio 连接并加载工具（在事件循环中运行）"""
        stdio_ctx = stdio_client(server_params)
        self._stdio_ctx = stdio_ctx  # 持有引用，防止 GC 关闭底层流
        self._read, self._write = await stdio_ctx.__aenter__()

        session_ctx = ClientSession(self._read, self._write)
        self._session_ctx = session_ctx
        self._session = await session_ctx.__aenter__()
        await self._session.initialize()

        # 加载工具列表
        mcp_tools = await self._session.list_tools()

        for tool in mcp_tools.tools:
            langchain_tool = self._wrap_tool(tool.name, tool.description or f"MCP tool: {tool.name}")
            self._tools.append(langchain_tool)

    def _wrap_tool(self, tool_name: str, description: str) -> BaseTool:
        """将 MCP 工具包装为 LangChain StructuredTool，复用常驻 session。
        同时提供 sync 和 async 两个入口，兼容 .invoke() 和 .ainvoke()。
        """

        def _make_sync_call(tool_name: str):
            def _call_tool_sync(**kwargs):
                # 同步入口：跨线程调度到常驻事件循环执行
                future = asyncio.run_coroutine_threadsafe(
                    self._call_tool_async(tool_name, kwargs),
                    self._loop
                )
                return future.result(timeout=20)
            return _call_tool_sync

        def _make_async_call(tool_name: str):
            async def _call_tool_async_wrapper(**kwargs):
                # 异步入口：直接跨线程调度（当前线程可能不在常驻 loop 中）
                future = asyncio.run_coroutine_threadsafe(
                    self._call_tool_async(tool_name, kwargs),
                    self._loop
                )
                return await asyncio.wrap_future(future)
            return _call_tool_async_wrapper

        return StructuredTool.from_function(
            name=tool_name,
            description=description,
            func=_make_sync_call(tool_name),
            coroutine=_make_async_call(tool_name),
            args_schema=_MCP_CATCH_ALL_SCHEMA,
        )

    async def _call_tool_async(self, tool_name: str, arguments: dict) -> str:
        """在事件循环内调用 MCP 工具"""
        result = await self._session.call_tool(tool_name, arguments=arguments)
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
        return "\n".join(texts) if texts else str(result.content)

    def get_tools(self) -> List[BaseTool]:
        return self._tools


# ============================================================
# 全局单例：常驻 MCP 连接
# ============================================================

_ddg_manager: Optional[McpToolManager] = None
_ddg_error: str = ""


def get_ddg_search_manager() -> McpToolManager:
    """
    获取 DuckDuckGo 搜索 MCP 管理器（单例，常驻连接）。
    首次调用时建立连接，之后复用。
    """
    global _ddg_manager, _ddg_error

    if _ddg_manager is not None:
        return _ddg_manager

    manager = McpToolManager()
    # 环境变量：包括代理设置（httpx 默认不读系统代理，需显式传入）
    env = {
        "DDG_REGION": os.getenv("DDG_REGION", "cn-zh"),
        "DDG_SAFE_SEARCH": os.getenv("DDG_SAFE_SEARCH", "MODERATE"),
    }
    # 透传代理：httpx 不自动读系统代理，必须显式传入 HTTP_PROXY/HTTPS_PROXY
    # 这样 MCP Server 内部的 httpx 才能走用户的代理（如 Clash 127.0.0.1:7897）
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        val = os.environ.get(proxy_var)
        if val:
            env[proxy_var] = val

    # 如果环境变量里没有，尝试从系统代理探测（Windows 注册表代理）
    if not any(k in env for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]):
        try:
            import urllib.request
            sys_proxies = urllib.request.getproxies()
            if sys_proxies.get("http"):
                env["HTTP_PROXY"] = sys_proxies["http"]
            if sys_proxies.get("https"):
                env["HTTPS_PROXY"] = sys_proxies["https"]
        except Exception:
            pass

    try:
        manager.connect(module_name="duckduckgo_mcp_server.server", env=env)
        _ddg_manager = manager
    except Exception as e:
        _ddg_error = str(e)
        raise

    return _ddg_manager


def load_ddg_search_tools() -> List[BaseTool]:
    """
    加载 DuckDuckGo 搜索 MCP 工具（真实联网，免费无需 Key）。
    返回 LangChain 兼容工具列表，连接保持常驻。
    """
    manager = get_ddg_search_manager()
    return manager.get_tools()


# ============================================================
# 必应中文搜索 MCP（推荐主力：国内可达、无需代理、中文优化）
# ============================================================

_bing_manager: Optional[McpToolManager] = None
_bing_error: str = ""


def get_bing_search_manager() -> McpToolManager:
    """
    获取必应中文搜索 MCP 管理器（单例，常驻连接）。

    通过 npx 启动 bing-cn-mcp（Node.js 实现），走 cn.bing.com，
    国内直连可达、无需代理、无需 API Key、中文搜索质量高。
    """
    global _bing_manager, _bing_error

    if _bing_manager is not None:
        return _bing_manager

    manager = McpToolManager()
    try:
        # npx -y bing-cn-mcp：自动下载并启动（-y 表示无需确认）
        manager.connect(command="npx", args=["-y", "bing-cn-mcp"])
        _bing_manager = manager
    except Exception as e:
        _bing_error = str(e)
        raise

    return _bing_manager


def load_bing_search_tools() -> List[BaseTool]:
    """
    加载必应中文搜索 MCP 工具（真实联网、免费、无需 Key、国内可达）。
    返回 LangChain 兼容工具列表，连接保持常驻。
    """
    manager = get_bing_search_manager()
    return manager.get_tools()


# ============================================================
# Tavily 联网搜索 MCP（主力：结构化 AI 搜索，需 TAVILY_API_KEY）
# ============================================================

_tavily_manager: Optional[McpToolManager] = None
_tavily_error: str = ""


def _ensure_env_loaded():
    """确保 TAVILY_API_KEY 已加载（兼容直接 import 未跑 load_dotenv 的场景）"""
    if os.getenv("TAVILY_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
        project_root = os.path.dirname(os.path.dirname(__file__))
        load_dotenv(os.path.join(project_root, ".env"))
    except Exception:
        pass


def get_tavily_search_manager() -> McpToolManager:
    """
    获取 Tavily 搜索 MCP 管理器（单例，常驻连接）。

    通过本项目的 tavily_search_server.py 启动本地 MCP Server。
    用当前解释器（sys.executable）启动，确保子进程能 import tavily + mcp——
    不依赖 .env 里 MCP_PYTHON_PATH 指向的那个 Python。

    必须显式把 TAVILY_API_KEY（及代理）传给子进程：mcp SDK 的 stdio_client
    默认只继承少量安全环境变量，不会带上自定义 Key。
    """
    global _tavily_manager, _tavily_error

    if _tavily_manager is not None:
        return _tavily_manager

    _ensure_env_loaded()

    manager = McpToolManager()
    tavily_server_path = os.path.join(os.path.dirname(__file__), "tavily_search_server.py")
    # 子进程需要 TAVILY_API_KEY + 代理设置（Tavily API 走代理更稳）
    env = {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")}
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        val = os.environ.get(proxy_var)
        if val:
            env[proxy_var] = val
    try:
        manager.connect(command=sys.executable, args=[tavily_server_path], env=env)
        _tavily_manager = manager
    except Exception as e:
        _tavily_error = str(e)
        raise

    return _tavily_manager


def load_tavily_search_tools() -> List[BaseTool]:
    """
    加载 Tavily 搜索 MCP 工具（主力联网搜索，结构化结果，需 TAVILY_API_KEY）。
    返回 LangChain 兼容工具列表，连接保持常驻。Tavily 不可用时由上层降级到其他搜索源。
    """
    manager = get_tavily_search_manager()
    return manager.get_tools()


# ============================================================
# 兼容旧接口：同步加载自研 MCP Server（一次性连接，用完即关）
# ============================================================

async def load_mcp_tools(
    server_script_path: Optional[str] = None,
    module_name: Optional[str] = None,
    env: Optional[dict] = None,
) -> List[BaseTool]:
    """
    连接一个 MCP Server 并加载其提供的所有工具（一次性连接）。
    注意：此函数返回后连接会关闭，仅适合"加载工具后立即在同一个 async 上下文内使用"的场景。
    对于需要长期复用的工具，请使用 McpToolManager（常驻连接）。
    """
    python_path = os.getenv("MCP_PYTHON_PATH", "python")

    if module_name:
        args = ["-m", module_name]
    elif server_script_path:
        args = [server_script_path]
    else:
        raise ValueError("必须提供 server_script_path 或 module_name 之一")

    server_params = StdioServerParameters(command=python_path, args=args, env=env or {})
    tools = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            for tool in mcp_tools.tools:
                def _make_call_tool(tool_name: str):
                    async def _call_tool(**kwargs):
                        result = await session.call_tool(tool_name, arguments=kwargs)
                        texts = []
                        for content in result.content:
                            if hasattr(content, "text"):
                                texts.append(content.text)
                        return "\n".join(texts) if texts else str(result.content)
                    return _call_tool

                langchain_tool = StructuredTool.from_function(
                    name=tool.name,
                    description=tool.description or f"MCP tool: {tool.name}",
                    coroutine=_make_call_tool(tool.name),
                    args_schema=_MCP_CATCH_ALL_SCHEMA,
                )
                tools.append(langchain_tool)

    return tools


def load_mcp_tools_sync(
    server_script_path: Optional[str] = None,
    module_name: Optional[str] = None,
    env: Optional[dict] = None,
) -> List[BaseTool]:
    """同步包装器"""
    return asyncio.run(load_mcp_tools(server_script_path, module_name, env))
