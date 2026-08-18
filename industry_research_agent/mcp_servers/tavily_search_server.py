"""
MCP Server: Tavily 联网搜索服务

真实 AI 搜索引擎，替代 DuckDuckGo 玩具级搜索。

启动方式：
    python mcp_servers/tavily_search_server.py

环境变量：
    TAVILY_API_KEY=你的Key

面试话术：
    "Agent 的联网搜索不是玩具级 DuckDuckGo，而是通过 MCP 协议接入了
     Tavily——专为 AI Agent 设计的搜索引擎，返回结构化结果。
     同时保留了本地 Fallback，Tavily 不可用时自动降级。"
"""
import os
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from tavily import TavilyClient

server = Server("tavily-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="tavily_search",
            description=(
                "主力联网搜索工具（Tavily 结构化 AI 搜索）。当本地 RAG 知识库未命中时优先使用此工具。"
                "输入搜索查询词，返回结构化搜索结果（标题、摘要、URL、相关度评分、综合答案）。"
                "搜索质量高，适合行业报告、市场规模、竞争格局、商业模式等内容检索。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词。建议包含行业名和具体主题，如'宠物烘焙 市场规模 增长率'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                        "default": 5
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "搜索深度: basic(快速) 或 advanced(深度)",
                        "enum": ["basic", "advanced"],
                        "default": "basic"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "tavily_search":
        return [TextContent(type="text", text=json.dumps({"error": f"未知工具: {name}"}))]

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return [TextContent(type="text", text=json.dumps({
            "error": "TAVILY_API_KEY 未配置，请在 .env 中设置"
        }, ensure_ascii=False))]

    try:
        client = TavilyClient(api_key=api_key)
        result = client.search(
            query=arguments.get("query", ""),
            max_results=arguments.get("max_results", 5),
            search_depth=arguments.get("search_depth", "basic"),
            include_answer=True,       # Tavily 会生成一个综合答案
            include_raw_content=False,  # 不返回原始 HTML
        )

        # 结构化输出
        output = {
            "query": result.get("query", ""),
            "answer": result.get("answer", ""),  # Tavily 综合答案
            "results": [],
            "total_results": len(result.get("results", []))
        }

        for r in result.get("results", []):
            output["results"].append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            })

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Tavily 搜索失败: {str(e)}"
        }, ensure_ascii=False))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
