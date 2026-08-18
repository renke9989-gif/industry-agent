"""
MCP Server: 行业数据服务

将 query_market_data 包装为标准 MCP Server，
Agent 通过 MCP 协议动态发现并调用此工具。

启动方式：
    python mcp_servers/market_data_server.py

面试话术：
    "Agent 的工具层接入了 MCP 协议。这个行业数据 Server 是示例——
     未来接入真实的行业数据库（如艾瑞、国家统计局），只需实现对应的 MCP Server，
     Agent 无需改一行代码即可获得新工具。"
"""
import json
import random
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 模拟行业数据库
MOCK_MARKET_DB = {
    "pet_economy": {
        "market_size": "约2500亿元", "annual_growth": "15.2%",
        "competitor_count": 3, "entry_barrier": "低", "profit_margin": "25-35%",
        "top_players": ["某头部连锁", "本地网红店", "电商品牌"],
    },
    "coffee": {
        "market_size": "约1300亿元", "annual_growth": "18.5%",
        "competitor_count": 5, "entry_barrier": "中", "profit_margin": "15-25%",
        "top_players": ["瑞幸", "库迪", "星巴克", "Manner", "本地精品店"],
    },
    "second_hand_luxury": {
        "market_size": "约800亿元", "annual_growth": "22.0%",
        "competitor_count": 2, "entry_barrier": "高", "profit_margin": "10-20%",
        "top_players": ["得物", "胖虎", "红布林"],
    },
}

server = Server("market-data-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """MCP 协议：声明本 Server 提供的工具列表"""
    return [
        Tool(
            name="query_market_data",
            description="查询行业市场数据（市场规模、增速、竞争格局、进入壁垒）。输入行业标识如 pet_economy、coffee。",
            inputSchema={
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "description": "行业标识，如 pet_economy / coffee / second_hand_luxury"
                    }
                },
                "required": ["industry"]
            }
        ),
        Tool(
            name="list_industries",
            description="列出所有已收录行业及其市场概况",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """MCP 协议：处理工具调用请求"""
    if name == "query_market_data":
        industry = arguments.get("industry", "")
        market = MOCK_MARKET_DB.get(industry)

        if market is None:
            result = {
                "industry": industry,
                "market_size": f"约{random.randint(100, 5000)}亿元",
                "annual_growth": f"{random.uniform(5, 30):.1f}%",
                "competitor_count": random.randint(1, 8),
                "entry_barrier": random.choice(["低", "中", "高"]),
                "profit_margin": f"{random.randint(8, 40)}-{random.randint(15, 50)}%",
                "top_players": ["待调研"],
                "status": "该行业暂无收录数据，以上为估算值，建议联网搜索核实"
            }
        else:
            result = {**market, "industry": industry, "status": "数据来源于行业报告（演示用模拟数据）"}

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    elif name == "list_industries":
        industries = []
        for ind, data in MOCK_MARKET_DB.items():
            industries.append({
                "industry": ind,
                "market_size": data["market_size"],
                "annual_growth": data["annual_growth"],
                "entry_barrier": data["entry_barrier"],
            })
        return [TextContent(type="text", text=json.dumps(industries, ensure_ascii=False))]

    return [TextContent(type="text", text=json.dumps({"error": f"未知工具: {name}"}))]


async def main():
    """启动 MCP Server（stdio 传输）"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
