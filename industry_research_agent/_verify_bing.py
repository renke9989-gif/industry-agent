import os
import sys

d = "D:/"
t = [x for x in os.listdir(d) if x.startswith("AI")][0]
base = os.path.join(d, t, "industry-agent", "industry_research_agent")
sys.path.insert(0, base)
os.chdir(base)

from dotenv import load_dotenv
load_dotenv()

print("=== 验证必应中文 MCP 真实搜索 ===")
from mcp_servers.mcp_client import load_bing_search_tools

try:
    tools = load_bing_search_tools()
    print("[1] MCP 工具加载成功:", [t.name for t in tools])
except Exception as e:
    print("[失败] MCP 加载:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 找到 bing_search 工具
search_tool = next((t for t in tools if "search" in t.name.lower() and "crawl" not in t.name.lower()), None)
if not search_tool:
    print("[失败] 未找到搜索工具")
    sys.exit(1)

print(f"[2] 调用 {search_tool.name} 搜索 '宠物烘焙 市场规模' ...")
try:
    result = search_tool.invoke({"query": "宠物烘焙 市场规模 增长率"})
    print("[3] 成功！搜索结果:")
    print(result[:1000])
except Exception as e:
    print("[失败] 搜索报错:", type(e).__name__, str(e)[:200])
    import traceback
    traceback.print_exc()
