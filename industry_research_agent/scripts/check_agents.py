"""
Agent 一致性检查：验证所有分析师的 System Prompt + Tool 绑定 + 维度映射
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

EXPECTED = {
    "market_analyst": {
        "tools": ["retrieve_knowledge", "web_search", "knowledge_upsert"],
        "dimensions": ["市场", "机会", "趋势"],
        "keywords": ["retrieve_knowledge", "web_search", "严谨"],
    },
    "business_analyst": {
        "tools": ["calculate_roi"],
        "dimensions": ["商业模式"],
        "keywords": ["calculate_roi", "ROI", "理性"],
    },
    "competition_analyst": {
        "tools": ["query_market_data", "list_industries"],
        "dimensions": ["竞争", "风险"],
        "keywords": ["query_market_data", "壁垒", "竞品"],
    },
}


def check_agent(name, config):
    """检查单个 Agent 的配置"""
    errors = []

    agent_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "agents", f"{name}.py"
    )
    if not os.path.exists(agent_file):
        return [f"❌ {name}: 文件不存在"]

    with open(agent_file, "r", encoding="utf-8") as f:
        code = f.read()

    for kw in config["keywords"]:
        if kw not in code:
            errors.append(f"⚠️  {name}: System Prompt 缺少关键词 '{kw}'")

    for tool_name in config["tools"]:
        if tool_name not in code:
            errors.append(f"⚠️  {name}: 未绑定工具 '{tool_name}'")

    if not errors:
        return [f"✅ {name}: 通过"]
    return errors


def main():
    print("🔍 Agent 一致性检查\n")

    all_ok = True
    for agent_name, config in EXPECTED.items():
        results = check_agent(agent_name, config)
        for r in results:
            print(f"  {r}")
            if not r.startswith("✅"):
                all_ok = False

    # 检查 Supervisor 路由维度完整性
    supervisor_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "agents", "supervisor.py"
    )
    if os.path.exists(supervisor_file):
        with open(supervisor_file, "r", encoding="utf-8") as f:
            code = f.read()
        all_dims = set()
        for config in EXPECTED.values():
            all_dims.update(config["dimensions"])
        for dim in all_dims:
            if dim not in code:
                print(f"  ⚠️  supervisor: 缺少维度 '{dim}' 的路由逻辑")
                all_ok = False
        if all(dim in code for dim in all_dims):
            print(f"  ✅ supervisor: 所有维度路由完整")

    print(f"\n{'✅ 全部通过' if all_ok else '⚠️  存在问题，请检查'}")


if __name__ == "__main__":
    main()
