"""
agent_tools.py —— 行业调研 Agent 的开发辅助工具函数

快捷调用：
  python scripts/agent_tools.py check     → 环境检查
  python scripts/agent_tools.py eval      → 跑 Eval Harness
  python scripts/agent_tools.py demo      → 一行跑预设对话
  python scripts/agent_tools.py agents    → Agent 一致性检查
"""
import os
import sys
import importlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_environment() -> dict:
    """检查工程健康状态"""
    result = {"python": False, "deps": {}, "env": {}, "files": {}, "all_ok": False}
    result["python"] = sys.version_info >= (3, 9)

    deps = {
        "langgraph": "langgraph", "langchain": "langchain",
        "langchain_openai": "langchain_openai", "chromadb": "chromadb",
        "sentence_transformers": "sentence_transformers",
        "dotenv": "dotenv", "openai": "openai",
    }
    for name, module in deps.items():
        try:
            importlib.import_module(module)
            result["deps"][name] = True
        except ImportError:
            result["deps"][name] = False

    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    for var in ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"]:
        val = os.getenv(var, "")
        result["env"][var] = bool(val and not val.startswith("sk-your-") and not val.startswith("your-"))

    expected_files = ["state.py", "graph.py", "main.py", "report_generator.py", "requirements.txt", ".env"]
    expected_dirs = ["agents", "tools", "knowledge_packs", "eval", "mcp_servers"]
    for f in expected_files:
        result["files"][f] = os.path.exists(os.path.join(PROJECT_ROOT, f))
    for d in expected_dirs:
        result["files"][d + "/"] = os.path.isdir(os.path.join(PROJECT_ROOT, d))

    result["all_ok"] = (
        result["python"] and all(result["deps"].values())
        and all(result["env"].values()) and all(result["files"].values())
    )
    return result


def run_eval_harness() -> dict:
    """跑 Eval Harness"""
    os.chdir(PROJECT_ROOT)
    from dotenv import load_dotenv; load_dotenv()
    from tools.knowledge_retriever import load_seed_knowledge
    load_seed_knowledge()
    from graph import compile_graph
    agent = compile_graph()
    from eval.eval_harness import load_cases, evaluate

    cases = load_cases(os.path.join(PROJECT_ROOT, "eval", "eval_cases.json"))
    results = [evaluate(agent, c) for c in cases]
    valid = [r for r in results if "error" not in r and r.get("scores")]

    if not valid:
        return {"error": "所有测试用例失败", "total_cases": len(results)}

    route_accs = [r["scores"].get("route_acc", 0) for r in valid if "route_acc" in r["scores"]]
    hallu = [r["scores"].get("hallucination", 0) for r in valid if "hallucination" in r["scores"]]
    turns = [r.get("turns", 0) for r in valid]

    return {
        "route_accuracy": sum(route_accs) / len(route_accs) if route_accs else 0,
        "hallucination_rate": 1 - (sum(hallu) / len(hallu)) if hallu else 0,
        "avg_turns": sum(turns) / len(turns) if turns else 0,
        "total_cases": len(results), "valid_cases": len(valid),
    }


def run_demo_diagnosis(test_input: str = None) -> dict:
    """跑一次完整调研"""
    import json
    if test_input is None:
        with open(os.path.join(PROJECT_ROOT, "eval", "eval_cases.json"), "r", encoding="utf-8") as f:
            test_input = json.load(f)[0]["input"]

    os.chdir(PROJECT_ROOT)
    from dotenv import load_dotenv; load_dotenv()
    from langchain_core.messages import HumanMessage
    from graph import compile_graph
    from tools.knowledge_retriever import load_seed_knowledge

    load_seed_knowledge()
    agent = compile_graph()
    initial_state = {
        "messages": [HumanMessage(content=test_input)],
        "scores": {"市场": 0, "竞争": 0, "商业模式": 0, "机会": 0, "风险": 0, "趋势": 0},
        "industry": "unknown", "region": "未知", "budget": "未知",
        "analyst_opinions": {}, "analyst_questions": {},
        "next_agent": "", "research_complete": False,
        "turn_count": 0, "supervisor_reason": "", "conflicts": [],
        "knowledge_cache": [], "search_triggered": False,
        "last_retrieval_score": 0.0, "error_count": 0, "force_finish": False,
    }

    final_state = {}
    for event in agent.stream(initial_state, {"configurable": {"thread_id": "demo"}}):
        for node_output in event.values():
            if node_output.get("research_complete"):
                final_state = node_output
            if "report_generator" in str(event):
                for msg in node_output.get("messages", []):
                    final_state["report"] = getattr(msg, "content", "")

    return {
        "input": test_input,
        "turns": final_state.get("turn_count", 0),
        "research_complete": final_state.get("research_complete", False),
        "industry": final_state.get("industry", "unknown"),
        "scores": final_state.get("scores", {}),
        "search_triggered": final_state.get("search_triggered", False),
        "report_length": len(final_state.get("report", "")),
    }


def check_agent_consistency() -> dict:
    """检查 Agent Prompt + Tool 绑定"""
    import subprocess
    script = os.path.join(PROJECT_ROOT, "scripts", "check_agents.py")
    if not os.path.exists(script):
        return {"error": "scripts/check_agents.py 不存在"}
    r = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=PROJECT_ROOT)
    return {"stdout": r.stdout, "all_ok": r.returncode == 0}


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import json
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"

    if cmd == "check":
        print(json.dumps(check_environment(), indent=2, ensure_ascii=False))
    elif cmd == "eval":
        print(json.dumps(run_eval_harness(), indent=2, ensure_ascii=False))
    elif cmd == "demo":
        inp = sys.argv[2] if len(sys.argv) > 2 else None
        print(json.dumps(run_demo_diagnosis(inp), indent=2, ensure_ascii=False))
    elif cmd == "agents":
        print(json.dumps(check_agent_consistency(), indent=2, ensure_ascii=False))
    else:
        print(f"用法: python agent_tools.py [check|eval|demo|agents]")
