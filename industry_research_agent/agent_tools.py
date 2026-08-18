"""
agent_tools.py —— 行业调研 Agent 的开发辅助工具函数

四个 slash command 共用：
- check_environment()      → /diagnose-check
- run_eval_harness()       → /diagnose-eval
- run_demo_diagnosis()     → /diagnose-demo
- check_agent_consistency() → /diagnose-check (第5步)
"""
import os
import sys
import importlib


# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 环境检查
# ============================================================

def check_environment() -> dict:
    """
    检查工程健康状态：Python 版本、依赖、.env、文件结构

    Returns:
        {"python": bool, "deps": dict, "env": dict, "files": dict, "all_ok": bool}
    """
    result = {"python": False, "deps": {}, "env": {}, "files": {}, "all_ok": False}

    py_version = sys.version_info
    result["python"] = (py_version.major, py_version.minor) >= (3, 9)

    deps = {
        "langgraph": "langgraph",
        "langchain": "langchain",
        "langchain_openai": "langchain_openai",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence_transformers",
        "dotenv": "dotenv",
        "openai": "openai",
    }
    for name, module in deps.items():
        try:
            importlib.import_module(module)
            result["deps"][name] = True
        except ImportError:
            result["deps"][name] = False

    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    required_env = ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"]
    for var in required_env:
        val = os.getenv(var, "")
        result["env"][var] = bool(val and not val.startswith("sk-your-") and not val.startswith("your-"))

    expected_files = [
        "state.py", "graph.py", "main.py", "report_generator.py",
        "requirements.txt", ".env",
    ]
    expected_dirs = ["agents", "tools", "knowledge_packs", "eval", "mcp_servers"]
    for f in expected_files:
        result["files"][f] = os.path.exists(os.path.join(PROJECT_ROOT, f))
    for d in expected_dirs:
        result["files"][d + "/"] = os.path.isdir(os.path.join(PROJECT_ROOT, d))

    result["all_ok"] = (
        result["python"]
        and all(result["deps"].values())
        and all(result["env"].values())
        and all(result["files"].values())
    )
    return result


# ============================================================
# 2. Eval Harness
# ============================================================

def run_eval_harness() -> dict:
    """
    跑 Eval Harness，返回评估指标
    """
    os.chdir(PROJECT_ROOT)
    from dotenv import load_dotenv
    load_dotenv()

    from tools.knowledge_retriever import load_seed_knowledge
    load_seed_knowledge()

    from graph import compile_graph
    agent = compile_graph()

    from eval.eval_harness import load_cases, evaluate
    import json

    cases_path = os.path.join(PROJECT_ROOT, "eval", "eval_cases.json")
    cases = load_cases(cases_path)

    results = []
    for case in cases:
        result = evaluate(agent, case)
        results.append(result)

    valid = [r for r in results if "error" not in r and r.get("scores")]
    if not valid:
        return {"error": "所有测试用例失败", "total_cases": len(results)}

    route_accs = [r["scores"].get("route_acc", 0) for r in valid if "route_acc" in r["scores"]]
    hallu_rates = [r["scores"].get("hallucination", 0) for r in valid if "hallucination" in r["scores"]]
    turns = [r.get("turns", 0) for r in valid]

    return {
        "route_accuracy": sum(route_accs) / len(route_accs) if route_accs else 0,
        "hallucination_rate": 1 - (sum(hallu_rates) / len(hallu_rates)) if hallu_rates else 0,
        "avg_turns": sum(turns) / len(turns) if turns else 0,
        "total_cases": len(results),
        "valid_cases": len(valid),
        "details": [{"id": r["case_id"], "name": r.get("name", ""), "scores": r.get("scores", {})} for r in results],
    }


# ============================================================
# 3. Demo 调研
# ============================================================

def run_demo_diagnosis(test_input: str = None) -> dict:
    """
    跑一次完整的调研流程

    Args:
        test_input: 测试输入，默认取 eval_cases.json 第一个用例

    Returns:
        {"turns": int, "research_complete": bool, "industry": str, "report_length": int}
    """
    import json

    if test_input is None:
        cases_path = os.path.join(PROJECT_ROOT, "eval", "eval_cases.json")
        with open(cases_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        test_input = cases[0]["input"] if cases else "我想在二线城市开一家宠物烘焙店，预算30万"

    os.chdir(PROJECT_ROOT)
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_core.messages import HumanMessage
    from state import ResearchState
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

    config = {"configurable": {"thread_id": "demo"}}
    final_state = {}

    for event in agent.stream(initial_state, config):
        for node_name, node_output in event.items():
            if node_output.get("research_complete"):
                final_state = node_output
            if node_name == "report_generator":
                msgs = node_output.get("messages", [])
                for msg in msgs:
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


# ============================================================
# 4. Agent 一致性检查
# ============================================================

def check_agent_consistency() -> dict:
    """
    检查所有 Agent 的 System Prompt + Tool 绑定 + 维度映射
    """
    import subprocess
    script_path = os.path.join(PROJECT_ROOT, "scripts", "check_agents.py")
    if not os.path.exists(script_path):
        return {"error": "scripts/check_agents.py 不存在"}

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "all_ok": result.returncode == 0,
    }
