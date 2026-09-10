"""Small project diagnostics CLI."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_environment() -> dict:
    required = ["langgraph", "langchain_core", "langchain_openai", "dotenv", "httpx", "bs4"]
    deps = {name: bool(importlib.util.find_spec(name)) for name in required}
    env = {name: bool(os.getenv(name)) for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL")}
    files = {name: (ROOT / name).exists() for name in ("state.py", "graph.py", "main.py", "research_tools.py", "requirements.txt")}
    return {"python": sys.version_info >= (3, 11), "deps": deps, "env": env, "files": files,
            "all_ok": sys.version_info >= (3, 11) and all(deps.values()) and all(files.values())}


def run_eval_harness() -> int:
    return subprocess.call([sys.executable, "-m", "eval.eval_harness"], cwd=ROOT)


def check_agent_consistency() -> int:
    return subprocess.call([sys.executable, "scripts/check_agents.py"], cwd=ROOT)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "check":
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
    elif command == "eval":
        raise SystemExit(run_eval_harness())
    elif command == "agents":
        raise SystemExit(check_agent_consistency())
    else:
        print("用法: python scripts/agent_tools.py [check|eval|agents]")
        raise SystemExit(2)
