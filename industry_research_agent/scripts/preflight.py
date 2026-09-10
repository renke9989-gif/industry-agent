"""Check the local environment before live evaluation or an interview demo."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_env_file() -> None:
    """Load local .env for a standalone preflight invocation.

    The web server already calls python-dotenv, but this script is commonly
    run directly from PowerShell.  Loading the file here prevents a false
    negative when keys are configured in the project .env rather than in the
    parent shell.  Existing process variables always win.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env", override=False)


def _module_available(name: str) -> bool:
    """Check an optional dependency without crashing on a missing parent."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def main() -> int:
    _load_env_file()
    checks = {
        "python_3_11_plus": sys.version_info >= (3, 11),
        "llm_api_key": bool(os.getenv("LLM_API_KEY")),
        "llm_model": bool(os.getenv("LLM_MODEL")),
        "search_sdk_installed": _module_available("tavily") or _module_available("ddgs"),
        "search_available": (bool(os.getenv("TAVILY_API_KEY")) and _module_available("tavily"))
        or _module_available("ddgs"),
        "sqlite_checkpointer_installed": _module_available("langgraph.checkpoint.sqlite"),
        "web_ui_exists": Path("web_ui/index.html").exists(),
        "cost_rates_configured": bool(os.getenv("LLM_INPUT_COST_PER_1K") and os.getenv("LLM_OUTPUT_COST_PER_1K")),
        "claim_judge_configured": bool(os.getenv("EVAL_JUDGE_API_KEY") and os.getenv("EVAL_JUDGE_MODEL")),
    }
    required = ("python_3_11_plus", "llm_api_key", "llm_model", "search_available", "sqlite_checkpointer_installed", "web_ui_exists")
    output = {
        "ready": all(checks[name] for name in required),
        "checks": checks,
        "required_checks": list(required),
        "optional_checks": ["cost_rates_configured", "claim_judge_configured"],
        "note": "cost_rates_configured 仅影响成本估算，不影响运行",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
