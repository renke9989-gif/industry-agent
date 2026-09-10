"""Backward-compatible diagnostics entry point.

The legacy helper used to initialize a local knowledge store as a side effect;
the compatibility entry point now delegates only to offline checks and never
starts a model or loads a local knowledge base.
"""
from scripts.agent_tools import check_agent_consistency, check_environment, run_eval_harness

__all__ = ["check_environment", "run_eval_harness", "check_agent_consistency"]


if __name__ == "__main__":
    import json
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "check":
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
    elif command == "eval":
        raise SystemExit(run_eval_harness())
    elif command == "agents":
        raise SystemExit(check_agent_consistency())
    else:
        print("用法: python agent_tools.py [check|eval|agents]")
        raise SystemExit(2)
