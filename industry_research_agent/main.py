"""Command-line entry point for one-shot web research."""
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from graph import compile_graph
from state import initial_state


load_dotenv()
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    if not os.getenv("LLM_API_KEY"):
        print("请复制 .env.example 为 .env，并配置 LLM_API_KEY。")
        return 2
    query = " ".join(sys.argv[1:]).strip() or input("请输入要调研的行业或生意：").strip()
    if not query:
        return 0
    agent = compile_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}
    for event in agent.stream(initial_state(HumanMessage(content=query)), config):
        for node, output in event.items():
            for message in output.get("messages", []):
                content = str(getattr(message, "content", "")).strip()
                if content and not content.startswith("[Supervisor 决策]"):
                    print(f"\n[{node}]\n{content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
