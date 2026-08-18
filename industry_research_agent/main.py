"""
入口文件 —— 行业调研多智能体协作系统

使用方式：
    python main.py
    然后输入想调研的行业/想做的生意，Agent 自动调研
"""
import os
import sys
from dotenv import load_dotenv

# 强制 stdout/stderr 使用 UTF-8，避免 Windows GBK 控制台打印 emoji 崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

# 检查 API Key
if not os.getenv("LLM_API_KEY"):
    print("❌ 请设置环境变量 LLM_API_KEY")
    print("   cp .env.example .env 并填入你的 API Key")
    sys.exit(1)

from langchain_core.messages import HumanMessage
from state import ResearchState
from graph import compile_graph
from tools.knowledge_retriever import load_seed_knowledge


def main():
    print("=" * 60)
    print("🔍 行业调研多智能体协作系统")
    print("   Multi-Agent Industry Research System")
    print("=" * 60)

    # 加载种子知识
    print("\n[初始化] 加载知识库...")
    total = load_seed_knowledge()
    print(f"[初始化] 知识库就绪，共 {total} 条种子知识")

    # 编译图
    print("[初始化] 编译 Agent 图...")
    agent = compile_graph()
    print("[初始化] Agent 就绪\n")

    print("💡 请输入想调研的行业/想做的生意描述，Agent 将自动调研")
    print("   示例：我想在二线城市开一家宠物烘焙店，预算30万")
    print("   输入 'quit' 退出\n")

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        print("\n🤖 Agent 正在分析...\n")

        # 初始化 State
        initial_state: ResearchState = {
            "messages": [HumanMessage(content=user_input)],
            "scores": {"市场": 0, "竞争": 0, "商业模式": 0, "机会": 0, "风险": 0, "趋势": 0},
            "industry": "unknown",
            "region": "未知",
            "budget": "未知",
            "analyst_opinions": {},
            "analyst_questions": {},
            "next_agent": "",
            "research_complete": False,
            "turn_count": 0,
            "supervisor_reason": "",
            "conflicts": [],
            "knowledge_cache": [],
            "search_triggered": False,
            "last_retrieval_score": 0.0,
            "error_count": 0,
            "force_finish": False,
        }

        # 运行 Agent（stream 模式，实时输出）
        config = {"configurable": {"thread_id": "demo-session"}}
        final_state = None
        try:
            for event in agent.stream(initial_state, config):
                for node_name, node_output in event.items():
                    # 保存最终状态
                    if node_name == "report_generator":
                        final_state = node_output

                    messages = node_output.get("messages", [])
                    for msg in messages:
                        role = {
                            "supervisor": "🤖 监督者",
                            "market_analyst": "📈 市场分析师",
                            "business_analyst": "💰 商业模式分析师",
                            "competition_analyst": "⚔️ 竞争分析师",
                            "conflict_resolver": "⚠️ 冲突裁决",
                            "report_generator": "📄 报告",
                        }.get(node_name, f"📌 {node_name}")

                        content = getattr(msg, "content", str(msg))

                        # Supervisor 决策只显示简短摘要
                        if content.startswith("[Supervisor 决策]"):
                            print(f"  {content.split(chr(10))[0]}")
                            continue

                        # 报告完整显示
                        if node_name == "report_generator":
                            print(f"\n{'='*60}")
                            print(content)
                            print(f"{'='*60}\n")
                        else:
                            # 分析师发言显示前 300 字符
                            display = content[:300]
                            print(f"{role}: {display}")
                            if len(content) > 300:
                                print(f"  ... (共 {len(content)} 字符)")
                        print()
        except Exception as e:
            print(f"\n❌ Agent 运行异常: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
