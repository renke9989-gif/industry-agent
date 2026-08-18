"""
Agent 评估框架 (Eval Harness)

评估维度：
- 路由准确率：Supervisor 决策是否与预期一致
- 工具调用准确率：是否在正确时机调用了正确的 Tool
- 信息完整率：6 维度是否全部覆盖
- 幻觉率：回答中无依据断言的比例
- 循环效率：达到 FINISH 所用轮次

使用方式：
    python eval/eval_harness.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from state import ResearchState
from graph import compile_graph
from tools.knowledge_retriever import load_seed_knowledge


def load_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(agent, case: dict) -> dict:
    """评估单个测试用例"""
    initial_state: ResearchState = {
        "messages": [HumanMessage(content=case["input"])],
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

    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}
    final_state = None

    try:
        for event in agent.stream(initial_state, config):
            for node_output in event.values():
                if node_output.get("research_complete"):
                    final_state = node_output
        if not final_state:
            final_state = agent.get_state(config).values
    except Exception as e:
        return {"case_id": case["id"], "error": str(e), "scores": {}}

    expected = case.get("expected", {})

    # 提取实际路由序列
    actual_route = []
    try:
        snapshot = agent.get_state(config)
        if snapshot and snapshot.values:
            for msg in snapshot.values.get("messages", []):
                content = getattr(msg, "content", "")
                if "[Supervisor 决策]" in content:
                    if "市场分析师" in content:
                        actual_route.append("market_analyst")
                    elif "竞争分析师" in content:
                        actual_route.append("competition_analyst")
                    elif "商业模式分析师" in content:
                        actual_route.append("business_analyst")
    except Exception:
        pass

    # 评分
    scores = {}

    # 1. 路由准确率
    expected_route = expected.get("route_sequence", [])
    if expected_route:
        matches = sum(1 for a, e in zip(actual_route, expected_route) if a == e)
        scores["route_acc"] = matches / max(len(expected_route), 1)

    # 2. 幻觉检测
    forbidden = expected.get("must_not_contain", [])
    all_text = ""
    try:
        snapshot = agent.get_state(config)
        if snapshot and snapshot.values:
            for msg in snapshot.values.get("messages", []):
                all_text += getattr(msg, "content", "")
    except Exception:
        pass

    if forbidden:
        violations = sum(1 for w in forbidden if w in all_text)
        scores["hallucination"] = 1.0 - (violations / len(forbidden))

    # 3. 关键词检测
    required = expected.get("must_contain_keywords", [])
    if required:
        hits = sum(1 for w in required if w in all_text)
        scores["keyword_coverage"] = hits / len(required)

    # 4. 搜索触发检测
    if "search_triggered" in expected:
        scores["search_triggered"] = 1.0 if final_state.get("search_triggered") else 0.0

    # 5. 循环效率
    max_turns = expected.get("max_turns", 15)
    actual_turns = final_state.get("turn_count", 0)
    if actual_turns > 0:
        scores["loop_efficiency"] = min(1.0, max_turns / max(actual_turns, 1))

    return {
        "case_id": case["id"],
        "name": case.get("name", ""),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "turns": actual_turns,
        "scores": scores
    }


def main():
    print("=" * 60)
    print("🧪 Agent Eval Harness")
    print("=" * 60)

    # 加载知识库
    print("\n[初始化] 加载知识库...")
    load_seed_knowledge()

    # 编译 Agent
    print("[初始化] 编译 Agent...")
    agent = compile_graph()

    # 加载测试用例
    cases_path = os.path.join(os.path.dirname(__file__), "eval_cases.json")
    cases = load_cases(cases_path)
    print(f"[初始化] 加载 {len(cases)} 个测试用例\n")

    # 批量评估
    results = []
    for case in cases:
        print(f"🧪 测试 {case['id']}: {case.get('name', '')} ... ", end="")
        result = evaluate(agent, case)
        results.append(result)

        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            scores = result.get("scores", {})
            route_acc = scores.get("route_acc", "N/A")
            hallu = scores.get("hallucination", "N/A")
            print(f"✅ 路由:{route_acc} 幻觉:{hallu} 轮次:{result.get('turns', '?')}")

    # 汇总
    print("\n" + "=" * 60)
    print("📊 评估汇总")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r and r.get("scores")]
    if valid_results:
        route_accs = [r["scores"].get("route_acc", 0) for r in valid_results]
        hallu_rates = [r["scores"].get("hallucination", 0) for r in valid_results]
        keyword_covs = [r["scores"].get("keyword_coverage", 0) for r in valid_results]
        turns = [r.get("turns", 0) for r in valid_results]

        print(f"测试用例数    : {len(results)}")
        print(f"有效结果数    : {len(valid_results)}")
        print(f"路由准确率    : {sum(route_accs)/len(route_accs):.1%}" if route_accs else "路由准确率: N/A")
        print(f"幻觉控制率    : {sum(hallu_rates)/len(hallu_rates):.1%}" if hallu_rates else "幻觉控制率: N/A")
        print(f"关键词覆盖    : {sum(keyword_covs)/len(keyword_covs):.1%}" if keyword_covs else "关键词覆盖: N/A")
        print(f"平均轮次      : {sum(turns)/len(turns):.1f}" if turns else "平均轮次: N/A")

    print("\n✅ 评估完成")


if __name__ == "__main__":
    main()
