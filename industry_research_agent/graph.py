"""
LangGraph 状态图定义 —— 整个 Agent 的核心编排

图结构：
    START → supervisor → [market_analyst / business_analyst / competition_analyst / conflict / report]
                ↑              ↓
                └──── 回到 supervisor（Loop）

Loop Engineering 在 LangGraph 中的体现：
- 条件边 (conditional_edges) = Supervisor 的 Plan 决策
- 循环边 = 分析师返回 supervisor，形成 Observe → Plan → Act → Reflect 循环
- 终止条件 = supervisor 输出 FINISH → 路由到 report_generator
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from agents.supervisor import supervisor_node
from agents.market_analyst import market_analyst_node
from agents.business_analyst import business_analyst_node
from agents.competition_analyst import competition_analyst_node
from report_generator import report_node


# ============================================================
# 路由函数：Supervisor 决策 → 下一个节点
# ============================================================
def route_after_supervisor(state: ResearchState) -> Literal[
    "market_analyst", "business_analyst", "competition_analyst", "conflict_resolver", "report_generator"
]:
    """
    根据 Supervisor 的 next_agent 字段路由到下一个节点。
    这是 LangGraph 条件边的核心——Loop Engineering 的 Plan 阶段产物。
    """
    decision = state.get("next_agent", "FINISH")

    if decision == "market_analyst":
        return "market_analyst"
    elif decision == "business_analyst":
        return "business_analyst"
    elif decision == "competition_analyst":
        return "competition_analyst"
    elif decision == "CONFLICT":
        return "conflict_resolver"
    else:  # FINISH 或未知
        return "report_generator"


def route_after_analyst(state: ResearchState) -> Literal["supervisor"]:
    """
    分析师回答完毕后，统一回到 Supervisor 进行下一轮 Observe → Plan。
    这是循环边的核心——形成 Loop。
    """
    return "supervisor"


# ============================================================
# 冲突消解节点
# ============================================================
def conflict_resolver_node(state: ResearchState) -> dict:
    """
    冲突消解节点 —— 安全级 Loop（第三层）

    当市场分析师建议进入赛道、竞争分析师反对时，
    Supervisor 不直接采纳任何一方，而是生成二选一问题交给用户。
    """
    from langchain_core.messages import AIMessage

    opinions = state.get("analyst_opinions", {})
    market_opinion = opinions.get("市场分析师", "无")
    competition_opinion = opinions.get("竞争分析师", "无")

    conflict_msg = AIMessage(content=f"""⚠️ **分析师意见冲突**

**市场分析师建议**：{market_opinion[:200]}

**竞争分析师意见**：{competition_opinion[:200]}

请选择：
- **A**：采纳市场分析师建议，进入该赛道
- **B**：采纳竞争分析师建议，暂不进入""")

    conflicts = state.get("conflicts", [])
    conflicts.append({
        "between": ["市场分析师", "竞争分析师"],
        "issue": f"市场:{market_opinion[:100]} vs 竞争:{competition_opinion[:100]}",
        "resolved_by": "等待用户选择"
    })

    return {
        "messages": [conflict_msg],
        "conflicts": conflicts,
        "next_agent": "FINISH",  # 等待用户回应后由 supervisor 重新路由
    }


# ============================================================
# 构建图
# ============================================================
def build_graph() -> StateGraph:
    """
    构建 LangGraph 状态图。

    图结构（ASCII）：
        START
          │
          ▼
      supervisor ──────────────┐
       │    │    │    │    │   │
       ▼    ▼    ▼    ▼    ▼   │
      mark busi comp confl rep │
       │    │    │    │        │
       └────┴────┴────┘        │
              │                │
              ▼                │
          supervisor ◄─────────┘
    """
    workflow = StateGraph(ResearchState)

    # 注册节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("business_analyst", business_analyst_node)
    workflow.add_node("competition_analyst", competition_analyst_node)
    workflow.add_node("conflict_resolver", conflict_resolver_node)
    workflow.add_node("report_generator", report_node)

    # 设置入口
    workflow.set_entry_point("supervisor")

    # 条件边：Supervisor → 各分析师 / 冲突 / 报告
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "market_analyst": "market_analyst",
            "business_analyst": "business_analyst",
            "competition_analyst": "competition_analyst",
            "conflict_resolver": "conflict_resolver",
            "report_generator": "report_generator",
        }
    )

    # 循环边：分析师 → 回到 Supervisor（核心 Loop）
    workflow.add_edge("market_analyst", "supervisor")
    workflow.add_edge("business_analyst", "supervisor")
    workflow.add_edge("competition_analyst", "supervisor")
    workflow.add_edge("conflict_resolver", "supervisor")

    # 终止边
    workflow.add_edge("report_generator", END)

    return workflow


# ============================================================
# 编译图（带内存 checkpoint）
# ============================================================
def compile_graph():
    """编译图，返回可执行的 Agent"""
    workflow = build_graph()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
