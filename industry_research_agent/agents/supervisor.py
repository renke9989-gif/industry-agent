"""
Agent 节点：协调监督者 (Supervisor)

核心职责：
1. Observe - 感知当前 State，提取行业/区域/预算
2. Plan   - 规则 + LLM 混合路由，决定下一个发言的分析师
3. Reflect - 判断是否 FINISH / 是否冲突 / 是否超轮次

这是 Loop Engineering 的第二层（决策级 Loop）和第三层（安全级 Loop）
"""
import os
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from state import ResearchState

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.1,  # 路由决策需要低温度保证稳定
)

# ============================================================
# 维度 → 分析师映射
# ============================================================
DIMENSION_TO_ANALYST = {
    "市场": "market_analyst",
    "竞争": "competition_analyst",
    "商业模式": "business_analyst",
    "机会": "market_analyst",
    "风险": "competition_analyst",
    "趋势": "market_analyst",
}

# 硬编码优先级（规则兜底）
RULE_PRIORITY = ["市场", "竞争", "商业模式", "机会", "风险", "趋势"]


# ============================================================
# 规则路由（硬编码，保证稳定性）
# ============================================================
def rule_based_routing(state: ResearchState) -> str:
    """
    硬编码优先级路由。80% 情况下走这条路径。
    """
    scores = state.get("scores", {})
    for dim in RULE_PRIORITY:
        if scores.get(dim, 0) == 0:
            return DIMENSION_TO_ANALYST[dim]
    return "FINISH"


# ============================================================
# LLM 路由（处理异常情况 + 行业识别 + 冲突检测）
# ============================================================
SUPERVISOR_PROMPT = """你是行业调研项目的总负责人（Supervisor）。

当前状态：
- 行业：{industry}
- 目标区域：{region}
- 预算：{budget}
- 已评估维度：{filled_dims}
- 未评估维度：{empty_dims}
- 当前轮次：{turn_count}/15
- 各分析师意见：{opinions}

你的决策选项：
1. "market_analyst"       - 调度市场趋势分析师（负责市场/机会/趋势维度）
2. "competition_analyst"  - 调度竞争格局分析师（负责竞争/风险维度）
3. "business_analyst"     - 调度商业模式分析师（负责商业模式维度）
4. "FINISH"               - 所有维度已填满，终止调研
5. "CONFLICT"             - 分析师意见冲突，需要生成二选一问题交用户裁决

规则：
- 优先填充未评估的维度
- 如果用户输入中明确提到"竞品""对手""红海""壁垒"，优先调度竞争分析师
- 如果用户提到"预算""成本""利润""赚钱""回本"，优先调度商业模式分析师
- 如果检测到市场分析师建议进入但竞争分析师判定已红海/壁垒低，输出 CONFLICT
- 只有全部6个维度都有评分时，才输出 FINISH
- 超过15轮，强制 FINISH

请只输出上述5个选项之一，不要输出其他内容。
"""


def llm_based_routing(state: ResearchState) -> str:
    """
    LLM 路由——处理规则覆盖不到的 20% 异常情况。
    同时负责行业识别和冲突检测。
    """
    scores = state.get("scores", {})
    filled = [d for d, s in scores.items() if s > 0]
    empty = [d for d, s in scores.items() if s == 0]
    turn_count = state.get("turn_count", 0)

    # 超轮次强制终止
    if turn_count >= 15:
        return "FINISH"

    # 全部填满 → 直接结束（必须先于冲突检测，否则已完成的调研会被误判为冲突死循环）
    if not empty:
        return "FINISH"

    # 冲突检测：市场分析师有建议 + 竞争分析师有反对
    opinions = state.get("analyst_opinions", {})
    if opinions.get("市场分析师") and opinions.get("竞争分析师"):
        market_opinion = opinions["市场分析师"]
        competition_opinion = opinions["竞争分析师"]
        enter_keywords = ["进入", "值得", "建议", "增长", "机会"]
        reject_keywords = ["红海", "壁垒", "竞争激烈", "不建议", "风险", "饱和"]
        if (any(kw in market_opinion for kw in enter_keywords) and
                any(kw in competition_opinion for kw in reject_keywords)):
            return "CONFLICT"

    # LLM 判断（处理用户提及特定维度关键词的情况）
    try:
        prompt = SUPERVISOR_PROMPT.format(
            industry=state.get("industry", "未知"),
            region=state.get("region", "未知"),
            budget=state.get("budget", "未知"),
            filled_dims=", ".join(filled) if filled else "无",
            empty_dims=", ".join(empty),
            turn_count=turn_count,
            opinions=str(opinions)[:300]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip()

        valid_decisions = ["market_analyst", "competition_analyst", "business_analyst", "FINISH", "CONFLICT"]
        if decision in valid_decisions:
            return decision
    except Exception:
        pass

    # LLM 失败时回退到规则路由
    return rule_based_routing(state)


# ============================================================
# 行业识别
# ============================================================
INDUSTRY_KEYWORDS = {
    "pet_economy": ["宠物", "猫", "狗", "宠物店", "宠物烘焙", "宠物食品", "宠物用品"],
    "coffee": ["咖啡", "咖啡店", "咖啡馆", "瑞幸", "库迪", "星巴克"],
    "second_hand_luxury": ["二手奢侈品", "奢侈品", "二手包", "得物", "中古", "回收"],
    "catering": ["餐饮", "餐厅", "饭店", "外卖", "加盟", "小吃"],
    "education": ["教培", "教育", "培训", "K12", "兴趣班"],
    "new_retail": ["新零售", "生鲜", "便利店", "社区团购"],
    "e_commerce": ["电商", "直播带货", "抖音", "拼多多", "淘宝"],
}


def identify_industry(user_input: str) -> str:
    """从用户输入中自动识别行业"""
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in user_input:
                return industry
    return "unknown"


# ============================================================
# Supervisor 节点函数（Observe → Plan → Act/Reflect）
# ============================================================
def supervisor_node(state: ResearchState) -> dict:
    """
    Supervisor 节点 —— LangGraph 节点函数

    每轮执行：Observe → Plan → 决策
    """
    turn_count = state.get("turn_count", 0) + 1
    messages = state.get("messages", [])

    # ===== Observe 阶段：感知当前 State =====

    latest_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break

    # 行业识别（仅首次）
    industry = state.get("industry", "unknown")
    if industry == "unknown" and latest_user_msg:
        industry = identify_industry(latest_user_msg)

    # 预算识别
    budget = state.get("budget", "未知")
    if budget == "未知" and latest_user_msg:
        match = re.search(r'(\d+)\s*万', latest_user_msg)
        if match:
            budget = f"{match.group(1)}万"

    # 区域识别
    region = state.get("region", "未知")
    if region == "未知" and latest_user_msg:
        match = re.search(r'([一二三四五六]线)[城市场]?', latest_user_msg)
        if match:
            region = match.group(1) + "城市"

    # ===== 自动评分：检测分析师是否已给出"信息充足"信号 =====
    scores = dict(state.get("scores", {}))
    dim_keywords = {
        "市场": "市场维度信息充足",
        "竞争": "竞争维度信息充足",
        "商业模式": "商业模式维度信息充足",
        "机会": "机会维度信息充足",
        "风险": "风险维度信息充足",
        "趋势": "趋势维度信息充足",
    }
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            for dim, signal in dim_keywords.items():
                if signal in content and scores.get(dim, 0) == 0:
                    scores[dim] = 3
            break  # 只看最近一条消息

    # 宽松兜底：分析师已产出意见 → 其负责维度视为已评估。
    # 防止"模型措辞不规范导致维度永不打分 → 路由死循环 → 轮次被单一分析师耗光"
    opinions = state.get("analyst_opinions", {})
    dim_to_analyst = {
        "市场": "市场分析师", "机会": "市场分析师", "趋势": "市场分析师",
        "竞争": "竞争分析师", "风险": "竞争分析师",
        "商业模式": "商业模式分析师",
    }
    for dim, analyst_key in dim_to_analyst.items():
        if scores.get(dim, 0) == 0 and opinions.get(analyst_key):
            scores[dim] = 3

    # ===== Plan 阶段：决定路由 =====

    # 用本回合最新 scores 路由（避免用上一轮旧 scores 重复调度同一分析师）
    routing_state = dict(state)
    routing_state["scores"] = scores
    decision = rule_based_routing(routing_state)

    # 检测用户输入中的维度关键词，临时调整路由
    if latest_user_msg:
        if any(kw in latest_user_msg for kw in ["竞品", "对手", "红海", "壁垒", "竞争"]):
            if scores.get("竞争", 0) == 0:
                decision = "competition_analyst"
        if any(kw in latest_user_msg for kw in ["预算", "成本", "利润", "赚钱", "回本", "盈利"]):
            if scores.get("商业模式", 0) == 0:
                decision = "business_analyst"

    # 冲突检测（走一遍 LLM 检查冲突，用最新 scores 避免重复调度已评估分析师）
    if decision != "CONFLICT":
        decision = llm_based_routing(routing_state)

    # ===== Reflect 阶段：输出决策 + 理由 =====
    reason_map = {
        "market_analyst": f"市场/机会/趋势维度未评估，调度市场分析师深挖（第{turn_count}轮）",
        "competition_analyst": f"竞争/风险维度未评估，调度竞争分析师（第{turn_count}轮）",
        "business_analyst": f"商业模式维度未评估，调度商业模式分析师（第{turn_count}轮）",
        "FINISH": f"所有维度已收集完毕，共{turn_count}轮，输出报告",
        "CONFLICT": "检测到分析师意见冲突，生成二选一问题交用户裁决",
    }

    reason = reason_map.get(decision, f"未知决策: {decision}")

    supervisor_msg = AIMessage(
        content=f"[Supervisor 决策] {reason}\n[下一节点] {decision}"
    )

    return {
        "messages": [supervisor_msg],
        "next_agent": decision,
        "turn_count": turn_count,
        "industry": industry,
        "budget": budget,
        "region": region,
        "scores": scores,
        "supervisor_reason": reason,
        "research_complete": (decision == "FINISH"),
    }
