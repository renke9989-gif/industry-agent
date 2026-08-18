"""
ResearchState —— Agent 全局状态定义
基于 TypedDict，LangGraph 原生支持
"""
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class KnowledgeItem(TypedDict, total=False):
    """知识库条目"""
    id: str
    industry: str
    topic: str
    content: str
    key_data: List[dict]          # [{"metric": "年增速", "value": "15%", "year": "2026"}]
    region: str
    related_dimensions: List[str]
    source: str                   # manual / web_search / gov / iresearch
    source_url: str
    confidence: float             # 0-1
    access_count: int


class ResearchState(TypedDict):
    # === 对话历史（LangGraph 要求用 Annotated + add_messages） ===
    messages: Annotated[List[BaseMessage], add_messages]

    # === 调研进度 ===
    scores: Dict[str, int]           # {"市场":0, "竞争":0, "商业模式":0, "机会":0, "风险":0, "趋势":0}
                                      # 0=未采集, 1-5=已评估

    # === 调研对象画像 ===
    industry: str                     # pet_economy / coffee / second_hand_luxury / unknown
    region: str                       # 目标区域
    budget: str                       # 用户预算

    # === 各分析师产出 ===
    analyst_opinions: Dict[str, str]  # {"市场分析师":"市场规模约X亿，年增速15%", ...}
    analyst_questions: Dict[str, str] # 各分析师当前待追问问题

    # === 路由与控制 ===
    next_agent: str                   # Supervisor 决定的下一个发言者
    research_complete: bool           # True → 进入报告生成
    turn_count: int                   # 当前轮次（≥15 强制结束）
    supervisor_reason: str            # Supervisor 本次路由决策的理由（可观测性）

    # === 冲突记录 ===
    conflicts: List[dict]             # [{"between":["市场分析师","竞争分析师"], "issue":"..."}]

    # === 动态知识库 ===
    knowledge_cache: List[dict]       # 本次调研新入库的知识
    search_triggered: bool            # 是否触发了联网搜索
    last_retrieval_score: float       # 最近一次 RAG 检索的相似度

    # === 错误处理 ===
    error_count: int                  # 连续错误计数
    force_finish: bool                # 超时/超轮次强制结束标志
