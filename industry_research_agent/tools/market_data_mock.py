"""
工具模块：行业数据模拟查询
竞争格局分析师的专属 Tool

模拟一个行业数据库，展示 Agent 调用外部 API 获取市场数据的能力。
"""
import random
from langchain_core.tools import tool


# 模拟行业数据库
_MOCK_MARKET_DB = {
    "pet_economy": {
        "market_size": "约2500亿元", "annual_growth": "15.2%",
        "competitor_count": 3, "entry_barrier": "低", "profit_margin": "25-35%",
        "top_players": ["某头部连锁", "本地网红店", "电商品牌"],
    },
    "coffee": {
        "market_size": "约1300亿元", "annual_growth": "18.5%",
        "competitor_count": 5, "entry_barrier": "中", "profit_margin": "15-25%",
        "top_players": ["瑞幸", "库迪", "星巴克", "Manner", "本地精品店"],
    },
    "second_hand_luxury": {
        "market_size": "约800亿元", "annual_growth": "22.0%",
        "competitor_count": 2, "entry_barrier": "高", "profit_margin": "10-20%",
        "top_players": ["得物", "胖虎", "红布林"],
    },
}


@tool
def query_market_data(industry: str) -> dict:
    """
    查询模拟的行业市场数据（市场规模、增速、竞争格局）。

    Args:
        industry: 行业标识，如 "pet_economy" / "coffee" / "second_hand_luxury"

    Returns:
        {"industry": str, "market_size": str, "annual_growth": str,
         "competitor_count": int, "entry_barrier": str, "profit_margin": str,
         "top_players": list, "status": str}
    """
    market = _MOCK_MARKET_DB.get(industry)

    if market is None:
        return {
            "industry": industry,
            "market_size": f"约{random.randint(100, 5000)}亿元",
            "annual_growth": f"{random.uniform(5, 30):.1f}%",
            "competitor_count": random.randint(1, 8),
            "entry_barrier": random.choice(["低", "中", "高"]),
            "profit_margin": f"{random.randint(8, 40)}-{random.randint(15, 50)}%",
            "top_players": ["待调研"],
            "status": "该行业暂无收录数据，以上为估算值，建议联网搜索核实"
        }

    return {**market, "industry": industry, "status": "数据来源于行业报告（演示用模拟数据）"}


@tool
def list_industries() -> list:
    """
    列出已收录的行业及其市场概况。
    """
    result = []
    for ind, data in _MOCK_MARKET_DB.items():
        result.append({
            "industry": ind,
            "market_size": data["market_size"],
            "annual_growth": data["annual_growth"],
            "entry_barrier": data["entry_barrier"],
        })
    return result
