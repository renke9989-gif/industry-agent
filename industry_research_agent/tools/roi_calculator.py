"""
工具模块：ROI 测算
商业模式分析师的专属 Tool
"""
from langchain_core.tools import tool


@tool
def calculate_roi(investment: float, annual_benefit: float) -> dict:
    """
    计算进入某赛道/投资方案的投入产出比。

    Args:
        investment: 投入金额（万元）
        annual_benefit: 预计年化收益（万元）

    Returns:
        {"roi": float, "payback_years": float, "verdict": str, "reason": str}
    """
    if investment <= 0:
        return {
            "roi": 0,
            "payback_years": 999,
            "verdict": "invalid",
            "reason": "投入金额必须大于 0"
        }

    roi = annual_benefit / investment
    payback_years = investment / annual_benefit if annual_benefit > 0 else 999

    if roi >= 1.5:
        verdict = "recommend"
        reason = f"ROI={roi:.1f}，{payback_years:.1f} 年回本，建议进入"
    elif roi >= 1.0:
        verdict = "cautious"
        reason = f"ROI={roi:.1f}，{payback_years:.1f} 年回本，可谨慎考虑"
    else:
        verdict = "reject"
        reason = f"ROI={roi:.1f}，{payback_years:.1f} 年回本，不建议进入"

    return {
        "roi": round(roi, 2),
        "payback_years": round(payback_years, 1),
        "verdict": verdict,
        "reason": reason
    }
