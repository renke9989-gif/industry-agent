"""
报告生成节点 —— 汇总三位分析师意见，输出结构化 Markdown 报告
"""
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from state import ResearchState

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.5,
)


REPORT_PROMPT = """你是一位资深的行业调研报告撰写专家。请根据以下调研数据，生成一份结构化的行业调研报告。

## 调研数据

- 行业：{industry}
- 目标区域：{region}
- 预算：{budget}
- 调研轮次：{turn_count} 轮
- 是否触发联网搜索：{search_triggered}

### 各维度评分（1-5分，5=优秀）
{scores_detail}

### 市场趋势分析师意见
{market_opinion}

### 竞争格局分析师意见
{competition_opinion}

### 商业模式分析师意见
{business_opinion}

### 分析师冲突记录
{conflicts}

## 报告要求

请严格按照以下结构输出 Markdown 格式报告：

---

# 🔍 行业调研报告

## 一、行业画像
- 行业、区域、预算
- 核心调研问题概述

## 二、六维度调研评估

| 维度 | 评分 | 现状 | 关键发现 |
|------|------|------|------|
| 市场规模 | X/5 | ... | ... |
| 竞争格局 | X/5 | ... | ... |
| 商业模式 | X/5 | ... | ... |
| 市场机会 | X/5 | ... | ... |
| 进入风险 | X/5 | ... | ... |
| 发展趋势 | X/5 | ... | ... |

## 三、雷达图数据（JSON）
```json
{{"市场": X, "竞争": X, "商业模式": X, "机会": X, "风险": X, "趋势": X}}
```

## 四、市场规模与趋势分析
市场规模、增速、驱动因素、政策环境

## 五、竞争格局分析
主要玩家、集中度、进入壁垒

## 六、商业模式与盈利测算
盈利模式、成本结构、客单价、回本周期

## 七、Top-3 决策建议（含明确结论）

### 🔴 核心建议
### 🟡 次优选择
### 🟢 备选方案

每条建议需包含：投入估算、预期收益、ROI、可行性

## 八、风险提示清单
逐条列出进入该赛道的主要风险

## 九、调研过程摘要
- 对话轮次、覆盖维度、知识检索情况、数据来源

---

**重要**：
- 所有结论必须追溯到具体分析师发言或检索结果，不能凭空捏造
- 每条关键数据标注来源（本地知识库/联网搜索）
- 如果某维度未采集到数据，标注"数据不足，建议补充调研"
- 联网搜索获取的数据标注来源和年份
"""


def report_node(state: ResearchState) -> dict:
    """
    报告生成节点 —— LangGraph 节点函数
    """
    scores = state.get("scores", {})
    opinions = state.get("analyst_opinions", {})
    conflicts = state.get("conflicts", [])

    # 构建评分详情
    scores_detail = "\n".join([
        f"- {dim}: {score}/5" for dim, score in scores.items()
    ]) if scores else "暂无评分数据"

    # 构建 Prompt
    prompt = REPORT_PROMPT.format(
        industry=state.get("industry", "未知"),
        region=state.get("region", "未知"),
        budget=state.get("budget", "未知"),
        turn_count=state.get("turn_count", 0),
        search_triggered="是" if state.get("search_triggered") else "否",
        scores_detail=scores_detail,
        market_opinion=opinions.get("市场分析师", "未参与调研"),
        competition_opinion=opinions.get("竞争分析师", "未参与调研"),
        business_opinion=opinions.get("商业模式分析师", "未参与调研"),
        conflicts=str(conflicts) if conflicts else "无冲突",
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
        "research_complete": True,
    }
