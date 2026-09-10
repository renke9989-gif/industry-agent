# 当前架构

```text
用户输入
  → Supervisor：识别 user_intent + business_stage
  → 生成 research_plan
  → 市场 / 竞争 / 商业分析师
  → MCP Provider（失败降级 Direct）
  → Evidence
  → 引用报告
```

核心运行时状态：

- `user_intent`：市场概览、进入可行性、实际 ROI、经营诊断、行业对比。
- `business_stage`：`pre_launch`、`planning`、`operating`、`unknown`。
- `assumption_mode`：筹备期允许使用公开行业基准进行情景测算。
- `research_plan`：根据用户旅程决定研究维度和模式。
- `evidence`：带来源、时间、摘录和置信度的结构化证据。

验证入口：

```powershell
.\industry_research_agent\.venv\Scripts\python.exe -m eval.eval_harness
.\industry_research_agent\.venv\Scripts\python.exe -m eval.online_eval --case coffee_hangzhou --runs 1 --provider direct
.\industry_research_agent\.venv\Scripts\python.exe -m eval.ablation --runs 1
```
