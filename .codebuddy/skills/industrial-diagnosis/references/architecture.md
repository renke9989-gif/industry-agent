# 架构总览

工业诊断 Agent 是基于 LangGraph 的 Supervisor-Worker 多智能体架构。

## 图结构

```
START → supervisor → [process_expert / finance_expert / equipment_expert]
                ↑              ↓
                └──── Loop 循环 ────┘
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `state.py` | DiagnosisState 定义 |
| `graph.py` | LangGraph 状态图 + 条件边 |
| `agents/supervisor.py` | 规则+LLM 混合路由 |
| `agents/process_expert.py` | 工艺专家 + RAG 工具链 |
| `agents/finance_expert.py` | 财务专家 + ROI 计算器 |
| `agents/equipment_expert.py` | 设备专家 + MCP |
| `tools/` | 知识检索/ROI/MES 工具 |
| `mcp_servers/` | MCP Server |
| `eval/` | Eval Harness |
| `knowledge_packs/` | 可插拔知识包 |
