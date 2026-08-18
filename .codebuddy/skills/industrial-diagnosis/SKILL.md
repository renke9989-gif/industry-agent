---
name: industrial-diagnosis
description: |
  工业诊断多智能体协作系统的开发辅助 skill。当用户需要：
  - 检查项目健康状态（依赖/配置/文件结构）
  - 跑 Eval Harness 回归测试
  - 一行命令跑预设测试对话
  - 检查 Agent Prompt 和 Tool 绑定一致性
  时触发此 skill。触发词："diagnose"、"诊断"、"Agent 测试"、"跑 eval"、"检查项目"、"demo"。
allowed-tools:
  - Bash
  - Read
---

# 工业诊断 Agent 开发辅助

## 项目路径

Agent 代码在 `industrial_diagnosis_agent/` 下。所有命令从项目根目录执行。

## 四个操作

| 命令 | 功能 |
|------|------|
| `python industrial_diagnosis_agent/scripts/agent_tools.py check` | 环境检查 |
| `python industrial_diagnosis_agent/scripts/agent_tools.py eval` | 跑 Eval Harness |
| `python industrial_diagnosis_agent/scripts/agent_tools.py demo` | 跑预设测试对话 |
| `python industrial_diagnosis_agent/scripts/agent_tools.py agents` | Agent 一致性检查 |

## 典型开发流程

```bash
# 1. 改完代码后先检查
python industrial_diagnosis_agent/scripts/agent_tools.py check

# 2. 跑 Harness 回归
python industrial_diagnosis_agent/scripts/agent_tools.py eval

# 3. 快速验证端到端
python industrial_diagnosis_agent/scripts/agent_tools.py demo

# 4. 手动对话测试
cd industrial_diagnosis_agent && python main.py
```

## 参考资料

- [references/architecture.md](references/architecture.md) — 架构总览
- [references/interview_cheatsheet.md](references/interview_cheatsheet.md) — 面试话术
- [assets/demo_inputs.md](assets/demo_inputs.md) — 测试输入合集
