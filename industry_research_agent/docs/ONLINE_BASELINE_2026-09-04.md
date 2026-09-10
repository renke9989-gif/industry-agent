# 在线评测基线（2026-09-04）

这是一份可复现的 Direct Provider 基线，不代表所有搜索供应商和模型版本下的固定表现。评测使用 `eval/online_cases.json` 中的 10 个案例，每个案例运行 3 次，共 30 次真实联网任务。

| 指标 | 结果 |
|---|---:|
| 任务完成率 | 100%（30/30） |
| URL 有效率 | 100% |
| Skill 初始路由准确率 | 100% |
| 工具成功率 | 97.2% |
| 平均总 Token | 14,638（输入 12,928 / 输出 1,710） |
| 原始引用覆盖率 | 93.9%（旧口径，修正规则后需重跑） |
| 平均耗时 | 28.7 秒 |
| P95 延迟 | 单案例低于 90 秒 |
| 目标年份证据占比 | 约 45% |
| Claim Judge 支持率 | unavailable（未配置独立 Judge） |
| 估算成本 | unavailable（未配置模型价格） |

## 如何复现

```powershell
cd D:\AI学习\industry-agent\industry_research_agent
.\.venv\Scripts\python.exe -m scripts.preflight
.\.venv\Scripts\python.exe -m eval.online_eval --runs 3 --provider direct
```

在线评测会消耗真实 LLM 和搜索额度。结果写入 `eval/results/`，该目录默认被 Git 忽略；提交代码时只提交本页和脱敏汇总，不提交完整网页正文或密钥。

## 当前限制

任务完成率和工具成功率已经达到可演示水平，但部分案例仍可能出现引用覆盖率不足或目标年份数据不足。没有配置独立 Judge 时，系统只执行规则校验并把 Claim 支持率标记为 `unavailable`，不会伪造质量数字。
