# 面试 Demo 手册

## 一键启动

```powershell
cd D:\AI学习\industry-agent\industry_research_agent
.\.venv\Scripts\python.exe -m scripts.preflight
.\scripts\run_web.bat
```

浏览器打开 `http://localhost:8000`。

## 推荐演示路径（约 90 秒）

1. 输入“我想在杭州开咖啡店，预算 50 万”。
2. 点击“进入访谈（推荐）”，展示动态问题，而不是固定报告模板。
3. 回答目标客户、经营形态和成功标准，展示左侧场景、Skill 和访谈进度。
4. 开始研究，展示 Supervisor → 分析师 → MCP/Direct Provider → Evidence 的过程。
5. 展开一条 Evidence，点击正文中的 `[E#]` 来源链接。
6. 生成报告，展示简洁结论、关键依据、行动建议和下载按钮。
7. 关闭页面后重新打开，点击“历史对话”，展示 SQLite Checkpoint 和 Run Store 恢复。
8. 发送“那上海呢？”或其他追问，展示 `followup_answer` 不会重新生成整份报告。

## 评测演示

```powershell
.\.venv\Scripts\python.exe -m eval.eval_harness
.\.venv\Scripts\python.exe -m eval.reliability_harness
.\.venv\Scripts\python.exe -m eval.skill_harness
.\.venv\Scripts\python.exe -m eval.routing_eval
```

## 发布前检查

- 不提交 `.env`、`.data/`、`.venv/` 和原始评测网页内容。
- README 中的指标必须带日期、Provider 和运行次数。
- 没有独立 Judge 或模型价格时，明确写 `unavailable`。
- GitHub 仓库首页放 README、架构图、Demo 视频和评测基线链接。
