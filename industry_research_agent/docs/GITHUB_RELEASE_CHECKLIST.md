# GitHub 与 Demo 发布清单

建议仓库名：`industry-research-agent`。

## 创建 GitHub 仓库

在 GitHub 创建空仓库后，从项目根目录执行：

```powershell
cd D:\AI学习\industry-agent
git remote add origin https://github.com/<your-account>/industry-research-agent.git
git add README.md industry_research_agent
git status
git commit -m "feat: evidence-first industry research agent"
git branch -M main
git push -u origin main
```

提交前检查：

- `.env`、`.data/`、`.venv/` 和 `eval/results/` 不在提交列表。
- 没有 API Key、Cookie、完整敏感网页正文和个人信息。
- 根 README 首屏能看到结果、架构图、启动命令和当前限制。
- GitHub Actions 通过 Python 3.11、pytest、流程 Harness、可靠性 Harness 和 Skill Harness。

## Demo

本地演示：

```powershell
cd D:\AI学习\industry-agent\industry_research_agent
.\.venv\Scripts\python.exe -m scripts.preflight
.\scripts\run_web.bat
```

打开 `http://localhost:8000`，按照 [Demo 手册](DEMO_RUNBOOK.md) 演示。

公网 Demo 需要在支持 Python 3.11 的平台配置 LLM 和搜索环境变量。地址必须经过真实访问验证后再写入简历；如果暂时没有稳定部署，优先放 60～90 秒录屏链接，不填写虚构地址。

## 面试官入口顺序

1. 根 README：结果和架构。
2. `docs/architecture.svg`：系统分层。
3. `docs/ONLINE_BASELINE_2026-09-04.md`：指标与限制。
4. `docs/DEMO_RUNBOOK.md`：演示路径。
5. 项目完整 README：API、Harness 和设计取舍。
