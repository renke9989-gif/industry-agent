# 简历项目描述（可直接使用）

## 行业研究 Agent｜多智能体联网研究系统（导师合作项目，2026-07～2026-09）

- **项目描述：** 面向创业构思、企业研究、经营诊断和工厂能力扩展，搭建“访谈澄清—联网检索—证据校验—决策报告”闭环，避免通用模板和无依据数字。
- **技术栈与后端：** Python、LangGraph、MCP、Tool Calling、Starlette/SSE、Pydantic、MySQL、Redis Streams、Uvicorn、Docker；生产模式使用 MySQL Run Store、Redis Streams 和独立 Worker，SQLite 仅用于 LangGraph Checkpoint 与本地开发模式。
- **工作流与路由：** 基于 LangGraph 实现 Planner（Supervisor）–Worker 和 Skill Router，按场景动态选择研究维度；支持访谈追问承接、会话恢复和范围明确问题的直接回答。
- **证据链：** 通过 MCP/Direct Provider 搜索并读取网页，将 URL、摘录、年份、地区和可信度建模为 Evidence，校验 Claim 与 Citation 绑定；校验失败的结论降级为证据不足。
- **私域知识检索：** 增加可选的 Markdown/TXT/JSON/CSV/PDF 文档解析与分块，使用 SQLite 保存元数据、ChromaDB + SentenceTransformer 建立向量索引，并结合 TF-IDF 混合召回与可选 CrossEncoder 重排，将命中文档片段及路径/页码注入分析师上下文。
- **可靠执行与记忆边界：** 生产模式以 MySQL 保存 Run 生命周期、幂等、Lease 和 fencing token，以 Redis Streams + Consumer Group + ACK + Pending 重领实现异步执行，由独立 Worker 处理任务；SQLite 仅承载 LangGraph Checkpoint，保存对话、访谈进度和 Agent 状态；Evidence、报告和 Trace 由 Artifact Store 按 run 原子写入，User Memory 只保存用户明确确认的长期事实。本地开发仍可切换为 SQLite + 进程内执行。
- **质量验证：** 构建 68 个 Python 回归测试、38 个流程 Harness、18 个可靠性故障场景和 12 个 Skill Harness，覆盖重复提交、断线恢复、Worker 崩溃、过期写入和非法 Blueprint；完成单 Agent / 多 Agent / Evidence 消融实验。
- **在线基线：** 10 个行业案例各运行 30 次，共完成 300 次 Direct Provider 联网评测；流程完成率 100%、URL 有效率 100%、工具成功率 95.9%、引用覆盖率 94.4%，平均耗时 28.7 秒、P95 35.8 秒；严格质量门槛通过率 71%。

GitHub：待发布仓库｜Demo：待发布演示视频或部署链接｜架构图：[architecture.svg](architecture.svg)
