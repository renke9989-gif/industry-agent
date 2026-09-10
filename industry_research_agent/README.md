# 行业研究 Agent

[架构图](docs/architecture.svg) · [在线评测基线（300 次）](docs/ONLINE_BASELINE_2026-09-07.md) · [Demo 手册](docs/DEMO_RUNBOOK.md) · [简历描述](docs/RESUME_PROJECT.md)

基于问卷式需求澄清、多智能体协作、MCP 联网检索、结构化证据链和引用验证的行业研究系统。项目目标不是生成通用行业报告，而是先理解用户的阶段、资源、预算和决策目标，再输出可追溯的针对性建议。当前基线已覆盖从用户输入、Skill 路由、访谈澄清、联网检索到证据约束报告的完整闭环。

## 面试可验证结果

| 项目 | 当前基线 |
|---|---:|
| Python 回归测试 | 68/68 |
| 流程 Harness | 38/38 |
| 可靠性故障 Harness | 18/18 |
| Skill 路由 Harness | 12/12 |
| 在线评测 | 10 个案例 × 30 次，共 300 次 |
| 在线任务完成率 | 100% |
| URL 有效率 | 100% |
| 工具成功率 | 95.9% |
| 平均耗时 | 28.7 秒 |
| P95 延迟 | 35.8 秒 |
| 引用覆盖率 | 94.4% |
| 严格质量门槛通过率 | 71%（213/300） |
| 5 并发联网任务 | 5/5 成功 |

在线基线使用 Direct Provider，运行日期为 2026-09-07。“流程完成率”表示研究链路成功结束；“严格质量门槛通过率”还要求引用覆盖、有效 URL、研究维度和时延同时达标。Claim Judge 和估算成本因未配置独立模型价格，明确标记为 `unavailable`。

## 核心调用链

```text
用户问题
  → Triage：识别构思验证、经营诊断、工厂扩展、企业研究等场景
  → 访谈模式确认：进入访谈（推荐）/ 直接开始调研
  → Interview：只询问会改变结论的高价值问题
  → Supervisor：按研究计划调度分析师
      → 市场/趋势/机会分析师
      → 竞争/风险分析师
      → 商业模式分析师
  → ResearchProvider
      → MCP search / fetch_page
      → MCP 失败时 Direct Provider 降级
      → Pydantic Evidence
  → Claim/引用检查与精简报告
  → SQLite Checkpoint 保存会话状态
  → Run Store 保存任务生命周期与恢复信息
```

可选私域知识库链路：`Markdown/TXT/JSON/CSV（PDF 需安装 pypdf） → 分块 → SQLite 元数据 → ChromaDB 向量索引 + TF-IDF 混合召回 → 可选 CrossEncoder 重排 → 分析师上下文`。默认关闭，使用 `PRIVATE_RAG_ENABLED=true` 开启；文档可通过 `POST /api/private-documents` 建索引，通过 `GET /api/private-documents?q=...` 检索。`PRIVATE_RAG_RETRIEVAL=hybrid` 启用 Embedding 检索，`PRIVATE_RAG_RERANK=true` 启用重排；无向量模型或依赖时自动回退到 TF-IDF。

状态边界保持清晰：Checkpoint 保存当前会话和研究过程；Run Store 保存 queued/running/retrying/succeeded 等执行状态、取消、重试和 Lease；长期记忆只保存用户明确确认的信息，不自动把整段聊天写入长期记忆。

## 访谈模式与研究范围

需要完整研究的请求不会立即搜索。系统先展示“访谈模式”确认卡：

- `进入访谈（推荐）`：根据前序回答动态追问；一般场景最多 6 个高价值问题，工厂扩展和经营诊断等复杂场景最多 8 个。
- `直接开始调研`：跳过访谈，使用已知信息和明确标注的假设开始研究。

访谈轮次、用户对话轮次和 Agent 执行步数分别计数。访谈不会消耗分析师工具预算；只有分析师实际执行搜索、网页读取和证据抽取才计入 `agent_step_count`。

Supervisor 先识别场景，再使用确定性规则生成最低研究范围：

| 场景 | 默认研究维度 |
|---|---|
| 创业构思 | 市场、竞争、商业模式、风险、机会 |
| 筹备落地 | 竞争、商业模式、风险、机会 |
| 经营诊断 | 竞争、商业模式、风险 |
| 工厂或产品扩展 | 竞争、商业模式、风险、机会 |
| 企业研究 | 市场、竞争、趋势、机会、风险 |
| 方案比较 | 市场、竞争、商业模式、风险 |
| 范围明确的单点问题 | 不启动完整多维报告，直接联网回答 |

计划内维度必须执行到 `complete` 或 `insufficient`。侧边栏中的 `4/5` 表示证据覆盖度，不是模型准确率；计划内尚未执行的维度显示“未完成”。

用户后续明确更新地区、预算或目标时，本轮新条件优先。例如“上海”会覆盖此前的“新一线/二线城市”；条件更新不得被解释成概念矛盾。

## 已实现能力

- LangGraph Supervisor–Worker 编排。
- 根据场景动态生成访谈问题，不固定套用六维模板。
- 使用版本化 DecisionTemplate 管理场景目的、最低研究维度、必问题目和结束规则；模板是决策流程，不是本地行业知识库。
- 低置信度路由先追问研究对象，不把“帮我看看这个”等模糊输入直接送入搜索。
- 新研究会清理旧任务的 Evidence、工具事件和访谈完成标记，避免跨场景串状态。
- 同一会话的省略式追问会承接最近问题主题；“大厂/中厂”等求职口语会规范化为科技公司语境，避免误检索为制造工厂。
- 数字化创业使用产品方向、目标客户、核心资源、团队阶段等专用问题，不套用门店选址问卷。
- MCP 搜索与网页读取，Direct Provider 自动降级。
- Evidence、EvidenceExtraction、ReportClaim 等 Pydantic 合同。
- 报告关键事实使用可点击 Evidence ID。
- 正式报告建立 `[U#] 用户条件 → [E#] 外部证据 → [R#] 行动建议` 追溯链；缺少条件、证据或验证标准的建议会被拦截或降级。
- SQLite Checkpoint，服务重启后恢复会话和前端历史。
- 独立的结构化长期记忆：只保存用户明确确认的事实，支持跨会话读取和删除。
- 节点级 RetryPolicy，瞬时失败最多重试一次。
- 用户取消研究，保留已完成证据和状态。
- 同步图运行在线程中，不阻塞 Web 事件循环；默认支持 5 个并发研究任务。
- 同一 session 防止重复提交，避免状态竞争和重复计费。
- 节点耗时、工具调用、Token 和估算成本追踪。
- 68 个 Python 回归测试、38 个流程 Harness 案例、18 个可靠性故障 Harness 和在线评测脚本。
- Single/Multi Agent × Evidence on/off 四版本消融实验。
- Durable Run Store：任务状态、去重、取消、Worker Lease、heartbeat、fencing token 和 Artifact 持久化。
- 结构化 Workflow Blueprint 与 Compiler：仅允许映射到已注册的 LangGraph 节点，不执行模型生成代码。
- Context Token Budget：在保留用户条件、Evidence ID 和冲突的前提下压缩工具上下文。
- Skill Registry：6 个白名单 Skill、规则优先路由、可选 LLM 二次判断、Skill Trace 和节点权限校验。

## 一键启动

```powershell
cd D:\AI学习\industry-agent\industry_research_agent
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\run_web.bat
```

打开 `http://localhost:8000`。健康检查位于 `http://localhost:8000/health`。
就绪检查位于 `http://localhost:8000/ready`，会额外返回当前并发任务数和并发上限。

当健康检查显示以下内容时，会话持久化已启用：

```json
{"checkpoint_backend": "sqlite", "checkpoint_warning": ""}
```

## 验证

```powershell
python -m pytest -q tests
python -m eval.eval_harness
python -m eval.reliability_harness
python -m eval.skill_harness
python -m eval.routing_eval
```

当前本地验收结果：Python 回归测试 `68/68`，流程 Harness `38/38`，可靠性故障 Harness `18/18`，Skill Harness `12/12`。

GitHub 发布前请查看 [发布清单](docs/GITHUB_RELEASE_CHECKLIST.md)。仓库地址和公网 Demo 地址需要在实际创建仓库、部署服务后补充，不能用本地地址冒充公开链接。

Skill 路由和上下文指标的离线验收：Skill Harness `12/12`，规则路由准确率基线 `100%`，平均路由延迟约 `0.1ms`。路由评测命令会把未配置的 LLM 二次路由标记为 `unavailable`，不会把规则结果冒充为 LLM 结果。

真实 API 评测或面试演示前检查：

```powershell
python -m scripts.preflight
```

在线评测会产生真实 LLM 与搜索费用：

```powershell
python -m eval.online_eval --case coffee_hangzhou --runs 3
```

2026-09-04 Direct Provider 基线已实际运行 10 个案例 × 3 次，共 30 次：任务完成率 `100%`，URL 有效率 `100%`，工具成功率 `97.2%`，Skill 初始路由准确率 `100%`，平均耗时 `28.7s`，P95 单案例耗时低于 `90s`。原始引用覆盖率因旧口径把行动计划数字纳入统计而被低估，修正后的评测会排除“未来 7 天、行动条目、验证标准”等非外部 Claim；部分报告仍可能因为 Claim/Citation 校验不足而不能作为正式报告。目标年份证据平均占比约 `45%`。由于未配置独立 Judge 和真实模型价格，`claim_entailment_rate` 与 `estimated_cost` 标记为 `unavailable`，不伪造数值。原始结果保存在 `eval/results/2026-09-04-summary.json`。

消融实验：

```powershell
python -m eval.ablation --input "想在杭州开咖啡店，预算50万" --runs 3
```

也可以先运行最关键的访谈消融子集，降低真实 API 成本：

```powershell
python -m eval.ablation --case coffee_hangzhou --runs 1 `
  --variant multi_agent_evidence `
  --variant multi_agent_interview_evidence
```

同一模型的普通直接回答与完整 Agent 对比：

```powershell
python -m eval.direct_vs_agent --case coffee_hangzhou --runs 3
```

访谈消融使用 `eval/interview_profiles.json` 中的固定用户条件。当前消融包含六个版本：

1. `single_agent`
2. `single_agent_evidence`
3. `multi_agent`
4. `multi_agent_evidence`
5. `multi_agent_interview`
6. `multi_agent_interview_evidence`

其中 3/4 对比 Evidence 作用，4/6 对比访谈条件作用，5/6 对比同一访谈条件下 Evidence 的作用。所有版本使用同一模型、输入和 Provider；实验不预设 Agent 必然获胜。

可靠执行层当前使用本地 SQLite，Run 状态为 `queued → running → retrying → succeeded/failed/timed_out/cancelled`。Run Store 与 LangGraph Checkpoint 分离：Checkpoint 保存 Agent 状态，Run Store 保存执行生命周期，Artifact Store 保存报告和运行摘要。Worker 使用 Lease 和 fencing token 防止过期 Worker 覆盖新结果；服务启动时会恢复有 Checkpoint 的过期任务。生产迁移时可将 Run Store 迁移到 PostgreSQL、将队列迁移到 Redis Streams，但这些组件不是当前项目的运行依赖。

故障注入 Harness：

```powershell
python -m eval.reliability_harness
```

该 Harness 覆盖搜索/网页超时、MCP 退出与格式错误、Provider 降级、临时错误重试、认证错误不重试、重复提交、断线、取消、Worker 崩溃、Lease 过期、fencing 拒绝旧写入、重启恢复、Checkpoint 缺失、Artifact 失败和非法 Blueprint 等 18 个场景。

Skill 路由使用 `skill_registry.py` 中的白名单，不执行模型生成代码。默认 `RuleIntentClassifier` 复用现有 triage；只有设置 `ROUTING_LLM_ENABLED=true` 且规则置信度处于 0.5～0.8 时才调用可选的 LLM 二次判断。可以使用 `python -m eval.routing_eval` 对比规则路由和可选 LLM 路由；未配置 LLM 时明确标记为 unavailable，不伪造对比结果。

启动 Web 服务后执行 5 并发验收：

```powershell
python -m eval.concurrency_eval --count 5
```

真实联网并发验证：使用 MCP Provider 启动本地 Web 服务后运行 `python -m eval.concurrency_eval --count 5 --mode live`，5/5 任务成功，Session 全部唯一，平均耗时约 `38.7s`，P95 约 `38.9s`。Smoke 模式不消耗真实 API 额度；Live 模式会产生费用。

## 可观测指标

`GET /api/sessions/{session_id}` 返回：

- `run_metrics.duration_ms`
- `run_metrics.tool_call_count`
- `run_metrics.tool_failure_count`
- `run_metrics.search_call_count`
- `run_metrics.fetch_page_count`
- `run_metrics.mcp_failure_count`
- `run_metrics.retry_count`
- `run_metrics.cost_by_dimension`
- `llm_usage.input_tokens`
- `llm_usage.output_tokens`
- `claim_judge_status`
- `claim_entailment_rate`
- `recommendation_trace_status`
- `run_metrics.recommendation_trace_rate`
- `run_metrics.skill_selection_accuracy`
- `run_metrics.routing_confidence`
- `run_metrics.clarification_rate`
- `run_metrics.context_before_tokens`
- `run_metrics.context_after_tokens`
- `run_metrics.context_budget`
- `run_metrics.context_compression_count`
- `run_metrics.context_overflow_count`
- `run_metrics.context_compression_ratio`
- `trace_events`
- Evidence 和维度完成状态

`GET /api/sessions` 返回本机最近的调研会话目录。网页端的“历史对话”可以恢复消息、访谈进度、Evidence 和报告；即使浏览器中的最后会话 ID 丢失，也会从 `.data/sessions.sqlite3` 自动选择最近会话。聊天正文与 Agent State 仍保存在 LangGraph 的 `.data/checkpoints.sqlite3`，会话目录只保存标题、时间和 `session_id`。

配置 `LLM_INPUT_COST_PER_1K` 和 `LLM_OUTPUT_COST_PER_1K` 后会记录估算成本。
每次完整研究还会追加一条脱敏 JSONL 记录到 `.data/runtime.jsonl`。

Docker Compose 会把 SQLite Checkpoint 和运行日志挂载到命名卷：

```powershell
docker compose up --build
```

首次使用 Docker 时，先将 `.env.docker.example` 复制为 `.env` 并填写 API Key；`.env` 不提交到 Git。

## 当前限制

- SQLite 适用于本地演示；生产环境应迁移到 PostgreSQL Checkpointer。
- 目前没有用户认证、租户隔离和分布式任务队列。
- 长期记忆目前使用浏览器生成的本地 user_id，不等同于生产身份认证。
- 同步网页工具只能在当前工具调用结束后响应取消，无法强制杀死正在执行的 HTTP 请求。
- Claim 默认执行规则校验；配置 `EVAL_JUDGE_API_KEY`、`EVAL_JUDGE_MODEL` 和可选的 `EVAL_JUDGE_BASE_URL` 后启用独立 LLM Judge。未配置时指标明确标记为 `unavailable`，不伪造支持率。
- 当前仍缺少完整人工 Claim 标注平台，这属于后续生产化工作。
- 在线评测结果依赖搜索供应商、模型版本和运行时间，应记录运行环境后再比较。

## 首次在线基线（2026-08-22）

`coffee_hangzhou` 单次 MCP 运行已完成但未通过验收：耗时 129.8 秒，证据 0 条，工具成功率 0，因此引用覆盖率和时延指标不达标。这个失败样例会保留在 `eval/results/` ，不隐藏搜索工具不可用的问题。

## 面试说明

MCP 的价值不是简单包装 Tavily，而是让搜索、网页读取和未来企业数据源共享统一协议，并允许独立替换、测试和部署。多 Agent 也不假设一定优于单 Agent，项目通过消融实验量化质量、成本和延迟之间的权衡。

## 实际评测结果（2026-08-23）

以下数据是访谈确认、地区覆盖和 Agent 步数修复之前保存的历史基线，用于对比而非代表最新代码的最终成绩。路由变更后应重新运行 10 案例 × 3 次在线评测，再更新最终指标。

10 个在线案例各运行 3 次，共 30 次：30/30 流程成功，15/30 达到当前自动验收标准；平均引用覆盖率 87.6%，URL 有效率 100%，工具成功率 94.2%，平均延迟 28.23 秒。流程可重复完成，但引用覆盖率会随搜索结果和模型输出波动；未通过案例主要是引用覆盖率低于 90%，不是流程崩溃。搜索结果会受供应商可达性、模型版本和运行时间影响，因此结果只作为当前环境基线。

四版本消融实验各运行 3 次，固定输入“想在杭州开咖啡店，预算50万元”：

| 版本 | 平均引用覆盖率 | 平均证据数 | 平均工具成功率 | 平均延迟 | 平均总 Token |
|---|---:|---:|---:|---:|---:|
| Single Agent | — | 0 | 87.5% | 29.42 秒 | 4,675 |
| Single Agent + Evidence | 87.5% | 12 | 95.0% | 24.77 秒 | 13,318 |
| Multi-Agent | — | 0 | 90.0% | 23.72 秒 | 6,678 |
| Multi-Agent + Evidence | 96.3% | 15 | 96.0% | 27.93 秒 | 18,544 |

当前实验结论：Evidence 是提升引用覆盖率和可追溯性的主要因素；Multi-Agent + Evidence 的引用覆盖率最高，但 Token 成本约为 Single Agent 的 4 倍。多 Agent 的价值应解释为维度拆分、工具事件隔离和更完整的研究覆盖，而不是简单宣称一定更快或更便宜。

## 同模型直接回答对比（2026-08-24，单次预跑）

固定 `coffee_hangzhou` 输入和同一份访谈条件，各运行 1 次。该结果只验证评测链路，不替代 3 次重复实验：

| 版本 | 延迟 | 总 Token | Evidence | 引用覆盖率 | 建议追溯率 |
|---|---:|---:|---:|---:|---:|
| 同模型直接回答 | 12.80 秒 | 983 | 0 | 0% | 0% |
| 完整研究 Agent | 79.76 秒 | 19,310 | 9 | 100% | 100% |

这次预跑说明 Agent 用显著更高的延迟和 Token 换取了 Evidence、引用和用户条件追溯。完整研究 Agent 仍因严格 Citation/Claim 校验被标记为 `insufficient`，所以不能据此宣称质量已经全面达标；原始结果保存在 `eval/results/2026-08-24-coffee_hangzhou-direct-vs-agent.json`。
