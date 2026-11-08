# agent-learn — Agent Memory

## Learned User Preferences

- 始终使用简体中文交流，代码注释与文档也用中文撰写。
- 偏好「直接编码」工作流：在代码里创建、调试、迭代，而不是只给方案说明。
- 每个学习项目都要配套一份详尽的 README.md，包含架构图、流程图、时序图、知识地图和「技术参考速查」表；内容必须对着源码写，不接受泛泛而谈。
- Mermaid 图必须先经脚本校验通过再提交；图只写在 README.md 里，不额外产出独立文档文件。
- Cursor 的 Markdown 预览不支持 `mindmap`，知识地图改用 `graph LR` + 嵌套 `subgraph` 表达。
- 每个项目配一个 Makefile（help/build/run/test/lint/fmt/clean + 项目专属目标），根目录再提供一个 workspace 级统一入口。
- 学习路线按编号目录渐进推进，每个目录一个 Cargo 成员包，README 需说明实现要点与踩坑。
- 依赖版本统一在根 `Cargo.toml` 管理，成员 crate 只写 `{ workspace = true }` 引用。
- 关注仓库体积：主动排除构建产物、模型缓存、容器数据等可再生文件。
- 需求常附带参考文档/论文链接，实现需对齐指定教程或论文的算法口径。
- 指标或评估异常时，希望给出修复前后可复现的差值对比，而不是只说「已修复」。

## Learned Workspace Facts

- 全仓库日志统一走 workspace 成员 crate `00-logging`（`agent-logging`）：`init(service)` 一次性初始化 subscriber、`trace_layer()` 提供 HTTP 请求追踪（`request_id`/`method`/`path`/`status`/`elapsed_ms`）、`stage_guard(name)` 做阶段耗时守卫，并再导出 `info!`/`debug!`/`warn!`/`error!`/`trace!`。所有 HTTP 项目的路由层都用 `agent_logging::trace_layer()` 替换 tower 的通用 `TraceLayer`。
- 日志三条硬约定：① **4xx 用 `warn!`、5xx 用 `error!`**（告警不能被 4xx 淹没）；② **降级必须打 `warn!`**（静默降级等于埋雷）；③ **业务逻辑里不打 `println!`**（没法按模块关掉，debug 噪音会永久留在生产路径）。过滤统一用 `RUST_LOG`（`EnvFilter` 语法，默认级别 `info`）。
- `agent-learn` 是单一 Cargo workspace，成员为 `00-logging`（`agent-logging`）与 `01-axum` … `14-agent-tool`（包名 `axum-01`、`rig-02`、`genai-03`、`rag-04` … `rag-12`、`agent-13`、`agent-14`），从 HTTP 基础到 RAG 全链路再到 Agent 核心概念与自定义工具开发全部手写实现，不依赖 LangChain/LlamaIndex。
- 根 `Cargo.toml` 的 `[workspace.dependencies]` 是依赖版本与 feature 的唯一来源；成员 crate 仅以 `{ workspace = true }` 引用。
- 整仓是一个 git 仓库（根目录 `git init -b main`），子项目不再有独立 `.git` / `Cargo.lock` / `.gitignore`；04-rag 合并前的历史保留在 `.legacy-git-history/` 的 git bundle 中。
- 构建产物只在根 `/target` 生成；`rust-toolchain.toml` 固定 1.99.1，clippy 以 `-D warnings` 零警告为门槛。
- 根 Makefile 提供 `make help/build/test/lint`、单项目快捷目标与 `serve-*` / `demo-*` / `compare-*` 运维目标；每个子项目也有自己的 Makefile。
- 每份 README.md 统一包含架构全景、流程图、时序图、知识地图与技术参考速查，全部 mermaid 化并对着源码撰写。
- Mermaid 图需同时兼容 GitHub（v10）与 Cursor 预览（其内置 `beautiful-mermaid` 渲染器不支持 `mindmap`，仅支持 graph/flowchart、sequenceDiagram、classDiagram、erDiagram、xychart、stateDiagram）；`scripts/check-mermaid.sh` + `make check-mermaid` 做批量校验。
- `.gitignore` 与 `.cursorignore` 按「可再生性」分层组织并有意保持不一致：`/target`、`.fastembed_cache/`、`.models/`、`*/index/`、`10-rag-milvus/docker/volumes/`、`09-rag-eval/reports/` 被忽略；`docker-compose.yml` 两边都保留。
- LLM 调用统一走 genai 的 DeepSeek 适配器；推理模型的 reasoning token 与 `max_tokens` 共用预算，LLM-as-a-judge 必须关闭思考并调大 `max_tokens`，否则会拿到空响应或被截断的 JSON。
- `09-rag-eval` 提供 22 条标注样本的评估集，指标分确定性（recall@k、nDCG、MRR、token F1）与 judge 类（faithfulness、relevancy、correctness），报告含基线与优化后差值及 ±5% 噪声带。
- `10-rag-milvus` 用 Docker 部署 Milvus standalone（etcd/minio/milvus 三容器）并与 Qdrant 对照，两库同题重合度实测 1.00；集成测试 10 项全通过，clippy 零警告。
- Milvus SDK 有两个必踩的坑：① `ConsistencyLevel::Bounded` 被 SDK 映射成字面 guarantee_ts=2，写完立刻查只能读到 1~2 条，必须用 `Strong`；② 主键若用自增序号，重复灌库会让 `count(*)` 线性膨胀（11 → 55），必须用 `stable_pk`（SHA-1 → 63 位正整数）从 `chunk_id` 稳定派生，Qdrant 侧同理派生 UUIDv5（其字符串 point id 必须是合法 UUID）。
- `10-rag-milvus` 集成测试用「tag + pid」拼 collection 名，因为 Rust 测试默认并行，共用 collection 会互相删除而随机失败；连不上后端时跳过，设 `RAG10_REQUIRE_BACKEND=1` 则直接失败。
- `11-rag-advance` 用 Unstructured/MinerU 处理含表格与图片的复杂 PDF，做表格几何还原、图片抽取与结构感知分块。
- `12-rag-summary` 把 04~11 的能力集成成一个摘要式 RAG 服务（端口 3050，collection `rag_summary_chunks`）：多问法改写 → Milvus 服务端混合检索 → 交叉编码器重排 → 抽取式上下文压缩 → LLM 生成 → 引用抽取；核心是**五档消融阶梯**（dense/keyword/hybrid/hybrid_rerank/full），相邻档位差值可归因。
- `12-rag-summary` 的三条硬约束：① 服务端混合检索返回的是**融合后的 RRF 分**，客户端拿不到 dense/BM25 分项，所以 `Hit.dense_score`/`bm25_score` 必须填 `None`，`min_score` 阈值也只对 `dense`/`client_fusion` 生效；② 跨问法 RRF 的排名必须是**路内排名**（用全局下标会让多路融合退化成「第一路说了算」），有回归测试 `cross_query_fusion_uses_per_route_ranks` 锁住；③ 客户端 BM25 用块正文 `text` 而非 `embedded_text`（含标题），否则标题词会主导打分。
- `12-rag-summary` 把检索层抽成 `Retriever` trait（`search<'a>(...) -> Pin<Box<dyn Future<...>>>`，因为 `async fn` 不是 dyn 兼容），测试注入 `FakeRetriever` 即可离线测 API 层；集成测试 22 项（含 3 项需 Milvus，`RAG12_REQUIRE_BACKEND=1` 强制后端可达）。
- 终端表格混排中文必须按**显示宽度**补空格（汉字占 2 列），`12-rag-summary/src/table.rs` 实现了 `display_width` / `Table`，用 `format!("{:<15}")` 会让中英文混排的列越往右越歪。
- `13-agent-core` 实现 ReAct（arXiv:2210.03629）Agent 循环（端口 3060）：`Thought / Action / Action Input / Observation / Final Answer` 文本协议，工具为 `search`（复用 08 的 BM25 索引，零模型依赖）+ `calculate`（手写算术求值器）；核心产出是**三模式消融**（`direct` 无工具无思考 / `act_only` 有工具无思考 / `react` 全有），三者变量单一，可量化「行动」与「思考」各自的贡献。
- `13-agent-core` 有两个实测踩过的真坑：① **模型会回显整个提示词**再继续，解析器若找**第一个** `Action:` 会匹配到回显里的旧动作 → 上一步工具被重跑、模型真正的答案被丢弃（实测症状是答案变成「平均每轮调用 3 次工具」这种无关内容）；修法是统一用 `rfind` 找**最后一个**标记，回归测试 `echo_does_not_cause_the_previous_action_to_repeat` 锁住。② **few-shot 示例的 Question 不能和真实提问撞车** —— 撞了模型会分不清「我刚查到的 Observation」和「示例里的」，从而照着示例编答案；示例问题必须和真实问题在文本上错开。
- `13-agent-core` 的 `ActOnly` 对照组**必须单独写一份不带 `Thought:` 的 few-shot**（`FEW_SHOT_ACT`），否则「思考」这个变量就不是单一的，消融实验作废。
- `13-agent-core` 复用 `07-query` 的 `Llm` trait 与 `ScriptedLlm`，让整个 Agent 循环能**零网络确定性测试**（单元 52 + 集成 15）；`Direct` 模式不挂工具且提示词里不出现 `Action:` 行（否则模型会编造动作），`ActOnly` 与 `ReAct` 的格式说明行也不同。
- `13-agent-core` 的工具失败**不返回错误**，而是把错误文本当作 `Observation` 喂回模型让它自我纠正（这就是 `AgentError::is_degradable` 只对 `Tool` 错误返回 `true` 的语义）；达到 `max_steps` 后的收尾调用若失败也必须**降级**（用 `?` 直接传播会丢掉前面几步辛苦收集的观察）。
- `14-agent-tool` 实现自定义工具开发（端口 3070 + 内置 MCP 服务器 3071）：核心产出是**查询天气的自定义工具**（`get_weather`，走 Open-Meteo 免 key 接口，两跳「地名 → 经纬度 → 预报」），并集成进 Agent。五个模块：`tool`（Tool trait / ToolRegistry / ToolPolicy）、`schema`（JSON Schema 子集校验器）、`weather`（要联网的工具）、`mcp`（MCP 客户端）、`mcp_server`（内置 MCP 服务器）。
- `14-agent-tool` 的三条核心原则：① **工具失败不是请求失败** —— 错误渲染成 `Observation` 喂回模型让它自我纠正（LangChain 的 `handle_tool_errors` 语义）；只有「用户直接调工具」时错误才变成 HTTP 响应，因为那时用户扮演了模型角色。② **参数错了别发请求** —— Schema 本地校验，错误信息必须「可执行」（指出哪个参数、期望什么、实际是什么）。③ **工具来源与使用方式解耦** —— 本地 `get_weather` 与 MCP `get_weather` 在 Agent 眼里完全一样（都只是 `ToolSpec`）；内置 MCP 服务器与本地工具**共用同一份 `Tool::call` 实现**，语义不可能漂移。
- `14-agent-tool` 的工具 trait 必须是 **async**（13 的工具是同步的，因为它只做内存操作；本项目要等外部 HTTP）。`async fn` 不是 dyn 兼容，所以用 `Pin<Box<dyn Future + Send + '_>>`（`ToolFuture`）显式装箱 —— 和 07 的 `LlmFuture` 同理。
- `14-agent-tool` 实测踩过四个坑：① **真实对话与 few-shot 示例不能长得一样** —— 真实问题写成 `Question: {q}`（和示例同格式）时，模型第 2 步会把自己上一轮的轨迹误认成第三个示例，于是「顺着示例格式」自己编了个 `Question: 深圳...` 去调工具（深圳从没出现在提示词里）；修法是显式标注「以下是真实对话（不是示例）」+ 把轨迹标注成「已完成的步骤」，回归测试 `real_dialogue_is_clearly_separated_from_examples`。这和 13 的坑 2 是同一坑的两面。② **MCP 的 camelCase 字段必须 `rename`**（`inputSchema` / `isError`），不写就静默拿到默认值 —— 单元测试用 Rust 结构体构造数据会绕过 JSON 反序列化，所以**MCP 测试必须走真实 HTTP 往返**才能发现。③ `#[serde(rename_all = "snake_case")]` 会把 `ReAct` 变成 `re_act`，和 CLI 名 `react` 对不上（编译期无警告），三处（CLI / 环境变量 / JSON）必须显式 `rename` 对齐。④ 工具参数错 / 未知工具应返回 **400** 而非 500 —— 500 会让调用方以为服务端坏了。
- `14-agent-tool` 的 MCP 实现覆盖 Streamable HTTP 的三个方法（`initialize` / `tools/list` / `tools/call`），会话 ID 在 `initialize` 的**响应头** `Mcp-Session-Id` 里；客户端遇 409/404 会清空会话自动重新握手；`parse_json_rpc_response` 同时兼容 `application/json` 和 `text/event-stream`（SSE）两种响应。
- `14-agent-tool` 复用 `13-agent-core` 的 `parse_react_output`（含「找最后一个标记」的修复）—— 解析器是 Agent 系统里最容易出错的部分，重写等于把坑再踩一次；测试用 `ScriptedLlm` + `FakeTool` 实现零网络确定性（单元 80 + 集成 29），HTTP API 层用 `tower::ServiceExt::oneshot` 直接调路由（不开端口）。
