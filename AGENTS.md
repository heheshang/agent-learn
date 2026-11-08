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

- `agent-learn` 是单一 Cargo workspace，成员为 `01-axum` … `12-rag-summary`（包名 `axum-01`、`rig-02`、`genai-03`、`rag-04` … `rag-12`），从 HTTP 基础到 RAG 全链路全部手写实现，不依赖 LangChain/LlamaIndex。
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
