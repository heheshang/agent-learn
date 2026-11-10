//! # 23-agent-batch —— 手撕 **批处理优化**（Batching）
//!
//! 这个项目回答一个问题：**Embedding 和 Reranker 怎么做批处理？**
//!
//! 参考资料：
//!
//! - 文档：[LangChain Models](https://docs.langchain.com/oss/python/langchain/models#batch)
//!   —— `batch()` / `batch_as_completed()` / `max_concurrency`
//!
//! ## 核心产出
//!
//! 三档批策略 + 一套能量化收益的对照实验：
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │  N 条文本                                            │
//! │            ↓                                         │
//! │  ★ 批策略（唯一变量）                                │
//! │  ┌──────────────┬──────────────┬──────────────┐      │
//! │  │ A 逐条       │ B 固定批     │ C 长度分桶   │      │
//! │  │  N 次前向    │ N/B 次前向  │ N/B 次前向  │      │
//! │  └──────┬───────┴──────┬───────┴──────┬───────┘      │
//! │         ↓              ↓              ↓              │
//! │  固定开销付 N 次   固定开销付 N/B 次  且 padding 浪费更少 │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 三档批策略
//!
//! | 策略 | 做法 | 对应 LangChain |
//! |---|---|---|
//! | `sequential` | 逐条调用（每条一次前向） | （基线） |
//! | `fixed` | 每 N 条一次前向 | `batch()` |
//! | `bucketed` | 按长度分桶后定批 | `batch()` + 自行分桶 |
//!
//! ## ⚠️ 五个实测踩出来的坑
//!
//! | 坑 | 现象 | 修法 |
//! |---|---|---|
//! | **1** | **只看总耗时** → 以为批处理「全面更快」 | 同时报吞吐**和单条延迟**（批处理会**抬高**单条延迟） |
//! | **2** | 固定批里一条长文本拖慢同批所有短文本（**padding 浪费**） | 长度分桶（`bucketed`） |
//! | **3** | 分桶排序打乱了输入顺序 → 结果**静默错位** | 每条带原始下标，返回前**还原到原位** |
//! | **4** | 攒批没有背压 → 请求被无限期压住 | **三个闸门**：最大批大小 / 最大等待 / 最大队列长度 |
//! | **5** | 远程批量**不一定更快**（上游可能内部串行） | 远程也做「逐条 HTTP vs 批量 HTTP」对照，用数字说话 |
//!
//! ## ⚠️ 一个必须先讲清的区分
//!
//! **LangChain 的 `batch()` 与 provider 的 Batch API 是两回事。**
//!
//! - LangChain 的 `batch()`：**客户端并行**发 N 个独立请求
//! - OpenAI / Anthropic 的 Batch API：**服务端**排队处理，通常**更便宜但更慢**
//!
//! 本项目实现的是**前者的极端形态**：把 N 条**合并进一次前向**
//! （本地 ONNX）或**合并进一次 HTTP 请求**（远程服务）。
//!
//! ## 复用关系（继承了前面所有项目的坑）
//!
//! | 来源 | 复用什么 | 继承的坑 |
//! |---|---|---|
//! | [`rag_05`] | `Embedder` trait / `LocalEmbedder` / 模型缓存目录 | 「缓存目录必须绝对路径」 |
//! | [`rag_08`] | `Reranker` trait / `RerankDocument` / `LocalCrossEncoder` | 「推理走 spawn_blocking」 |
//!
//! ## 模块划分
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`batching`] | **核心**：批策略 + 分桶 + 背压三闸门 |
//! | [`embedder`] | 批量嵌入（三档策略） |
//! | [`reranker`] | 批量重排（三档策略） |
//! | [`remote`] | 远程 HTTP 批处理后端 + 对照实验 |
//! | [`metrics`] | 度量（吞吐 / 延迟 / 百分位） |
//! | [`bench`] | ★ 对照实验 |
//! | [`pipeline`] | 批量索引 + 批量检索端到端链路 |
//! | [`api`] | HTTP 接口 |
//! | [`table`] | 终端表格（按**显示宽度**补空格） |
//! | [`testkit`] | 测试替身（零网络、零模型、确定性） |

pub mod api;
pub mod batching;
pub mod bench;
pub mod config;
pub mod embedder;
pub mod error;
pub mod metrics;
pub mod padding;
pub mod pipeline;
pub mod remote;
pub mod reranker;
pub mod sweep;
pub mod table;
pub mod testkit;

pub use batching::{
    BatchAccumulator, BatchConfig, BatchInfo, BatchReport, BatchStrategy, bucket_by_length,
    chunk_fixed, estimate_tokens,
};
pub use config::AppConfig;
pub use embedder::BatchEmbedder;
pub use error::{AppError, AppResult};
pub use metrics::{BatchStats, mean, percentile, stddev};
pub use padding::{PaddingComparison, PaddingReport, compare_padding};
pub use pipeline::{InMemoryVectorStore, IndexReport, SearchReport, build_index, search};
pub use remote::{OllamaEmbedder, RemoteCompareReport, RemoteEmbedder, compare_remote};
pub use reranker::BatchReranker;
pub use sweep::{SweepPoint, SweepReport, sweep_batch_sizes};
