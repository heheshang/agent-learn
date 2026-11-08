//! 09-rag-eval —— 手撕 **RAG 评估（Evaluation）** 层。
//!
//! 前面 8 个项目把 RAG 链路**建**起来了：
//!
//! ```text
//! 04-rag  加载 + 分割
//! 05-rag  向量化 + 存储
//! 06-rag  完整 Naive RAG
//! 07-rag  查询改写
//! 08-rag  混合检索 + 重排
//! ```
//!
//! 但每一个项目都在回答同一个问题：**「我这个改动到底有没有用？」**
//!
//! 而 06/07/08 的答案是「看起来有用」—— 靠肉眼看召回结果、
//! 靠感觉判断答案变好了。**这不是工程**。
//!
//! ## 本项目解决什么
//!
//! 把「看起来有用」变成**「可复现的差值」**。
//!
//! ```text
//!                评估集（地面真相）
//!                      │
//!         ┌────────────┼────────────┐
//!         ▼            ▼            ▼
//!     baseline      optimized      optimized
//!    （只走向量）  （混合+重排）  （+更多改动）
//!         │            │            │
//!         └────────────┴────────────┘
//!                      │
//!              指标计算（确定性 + judge）
//!                      │
//!                 对比报告（差值表）
//! ```
//!
//! ## 指标的三个来源
//!
//! | 框架 | 贡献的指标 | 特点 |
//! |---|---|---|
//! | [RAGAs](https://docs.ragas.io/en/latest/index.html) | faithfulness / answer_relevancy / context_relevancy / answer_correctness | LLM-as-a-judge，语义层面 |
//! | [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | recall@k / precision@k / nDCG / MRR / EM / F1 | 检索与答案的量化对比，benchmark 导向 |
//! | [DeepEval](https://github.com/confident-ai/deepeval) | RAGAS 聚合口径、context precision/recall | 与 RAGAs 同源的另一套实现，可交叉验证 |
//!
//! ## 为什么这些指标必须分成两组
//!
//! | | 确定性指标 | LLM-as-a-judge 指标 |
//! |---|---|---|
//! | 例子 | recall@k、nDCG、token F1 | faithfulness、answer relevancy |
//! | 速度 | 毫秒 | 秒 |
//! | 成本 | 零 | 每条样本几分钱 |
//! | 可复现 | ✅ 完全确定 | ⚠️ 有抖动 |
//!
//! 如果只有 judge 指标，「优化后涨了 0.05」这个结论**不可信** ——
//! judge 自身的抖动就有 ±0.05。
//!
//! 所以本项目的策略是：
//!
//! - **用确定性指标锚定「检索准不准」**（地基）
//! - **用 judge 指标衡量「答案好不好」**（补充）
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`dataset`] | 评估集 + 章节 → 块 ID 的映射 | Ground Truth |
//! | [`metrics::deterministic`] | recall@k / nDCG / MRR / token F1 | FlashRAG 风格的量化指标 |
//! | [`metrics::judge`] | faithfulness / relevancy / correctness | RAGAs 风格的 LLM judge |
//! | [`report`] | 报告 + 对比报告 | Eval Report |
//! | [`pipeline`] | 编排：跑系统 → 算指标 → 出报告 | Evaluation Harness |
//! | [`api`] | HTTP 接口 | — |
//!
//! ## 核心认知
//!
//! | 认知 | 说明 |
//! |---|---|
//! | **评估集是地基** | 评估集标错了，后面全是错的 —— 而且**不会报错** |
//! | **检索指标是锚点** | judge 指标有抖动，确定性指标才能证明真实收益 |
//! | **差值要有噪声带** | 小于 5% 的差值不能当成真实收益 |
//! | **失败 ≠ 0 分** | 指标算不出来 ≠ 指标是 0 —— 报告必须分开 |
//! | **裁判不能是选手** | judge 模型和生成模型最好是不同的 |

pub mod api;
pub mod config;
pub mod dataset;
pub mod error;
pub mod metrics;
pub mod pipeline;
pub mod report;

pub use config::EvalConfig;
pub use dataset::{ChunkIndex, EvalCase, EvalDataset};
pub use error::{EvalError, EvalResult};
pub use metrics::{EvalContext, Metric, MetricRegistry, MetricScore};
pub use pipeline::{EvalPipeline, EvalVariant, SystemUnderEval};
pub use report::{ComparisonReport, EvalReport, Evaluator, MetricDelta, Verdict};
