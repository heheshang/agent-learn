//! # 22-agent-async —— 手撕 **异步处理**（并发工具执行 + 限流 + 线程池轨道）
//!
//! 这个项目回答一个问题：**Agent 里的 I/O 密集型操作，怎么改成异步？**
//!
//! 参考资料：
//!
//! - 文档：[FastAPI Async](https://fastapi.tiangolo.com/async/)
//!   —— 「并发 vs 并行」「In a hurry?（`async def` vs `def`）」
//! - 文档：[LangChain Models](https://docs.langchain.com/oss/python/langchain/models)
//!   —— `invoke` / `stream` / `batch` / `batch_as_completed` / `max_concurrency`
//!
//! ## 核心产出
//!
//! 一个**工具并发执行**的 Agent：
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │  模型一轮返回 N 个工具调用                                │
//! │            ↓                                             │
//! │  ★ 并发执行（不再是串行的 for 循环）                       │
//! │  ┌─────────┬─────────┬─────────┐                        │
//! │  │ call1   │ call2   │ call3   │  ← 同时跑              │
//! │  └────┬────┴────┬────┴────┬────┘                        │
//! │       │         │         │                              │
//! │  限流？Bounded(n)  ← 防止打爆上游                        │
//! │       ↓         ↓         ↓                              │
//! │  ⚠️ 按 call_id 还原顺序 → 每个 call_id 恰好一条回执       │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 四种并发模式（对应 LangChain）
//!
//! | 模式 | Rust 原语 | LangChain 对应 | 结果顺序 |
//! |---|---|---|---|
//! | 串行 | `for` + `.await` | （手写循环） | 输入顺序 |
//! | 全并发保序 | `futures::future::join_all` | `batch()` | **输入顺序** |
//! | 全并发乱序 | `FuturesUnordered` | `batch_as_completed()` | **完成顺序** |
//! | 限流并发 | `StreamExt::buffer_unordered(n)` | `max_concurrency=n` | 完成顺序 |
//!
//! ## ⚠️ 六个实测踩出来的坑
//!
//! | 坑 | 现象 | 修法 |
//! |---|---|---|
//! | **1** | 并发执行后结果乱序，回执跟着乱 | 按 `index` **还原输入顺序**再生成回执（16 的坑 1：少一条就 400） |
//! | **2** | 同步阻塞的工具塞进 `async` → **整个事件循环卡住** | 阻塞工具走 `spawn_blocking`（FastAPI 的 `def` → 线程池） |
//! | **3** | `buffer_unordered(0)` → **请求永远挂住** | 并发上限 clamp 到 ≥ 1（配置层 + 执行层各挡一次） |
//! | **4** | 以为「并发 = 更快」→ CPU 密集任务并发**零收益** | 澄清**并发 ≠ 并行**：并发提升的是「等待」的吞吐 |
//! | **5** | **致命错误被降级成观察**（模型转而编答案） | 任务结果带**原始错误对象** → Fatal 终止循环 |
//! | **6** | **重试次数被写死成 1**（`retried_calls` 恒为 0） | 任务结果带 `attempts`，搬运链上全程保留 |
//!
//! ⚠️ **坑 5 与坑 6 是同一个根因的两面** ——
//! 任务输出类型曾是 `Result<String, ToolError>`，
//! 它把「重试次数」和「错误类别」丢在了并发层。
//! **语义字段必须在搬运链上全程保留。**
//!
//! ## 三条执行轨道
//!
//! | 轨道 | FastAPI | Rust | 本项目的样本 |
//! |---|---|---|---|
//! | 异步 | `async def` | `async fn` + `.await` | `get_weather`（真 HTTP） |
//! | 线程池 | `def` | `spawn_blocking` | `slow_report`（真阻塞） |
//! | CPU | `def` + 多进程 | `rayon` | `calculate`（纯计算） |
//!
//! ## 复用关系（继承了前面所有项目的坑）
//!
//! | 来源 | 复用什么 | 继承的坑 |
//! |---|---|---|
//! | [`agent_18`] | 错误三分类 + 中间件 + 工具注册表 | 「每个 call_id 一条回执」「Fatal 不降级」「Business 不重试」 |
//! | [`agent_16`] | `ToolSpec`、参数规整 | 「Schema 是对模型的承诺」 |
//! | [`agent_14`] | 天气工具（真 HTTP）+ Schema 校验器 | 「参数错了别发请求」 |
//! | [`agent_13`] | 算术求值器 | 「统计数字不走模型心算」 |
//!
//! ⚠️ **「继承」不等于「自动生效」** ——
//! 本项目曾**声称**继承了 18 的「Fatal 不降级」，但因为并发层
//! 把错误类别丢在了半路，这条其实**没生效**（坑 5）。
//! **复用类型只是第一步，还要保证语义字段在搬运链上全程保留。**
//!
//! ## 模块划分
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`concurrency`] | **核心**：四种并发模式 + 限流 + 报告 |
//! | [`tools`] | 三条执行轨道的工具 + 适配器（翻译成 18 的 trait） |
//! | [`agent`] | 并发执行工具调用的 Agent 循环 |
//! | [`bench`] | 并发对照实验（串行 / 全并发 / 乱序 / 限流） |
//! | [`prompt`] | 教模型**怎么读并行工具结果** |
//! | [`api`] | HTTP 接口 |
//! | [`table`] | 终端表格（按**显示宽度**补空格） |

pub mod agent;
pub mod api;
pub mod bench;
pub mod concurrency;
pub mod config;
pub mod error;
pub mod llm;
pub mod prompt;
pub mod table;
pub mod tools;

pub use agent::{Agent, AgentConfig, AgentRun, StepRecord, ToolCallRecord};
pub use concurrency::{BatchReport, ExecutionMode, Task, TaskOutput, TaskResult};
pub use config::AppConfig;
pub use error::{AppError, AppResult, ErrorCode, ErrorKind, ToolError};
pub use llm::{FunctionLlm, GenaiFunctionLlm, LlmStep, ScriptedLlm, ToolChoice};
pub use tools::{AsyncTool, ToolContext};
