//! # 16-agent-function-calling —— 手撕 **Function Calling**（结构化工具调用）
//!
//! 这个项目回答一个问题：**模型怎么「调用函数」？和 ReAct 的文本协议差在哪？**
//!
//! 参考资料：
//!
//! - 文档：[Function Calling](https://platform.openai.com/docs/guides/function-calling)
//!   —— 工具定义 / 工具调用 / 工具输出三件套，五步调用流程
//! - 指南：[GPT Best Practices](https://platform.openai.com/docs/guides/gpt-best-practices)
//!   —— 提示词结构（Identity / Instructions / Examples）、工具描述怎么写
//!
//! ## 核心产出
//!
//! 一个**真正走 OpenAI 结构化协议**的 Agent：
//!
//! ```text
//! 请求：tools = [{ name, description, parameters }]
//! 响应：tool_calls = [{ id, function: { name, arguments } }]   ← 结构化 JSON
//! 回执：role="tool", tool_call_id=<id>, content=<结果>
//! ```
//!
//! ## 和 13/14/15 的分水岭
//!
//! | 维度 | 13/14/15（ReAct 文本协议） | 16（Function Calling 结构化协议） |
//! |---|---|---|
//! | 工具定义 | 拼进**提示词文本** | 放进 API 的 **`tools` 字段** |
//! | 模型输出 | **纯文本**，靠正则解析 `Action:` | **结构化 JSON**，直接读字段 |
//! | 工具结果回传 | 拼成 `Observation:` 文本 | 一条 **`tool` 角色的消息** |
//! | 并行调用 | 不支持 | **支持**（一次多个 `tool_calls`） |
//! | 参数校验 | 事后校验 | Schema 直接参与模型训练 |
//!
//! ## 四个实测踩出来的坑（本项目最有价值的部分）
//!
//! | 坑 | 现象 | 修法 |
//! |---|---|---|
//! | **1** | 模型返回 2 个 `tool_calls`，你只回 1 条 → **整个请求 400** | 每个 `call_id` 都必须有回执（成功和失败都算） |
//! | **2** | 推理模型不把 `reasoning_content` 回传 → **整个请求 400** | assistant 消息里带上思考内容 |
//! | **3** | `parallel_tool_calls: false` 被 **DeepSeek 忽略** | 并行度只能自己在应用层限制 |
//! | **4** | `strict: true` 被接受但**不强制执行** | 本地 Schema 校验才是真兜底 |
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应 OpenAI 概念 |
//! |---|---|---|
//! | [`tools`] | 工具定义 + strict Schema 改写 | `tools` 参数 / strict mode |
//! | [`tool`] | 工具 trait / 注册表 / 参数校验 / 失败转回执 | 工具执行 + 校验 |
//! | [`functions`] | 三个具体工具（天气 / 知识库 / 计算） | function 定义 |
//! | [`agent`] | **Function Calling 循环** | 五步调用流程 |
//! | [`llm`] | 带工具的 LLM 抽象 | `tool_choice` / `tool_calls` |
//! | [`prompt`] | 提示词组装 | Identity / Instructions / Examples |
//! | [`api`] | HTTP 接口（含 **tool_choice 四档对照**） | — |

pub mod agent;
pub mod api;
pub mod config;
pub mod error;
pub mod functions;
pub mod llm;
pub mod prompt;
pub mod tool;
pub mod tools;

pub use agent::{
    Agent, AgentRun, StepRecord, TokenUsage, ToolCallRecord, build_assistant_message, normalize_usage,
};
pub use config::{AgentConfig, AppConfig, PromptConfig, ToolConfig};
pub use error::{AppError, AppResult, ToolError};
pub use functions::{
    CALCULATE_TOOL_NAME, GET_WEATHER_TOOL_NAME, SEARCH_KNOWLEDGE_TOOL_NAME, default_registry,
};
pub use llm::{
    FunctionLlm, GenaiFunctionLlm, ScriptedFunctionLlm, ScriptedStep, ScriptedToolCall, ToolChoice,
    parse_tool_choice, tool_choice_name,
};
pub use tool::{FunctionTool, ToolContext, ToolFuture, ToolOutcome, ToolRegistry};
pub use tools::{ToolSpec, check_strict_compliance, strict_schema};
