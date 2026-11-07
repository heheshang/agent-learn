//! 03-genAI —— 用 `genai` crate 跑通「模型调用 → 消息 → 工具 → Agent」这条主线。
//!
//! 这个项目分成两层，刻意分开：
//!
//! | 层 | crate | 负责什么 |
//! |---|---|---|
//! | 模型/消息层 | `genai` | 单次请求-响应、消息历史、流式输出、原始工具调用 |
//! | Agent 层 | `rust-genai-agent` | 状态机化的对话循环：自动调工具、回填结果、多轮收敛 |
//!
//! `genai` 本身**不含** agent 循环。`rust-genai-agent` 是独立 crate，它把
//! `genai` 的流式接口包装成一个 `StreamFn`，然后在上面跑一个带状态的
//! `Agent` 循环。
//!
//! ⚠️ 依赖陷阱：`rust-genai-agent` 依赖的是 **AgentPrism fork** 的 `genai`
//! （crates.io 上的 `genai-agentprism`，lib 名仍叫 `genai`），不是上游
//! `genai 0.7`。两者类型**不互通**，混用会编译失败。所以本项目的
//! `Cargo.toml` 里只有一个 `genai` 依赖，指向 fork。

mod agent_demo;
mod model_demo;
mod tools;

use anyhow::Result;

/// 顶层入口：按名字跑各个 demo。
///
/// 用法：
/// ```text
/// cargo run -- chat      # 模型/消息层：单轮、多轮、流式、JSON
/// cargo run -- tools     # 模型层：原始工具调用（手工执行 + 回填）
/// cargo run -- agent     # Agent 层：create_agent + 自动工具循环
/// cargo run -- hooks     # Agent 层：工具钩子（middleware）
/// cargo run -- http      # Agent 层：HTTP 边界 middleware
/// cargo run -- state     # Agent 层：状态累积与重置
/// cargo run -- all       # 全部
/// ```
#[tokio::main]
async fn main() -> Result<()> {
    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "chat" => model_demo::run_chat_demos().await?,
        "tools" => model_demo::run_tool_demos().await?,
        "agent" => agent_demo::run_agent_demo().await?,
        "hooks" => agent_demo::run_hooks_demo().await?,
        "http" => agent_demo::run_http_middleware_demo().await?,
        "state" => agent_demo::run_state_demo().await?,
        "all" => {
            model_demo::run_chat_demos().await?;
            model_demo::run_tool_demos().await?;
            agent_demo::run_agent_demo().await?;
            agent_demo::run_hooks_demo().await?;
            agent_demo::run_http_middleware_demo().await?;
            agent_demo::run_state_demo().await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: chat / tools / agent / hooks / http / state / all）"
            );
        }
    }

    Ok(())
}
