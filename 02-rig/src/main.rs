//! 02-rig —— 基于 Rig 0.46 的四个核心概念示例。
//!
//! 1. [`model_call`]：模型与消息调用（最底层，没有 Agent）。
//! 2. [`agent`]：创建 Agent，让模型能自己调用工具。
//! 3. [`tools`]：工具的定义与执行。
//! 4. [`middleware`]：传输层的中间件。
//!
//! 运行：`cargo run`

mod agent;
mod middleware;
mod model_call;
mod tools;

use agent_logging::{info, init};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    init("rig-02");

    // 需要环境变量 DEEPSEEK_API_KEY
    if std::env::var("DEEPSEEK_API_KEY").is_err() {
        agent_logging::error!("未设置环境变量 DEEPSEEK_API_KEY");
        std::process::exit(1);
    }

    info!("开始运行 02-rig 的四个核心概念 demo");

    println!("############ 1. 模型与消息调用 ############");
    model_call::single_turn().await?;
    model_call::multi_turn().await?;
    model_call::raw_tool_call().await?;

    println!("\n############ 2. Agent 与工具 ############");
    agent::agent_with_tools().await?;
    agent::agent_state().await?;
    agent::agent_with_history().await?;

    println!("\n############ 3. 工具上下文 ############");
    tools::context_demo().await?;

    println!("\n############ 4. Middleware ############");
    middleware::middleware_demo().await?;

    info!("02-rig 的全部 demo 执行完毕");

    Ok(())
}
