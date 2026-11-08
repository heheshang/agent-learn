//! Agent —— 把「模型调用」变成一个会自动循环的对话代理。
//!
//! `AgentBuilder` 是构造入口：
//!
//! ```ignore
//! AgentBuilder::new(model)      // 注册模型为 "default"
//!     .preamble("你是……")        // 系统提示词
//!     .tool(SomeTool)            // 挂上工具
//!     .build()                   // 得到 Agent
//! ```
//!
//! 然后 `agent.prompt("……").await?` 触发一轮运行。Rig 在内部做的事，
//! 是把 `model_call.rs` 里手工写的循环自动化：
//!
//! ```text
//! 发请求 → 模型要调工具？→ 执行工具 → 结果作为 tool_result 追加进历史
//!        → 再发请求 …… → 模型给出纯文本回答 → 结束
//! ```
//!
//! 每轮的工具调用、结果回填、重试，都是 Agent 负责的。

use agent_logging::{debug, info};
use rig::prelude::{Agent, AgentBuilder};
use rig::providers::deepseek;
use rig::providers::openai::OpenAI;

use crate::tools::{AddTwo, CountCalls, GetWeather, Whoami};

/// 构造一个挂了工具的 Agent。
///
/// `AgentBuilder::new(model)` 把模型注册成 `"default"`，之后
/// `agent.prompt(...)` 就是用它。
pub fn build_agent() -> anyhow::Result<Agent> {
    build_agent_with(false)
}

/// 构造一个 Agent，`with_whoami` 决定是否挂上读取 `ToolContext` 的工具。
///
/// `Tool` 要求 `Sized`，不是 trait object，所以工具是**逐个**注册进
/// builder 的（`AgentBuilder::tool` 是泛型方法）。
pub fn build_agent_with(with_whoami: bool) -> anyhow::Result<Agent> {
    let client: OpenAI = deepseek::from_env()?;
    let model = client.completion(deepseek::DEEPSEEK_V4_FLASH);

    let mut builder = AgentBuilder::new(model)
        .name("assistant")
        .description("一个会使用工具的通用助手")
        .preamble("你是一个乐于助人的助手。需要外部信息时，主动调用工具，不要凭空猜测。")
        // 工具调用会占掉一轮：调工具 + 读结果 + 作答，至少要 2~3 轮。
        .default_max_turns(8)
        .tool(AddTwo)
        .tool(GetWeather)
        .tool(CountCalls::default());

    if with_whoami {
        builder = builder.tool(Whoami);
    }

    info!(
        with_whoami,
        tools = if with_whoami { 4 } else { 3 },
        "Agent 已构造"
    );
    Ok(builder.build())
}

/// 演示 1：Agent 自主调用工具。
///
/// 提示语刻意留了余地（「先查天气再回答」），模型会先调 `get_weather`，
/// Rig 执行它并把结果回填，然后模型据此作答。整条循环都在 `prompt` 内部。
pub async fn agent_with_tools() -> anyhow::Result<()> {
    let agent = build_agent()?;

    info!("发起 Agent 提示（max_turns=5）");
    let response = agent
        .prompt("北京今天天气怎么样？直接告诉我结论就行。")
        .max_turns(5)
        .await?;
    info!(message_count = response.messages().len(), "Agent 运行结束");

    println!("=== Agent 自主调用工具 ===");
    println!("最终回答: {}", response.output());
    println!("本次运行共 {} 条消息:", response.messages().len());
    for message in response.messages() {
        debug!(message = ?message, "Agent 消息");
        println!("  - {message:?}");
    }
    Ok(())
}

/// 演示 2：Agent 的状态 —— 谁在共享状态。
///
/// Rig 里「状态」分三层，职责完全不同：
///
/// | 层 | 载体 | 生命周期 | 用途 |
/// |----|------|----------|------|
/// | 工具实例状态 | 工具结构体自身的字段 | 跟随 Agent | 计数器、连接池、缓存 |
/// | 运行内状态 | `ToolContext` | 一次 `prompt` | 把宿主数据喂给工具 |
/// | 跨运行状态 | `AgentBuilder::memory` | 跨多次 `prompt` | 会话记忆、持久化 |
/// | Hook 状态 | `Scratchpad` | 一次 run 的 hook 之间 | 中间计数、路由决策 |
///
/// 下面演示工具实例状态与 `ToolContext` 状态。
pub async fn agent_state() -> anyhow::Result<()> {
    let agent = build_agent()?;

    // 工具实例状态：CountCalls 内部用 Arc<AtomicUsize> 计数，
    // 同一次运行里多次调用会累加。
    info!("发起 Agent 提示：连续调用 count_calls 三次");
    let response = agent
        .prompt("请连续调用 count_calls 工具三次，然后告诉我调用了几次。")
        .max_turns(8)
        .await?;
    info!("工具实例状态 demo 完成");
    println!("=== 工具实例状态 ===");
    println!("{}", response.output());

    // 运行内状态：把宿主数据放进 ToolContext，工具用 require 读。
    // 这里单独建一个挂了 whoami 工具的 Agent —— 只有它能读 ToolContext。
    let agent = build_agent_with(true)?;

    let mut context = rig::prelude::ToolContext::new();
    context.insert(crate::tools::SessionUser {
        name: "Rodriguez".to_string(),
        tier: "pro".to_string(),
    })?;

    // 模型看不到 ToolContext 里的值，必须通过调用 `whoami` 工具才能知道。
    info!("发起 Agent 提示：通过 whoami 工具读取 ToolContext");
    let response = agent
        .prompt("调用 whoami 工具，告诉我我是谁。")
        .tool_context(context)
        .await?;
    info!("ToolContext demo 完成");
    println!("\n=== 运行内状态 (ToolContext) ===");
    println!("{}", response.output());

    Ok(())
}

/// 演示 3：Agent 的对话历史由调用方持有。
///
/// `agent.chat(prompt, &mut history)` 把历史作为输入，跑完把新消息写回
/// 同一个 `Vec`。这是「跨运行状态」最轻量的做法：状态在你自己手里，
/// 想存哪就存哪。
pub async fn agent_with_history() -> anyhow::Result<()> {
    let agent = build_agent()?;
    let mut history: Vec<rig::prelude::Message> = Vec::new();

    info!("发起第 1 轮对话");
    let first = agent.chat("我叫 Rodriguez，记住我。", &mut history).await?;
    println!("=== 对话历史 ===");
    println!("第 1 轮: {}", first.output());

    info!(history_len = history.len(), "发起第 2 轮对话");
    let second = agent.chat("我刚才说我叫什么？", &mut history).await?;
    println!("第 2 轮: {}", second.output());
    println!("历史消息数: {}", history.len());
    info!(history_len = history.len(), "对话历史 demo 完成");

    Ok(())
}
