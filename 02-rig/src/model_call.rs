//! 模型与消息调用 —— Rig 的最底层用法。
//!
//! 这一层不涉及 Agent，只是「构造请求 → 发给模型 → 读回复」。
//! 三个核心概念：
//!
//! - [`Message`]：一条对话消息（system / user / assistant / tool_result）。
//! - [`CompletionRequest`]：一次请求 = 一串消息 + 采样参数 + 工具声明。
//! - [`Model`]：一个模型端点（wire + transport），`call` 发一次，`stream` 流式读。
//!
//! 与 Agent 的区别：这里没有循环。模型让你调工具，你得自己执行、再把结果
//! 作为 `Message::tool_result` 追加回历史，然后自己再发一次请求。

use agent_logging::{debug, info};
use rig::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use rig::prelude::Message;
use rig::providers::deepseek;
use rig::providers::openai::{OpenAI, wire::OpenAiWire};

/// 构造一个 DeepSeek 模型。
///
/// 凭据来自环境变量 `DEEPSEEK_API_KEY`（由 `deepseek::from_env` 读取），
/// 模型的 base url 由 provider 内置。
pub fn model() -> anyhow::Result<rig::driver::Model<OpenAiWire>> {
    let client: OpenAI = deepseek::from_env()?;
    debug!("DeepSeek 客户端已构造");
    Ok(client.completion(deepseek::DEEPSEEK_V4_FLASH))
}

/// 演示 1：单轮调用 —— 一条 user 消息进去，一段文本出来。
pub async fn single_turn() -> anyhow::Result<()> {
    let model = model()?;

    let request = CompletionRequest::new("用一句话解释 Rust 的所有权机制。")
        .preamble("你是一个简洁的 Rust 导师，回答不超过两句话。")
        .temperature(0.3)
        .max_tokens(512);

    info!("发起单轮模型调用（model=deepseek-v4-flash）");
    let response: CompletionResponse = model.call(request).await?;
    info!(usage = ?response.usage, "单轮调用完成");

    println!("=== 1. 单轮调用 ===");
    println!("回复: {}", response.text());
    println!("用量: {:?}", response.usage);
    Ok(())
}

/// 演示 2：多轮消息 —— 手工维护 chat_history。
///
/// `CompletionRequest::from(Vec<Message>)` 把一串消息作为完整历史发送；
/// `Message::user` / `Message::assistant` / `Message::system` 构造各角色。
pub async fn multi_turn() -> anyhow::Result<()> {
    let model = model()?;

    let history = vec![
        Message::system("你是数学助手，回答尽量简洁。"),
        Message::user("什么是素数？"),
        Message::assistant("素数是大于 1 且只有 1 和自身两个正因数的自然数。"),
        Message::user("那 97 呢？"),
    ];

    debug!(message_count = history.len(), "发起多轮模型调用");

    let request = CompletionRequest::from(history);
    let response = model.call(request).await?;

    println!("\n=== 2. 多轮消息 ===");
    println!("回复: {}", response.text());
    Ok(())
}

/// 演示 3：声明工具但不由 Rig 执行 —— 观察模型如何请求调用工具。
///
/// `CompletionRequest::tool` 把工具的「名字 / 描述 / 参数 JSON Schema」发给模型。
/// 模型返回 `AssistantContent::ToolCall`，但**没有人去执行它** —— 这是
/// `AgentBuilder` 存在的原因。
pub async fn raw_tool_call() -> anyhow::Result<()> {
    let model = model()?;

    let tool = ToolDefinition::new(
        rig::message::ToolName::new("get_weather")?,
        "查询某个城市的天气",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "城市名" }
            },
            "required": ["city"]
        }),
    );

    let request = CompletionRequest::new("北京今天天气怎么样？").tool(tool);
    info!("发起带工具声明的模型调用（工具不会被执行，只观察模型的请求）");
    let response = model.call(request).await?;

    println!("\n=== 3. 原始工具调用 ===");
    let calls: Vec<_> = response.tool_calls().collect();
    if calls.is_empty() {
        debug!("模型没有请求调用工具，直接给出了答案");
        println!("模型没有请求调用工具，直接回答: {}", response.text());
    } else {
        for call in calls {
            info!(
                tool = %call.function.name,
                args = ?call.function.arguments,
                "模型请求调用工具"
            );
            println!(
                "模型请求调用工具: {} 参数 {:?}",
                call.function.name, call.function.arguments
            );
        }
    }
    Ok(())
}
