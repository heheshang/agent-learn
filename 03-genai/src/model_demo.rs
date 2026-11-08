//! 模型/消息层 demo —— 直接用 `genai::Client` 发请求。
//!
//! 这一层**没有** agent 循环：每一次 `exec_chat` 都是一次独立的
//! 请求-响应。你需要自己把历史消息带过去，工具调用也要自己执行、自己
//! 回填。理解这一层是理解 agent 层的前提。

use agent_logging::{debug, info, warn};
use anyhow::Result;
use genai::Client;
use genai::chat::printer::print_chat_stream;
use genai::chat::{ChatMessage, ChatRequest, ChatResponseFormat, Tool, ToolResponse};
use serde_json::json;

use crate::tools::{ToolError as CalcError, calculate};

/// 使用的模型。`genai` 按模型名前缀自动选适配器：`deepseek-*` → DeepSeek
/// （OpenAI 兼容协议），API key 从 `DEEPSEEK_API_KEY` 读。
pub const MODEL: &str = "deepseek-v4-flash";

/// 单轮对话：一次请求-响应。
///
/// `ChatRequest` = system 提示 + 消息列表。响应 `ChatResponse` 里可以拿到
/// 文本、usage、stop_reason。
pub async fn run_single_turn() -> Result<()> {
    println!("\n=== 1. 单轮对话 ===");

    let client = Client::builder().build();

    let chat_req = ChatRequest::new(vec![
        ChatMessage::system("你是一个简洁的助手，回答不超过一句话。"),
        ChatMessage::user("用一句话解释什么是 RAII。"),
    ]);

    let chat_res = client.exec_chat(MODEL, chat_req, None).await?;

    let usage = &chat_res.usage;
    info!(
        model = %chat_res.model_iden,
        prompt_tokens = usage.prompt_tokens.unwrap_or(0),
        completion_tokens = usage.completion_tokens.unwrap_or(0),
        total_tokens = usage.total_tokens.unwrap_or(0),
        "单轮对话完成"
    );

    println!("回答: {}", chat_res.first_text().unwrap_or("(空)"));
    println!("模型: {}", chat_res.model_iden);
    println!(
        "用量: prompt={} completion={} total={}",
        usage.prompt_tokens.unwrap_or(0),
        usage.completion_tokens.unwrap_or(0),
        usage.total_tokens.unwrap_or(0)
    );
    if let Some(stop) = chat_res.stop_reason.as_ref() {
        debug!(stop_reason = ?stop, "模型停止原因");
        println!("停止原因: {stop:?}");
    }

    Ok(())
}

/// 多轮对话：把历史消息一直带着发。
///
/// 这是 agent 的「状态」在模型层的原形：状态 = 一串消息。`genai` 本身
/// 不替你保管历史（`previous_response_id` 只对 OpenAI Responses API 有效）。
pub async fn run_multi_turn() -> Result<()> {
    println!("\n=== 2. 多轮对话（手工维护历史） ===");

    let client = Client::builder().build();

    let mut chat_req = ChatRequest::from_system("你是一个耐心的老师，回答尽量简短。");

    let questions = ["什么是所有权？", "那借用检查器又在解决什么问题？"];

    for q in questions {
        chat_req = chat_req.append_message(ChatMessage::user(q));

        info!(question = %q, message_count = chat_req.messages.len(), "发起多轮对话请求");
        let chat_res = client.exec_chat(MODEL, chat_req.clone(), None).await?;
        let answer = chat_res.first_text().unwrap_or("(空)").to_string();

        println!("问: {q}");
        println!("答: {answer}\n");

        // 关键：把这一轮 assistant 的回答也塞回历史，下一轮模型才看得见。
        chat_req = chat_req.append_message(ChatMessage::assistant(answer));
    }

    info!(
        message_count = chat_req.messages.len(),
        "多轮对话结束，历史已累积"
    );
    println!("历史消息条数: {}", chat_req.messages.len());

    Ok(())
}

/// 流式输出：逐块打印。
///
/// `exec_chat_stream` 返回一个 `ChatStream`，实现 `Stream<Item = Result<ChatStreamEvent>>`。
/// `printer::print_chat_stream` 是个便利封装，返回拼好的完整文本。
pub async fn run_streaming() -> Result<()> {
    println!("\n=== 3. 流式输出 ===");

    let client = Client::builder().build();

    let chat_req = ChatRequest::new(vec![ChatMessage::user("用三句话讲讲 Rust 的所有权模型。")]);

    let chat_res = client.exec_chat_stream(MODEL, chat_req, None).await?;

    print!("流式回答: ");
    let full = print_chat_stream(chat_res, None).await?;
    println!("\n(完整长度: {} 字符)", full.len());
    info!(chars = full.len(), "流式输出结束");

    Ok(())
}

/// 结构化输出：让模型按 JSON schema 返回。
pub async fn run_json_mode() -> Result<()> {
    println!("\n=== 4. JSON 模式 ===");

    let client = Client::builder().build();

    let chat_req = ChatRequest::new(vec![ChatMessage::user(
        "把下面这段话拆成要点列表，并以 json 格式输出：Rust 的所有权保证没有数据竞争。",
    )]);

    let options =
        genai::chat::ChatOptions::default().with_response_format(ChatResponseFormat::JsonMode);

    let chat_res = client.exec_chat(MODEL, chat_req, Some(&options)).await?;

    let text = chat_res.first_text().unwrap_or("(空)");
    println!("原始返回: {text}");
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(v) => {
            info!("JSON 模式解析成功");
            println!("解析成功: {v}")
        }
        Err(e) => {
            warn!(error = %e, "JSON 模式解析失败（模型没听话）");
            println!("解析失败(模型没听话): {e}")
        }
    }

    Ok(())
}

/// 原始工具调用：模型返回「我要调工具」，但**执行和回填都得自己写**。
///
/// 这正是 agent 层要自动化掉的那部分：
///
/// ```text
/// 发请求 → 模型返回 tool_calls → 你执行工具 → 结果作为 tool 消息回填
///        → 再发请求 → 模型给出最终答案
/// ```
pub async fn run_raw_tool_call() -> Result<()> {
    println!("\n=== 5. 原始工具调用（手工执行 + 回填） ===");

    let client = Client::builder().build();

    // 5.1 声明工具。schema 是标准 JSON Schema，模型靠它决定怎么调。
    let tool = Tool::new("calculate")
        .with_description("计算一个算术表达式，返回结果")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "算术表达式，例如 (1 + 2) * 3"
                }
            },
            "required": ["expression"]
        }));

    let mut chat_req = ChatRequest::new(vec![ChatMessage::user(
        "帮我算一下 (12 + 34) * 2，然后用一句话告诉我结果。",
    )])
    .with_tools(vec![tool]);

    // 5.2 第一轮：模型应该返回 tool_calls 而不是最终答案。
    let chat_res = client.exec_chat(MODEL, chat_req.clone(), None).await?;
    let tool_calls = chat_res
        .tool_calls()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();

    if tool_calls.is_empty() {
        warn!("模型没有调用工具，直接给出了答案");
        println!(
            "模型没调工具，直接回答: {}",
            chat_res.first_text().unwrap_or("(空)")
        );
        return Ok(());
    }

    // 5.3 执行每个工具调用。
    let mut responses = Vec::new();
    for call in &tool_calls {
        info!(tool = %call.fn_name, args = %call.fn_arguments, "模型要求调用工具");
        println!("模型要求调用: {} args={}", call.fn_name, call.fn_arguments);

        let result = match call.fn_name.as_str() {
            "calculate" => {
                let expr = call.fn_arguments.get("expression").and_then(|v| v.as_str());
                match expr {
                    Some(expr) => match calculate(expr) {
                        Ok(value) => {
                            debug!(expression = %expr, value, "工具 calculate 执行成功");
                            ToolResponse::new(
                                call.call_id.clone(),
                                json!({ "result": value }).to_string(),
                            )
                        }
                        Err(CalcError::InvalidArguments(msg)) => {
                            warn!(expression = %expr, error = %msg, "工具 calculate 参数不合法");
                            ToolResponse::new(
                                call.call_id.clone(),
                                json!({ "error": msg }).to_string(),
                            )
                        }
                    },
                    None => {
                        warn!("工具 calculate 缺少 expression 参数");
                        ToolResponse::new(
                            call.call_id.clone(),
                            json!({ "error": "缺少 expression 参数" }).to_string(),
                        )
                    }
                }
            }
            other => {
                warn!(tool = %other, "收到未知工具的调用请求");
                ToolResponse::new(
                    call.call_id.clone(),
                    json!({ "error": format!("未知工具: {other}") }).to_string(),
                )
            }
        };
        responses.push(result);
    }

    // 5.4 回填：assistant 的 tool_calls + tool 结果，一起塞回历史。
    chat_req = chat_req
        .append_message(tool_calls)
        .append_message(responses);

    // 5.5 第二轮：模型看到工具结果，给出最终答案。
    let chat_res = client.exec_chat(MODEL, chat_req, None).await?;
    info!("工具结果已回填，发起第二轮请求");
    println!("最终回答: {}", chat_res.first_text().unwrap_or("(空)"));

    Ok(())
}

/// 跑完模型层所有 demo。
pub async fn run_chat_demos() -> Result<()> {
    run_single_turn().await?;
    run_multi_turn().await?;
    run_streaming().await?;
    run_json_mode().await?;
    Ok(())
}

/// 跑完工具层 demo。
pub async fn run_tool_demos() -> Result<()> {
    run_raw_tool_call().await?;
    Ok(())
}
