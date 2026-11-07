//! Agent 层 demo —— 用 `rust-genai-agent` 的 `Agent` 跑自动工具循环。
//!
//! 对比 `model_demo` 里的手工循环，agent 层做的是同一件事，但：
//!
//! - **状态**：`AgentState` 持有 system prompt、模型、工具、消息历史。
//!   `Agent` 是有状态的 facade，历史会累积在 `state().messages` 里。
//! - **工具**：`AgentTool` 带 JSON Schema，agent 循环负责**校验参数**、
//!   **执行**、**回填结果**、**再问模型**，直到模型给出纯文本。
//! - **middleware（钩子）**：`before_tool_call` / `after_tool_call` /
//!   `transform_context` 等扩展点，在工具调用前后拦截、改写、审计。
//!
//! 事件系统：`AgentEvent` 是循环的生命周期流，`subscribe_fn` 注册监听器，
//! 返回的 `Subscription` 必须持有到运行结束（RAII 退订）。

use std::sync::Arc;

use anyhow::Result;
use rust_genai_agent::{
    AfterToolCallContext, AfterToolCallResult, Agent, AgentConfig, AgentEvent, AgentMessage,
    AgentState, AgentTool, AgentToolResult, AssistantMessageEvent, BeforeToolCallContext,
    BeforeToolCallResult, CancellationToken, FnTool, GenaiStreamFn, ToolError, ToolSpec,
};
use serde_json::{Value, json};

use crate::model_demo::MODEL;
use crate::tools::{ToolError as CalcError, calculate, calculate_schema};

/// 把算术工具包成 agent 工具。
///
/// `FnTool::from_value_fn` 是最省事的写法：只关心 `args`（已校验、
/// 已按 schema 强制类型后的 JSON），返回一个 `AgentToolResult`。
///
/// 更复杂的场景用 `FnTool::new`，能拿到 `ToolCallContext`、`CancellationToken`
/// 和 `UpdateSink`（可以流式上报中间结果）。
fn calculate_tool() -> Arc<dyn AgentTool> {
    let tool = FnTool::from_value_fn(
        ToolSpec::new(
            "calculate",
            "计算一个算术表达式并返回结果。支持 + - * / 和括号。",
            calculate_schema(),
        ),
        |args: Value| async move {
            let expr = args
                .get("expression")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            match calculate(&expr) {
                Ok(value) => Ok(AgentToolResult::json(json!({
                    "expression": expr,
                    "result": value,
                }))),
                Err(CalcError::InvalidArguments(msg)) => {
                    // 参数问题：报成工具错误，模型能读到并自我纠正。
                    Err(ToolError::InvalidArguments(msg))
                }
            }
        },
    );

    Arc::new(tool)
}

/// 一个「带状态的工具」示例：统计被调用了几次。
///
/// 工具对象本身持有 `Arc<AtomicUsize>`，状态跨调用共享。
fn counting_tool() -> Arc<dyn AgentTool> {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let tool = FnTool::from_value_fn(
        ToolSpec::new(
            "count_calls",
            "返回这个工具到目前为止被调用的次数",
            json!({ "type": "object", "properties": {} }),
        ),
        move |_args: Value| {
            let counter = counter.clone();
            async move {
                let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                Ok(AgentToolResult::json(json!({ "calls": n })))
            }
        },
    );

    Arc::new(tool)
}

/// `create_agent` —— 构造一个挂了工具的 agent。
///
/// 对应 Python LangChain 的 `create_agent` / OpenAI Agents SDK 的 `Agent`：
/// 给它一个模型 + 一组工具 + 一个系统提示，它就能自己跑工具循环。
///
/// 三种构造方式：
/// 1. `Agent::new(AgentConfig)` —— 闭包风格，最灵活
/// 2. `AgentBuilder::new(AgentSetup)` —— 数据 + 行为分离（适合非 Rust 宿主）
/// 3. `AgentState` + `AgentConfig` —— 本函数用的
fn create_agent(with_hooks: bool) -> Result<Agent> {
    let client = genai::Client::builder().build();

    let state = AgentState {
        model: MODEL.into(),
        system_prompt:
            "你是一个会使用工具的助手。需要算数时必须调用 calculate 工具，不要自己心算。".into(),
        tools: vec![calculate_tool(), counting_tool()],
        ..AgentState::default()
    };

    let mut config = AgentConfig::default()
        .with_initial_state(state)
        .with_stream_fn(Arc::new(GenaiStreamFn::new(client)));

    if with_hooks {
        config = config
            // ── middleware 1：执行前审计 + 改写参数 ──────────────────
            //
            // `BeforeToolCallHook` 拿到的是「已按 schema 校验并强制类型」的
            // 参数。改写 `ctx.args` 就会直接传给工具（不会再校验一次）。
            .with_before_tool_call(Arc::new(
                |ctx: &mut BeforeToolCallContext, _cancel: CancellationToken| {
                    Box::pin(async move {
                        println!(
                            "  [middleware] before: {} args={}",
                            ctx.tool_call.name, ctx.args
                        );

                        // 例：把表达式里的空格去掉，减少解析失败
                        if ctx.tool_call.name == "calculate"
                            && let Some(expr) = ctx.args.get("expression").and_then(|v| v.as_str())
                        {
                            let cleaned = expr.replace(' ', "");
                            if let Some(obj) = ctx.args.as_object_mut() {
                                obj.insert("expression".into(), Value::String(cleaned));
                            }
                        }

                        // 返回 `None`（或 `block: false`）表示放行。
                        None::<BeforeToolCallResult>
                    })
                },
            ))
            // ── middleware 2：执行后审计 ──────────────────────────
            //
            // `AfterToolCallHook` 拿到的是工具执行完的结果。返回
            // `Some(AfterToolCallResult)` 可以覆盖结果的 content / is_error。
            .with_after_tool_call(Arc::new(
                |ctx: AfterToolCallContext, _cancel: CancellationToken| {
                    Box::pin(async move {
                        println!(
                            "  [middleware] after: {} is_error={}",
                            ctx.tool_call.name, ctx.is_error
                        );

                        // 例：把成功结果统一加一个字段
                        if !ctx.is_error {
                            let mut details = ctx.result.details.clone();
                            if details.is_object() {
                                details["_audited"] = json!(true);
                            }
                            return Some(AfterToolCallResult {
                                details: Some(details),
                                ..Default::default()
                            });
                        }
                        None
                    })
                },
            ));
    }

    Ok(Agent::new(config))
}

/// Agent demo：自动工具循环 + 事件流。
pub async fn run_agent_demo() -> Result<()> {
    println!("\n=== 6. Agent 自动工具循环 ===");

    let agent = create_agent(false)?;

    // 事件监听器：订阅后必须持有 `Subscription`，drop 即退订。
    let subscription = agent.subscribe_fn(|event, _cancel| async move {
        match event {
            AgentEvent::MessageUpdate {
                assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
                ..
            } => {
                print!("{delta}");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } => {
                println!("\n  [事件] 开始调用工具 {tool_name} args={args}");
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                result,
                is_error,
                ..
            } => {
                println!(
                    "  [事件] 工具 {tool_name} 结束 error={is_error} result={}",
                    result.details
                );
            }
            AgentEvent::TurnEnd { tool_results, .. } => {
                println!("  [事件] 本轮结束，工具结果 {} 条", tool_results.len());
            }
            _ => {}
        }
    });

    agent
        .prompt("帮我算一下 (12 + 34) * 2，然后再算一次 100 / 4。")
        .await?;
    println!();

    // 打印 agent 状态：历史消息累积在这里。
    let state = agent.state();
    println!("\n--- Agent 状态 ---");
    println!("消息条数: {}", state.messages.len());
    println!("工具数: {}", state.tools.len());
    if let Some(err) = state.error_message.as_ref() {
        println!("错误: {err}");
    }
    for (i, msg) in state.messages.iter().enumerate() {
        let preview = match msg {
            AgentMessage::User(m) => format!("{:?}", m.content),
            AgentMessage::Assistant(m) => {
                let text = m.text();
                let tools: Vec<_> = m.tool_calls().map(|c| c.name.clone()).collect();
                if tools.is_empty() {
                    text.to_string()
                } else {
                    format!("{text} [调用工具: {tools:?}]")
                }
            }
            AgentMessage::ToolResult(m) => format!("{} -> {:?}", m.tool_name, m.content),
            AgentMessage::Custom(m) => format!("{:?}", m.role),
        };
        let preview = preview.chars().take(120).collect::<String>();
        println!("  [{i}] {}: {preview}", msg.role());
    }

    drop(subscription);
    Ok(())
}

/// middleware demo：装上钩子后再跑一遍，看钩子生效。
pub async fn run_hooks_demo() -> Result<()> {
    println!("\n=== 7. Agent middleware（钩子） ===");

    let agent = create_agent(true)?;

    let subscription = agent.subscribe_fn(|event, _cancel| async move {
        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = event
        {
            print!("{delta}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    });

    // 故意带空格，验证 before 钩子的参数改写生效。
    agent.prompt("计算 ( 12 + 34 ) * 2").await?;
    println!();

    // 检查 after 钩子加上的 `_audited` 字段有没有落进 tool result。
    let state = agent.state();
    for msg in &state.messages {
        if let AgentMessage::ToolResult(m) = msg {
            println!("\n工具结果详情: {}", m.details);
        }
    }

    drop(subscription);
    Ok(())
}

/// HTTP 边界的 middleware：`on_payload` / `on_response`。
///
/// 前面两个钩子挂在**工具**上。这一对挂在**HTTP 请求/响应**上，是 agent
/// 与 provider 之间唯一的一道关口：
///
/// - `on_payload(payload, model)` —— 请求体序列化之后、真正发出去之前。
///   返回 `Some(v)` 替换请求体，返回 `None` 保持原样。
/// - `on_response(info, model)` —— 收到响应头时（**body 还没读**）触发，
///   4xx/5xx 也会触发。纯观测。
///
/// 典型用途：加 trace header、脱敏、审计、计费打点。
///
/// ⚠️ 语义要点：per-request 的钩子会**替换**构造期的钩子，两者不会叠加
/// —— 每次物理请求（包括重试）每个通道只触发一次。
pub async fn run_http_middleware_demo() -> Result<()> {
    println!("\n=== 8. HTTP 边界 middleware（on_payload / on_response） ===");

    let on_payload: rust_genai_agent::OnPayloadHook = Arc::new(|payload, model| {
        Box::pin(async move {
            // 观测：打印将要发出的请求体摘要。
            let msg_count = payload
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            println!("  [on_payload] model={model} messages={msg_count}");

            // 返回 Some 替换请求体：这里注入一个自定义字段做演示。
            let mut payload = payload;
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("_injected_by_middleware".into(), Value::Bool(true));
            }
            Some(payload)
        })
    });

    let on_response: rust_genai_agent::OnResponseHook = Arc::new(|info, model| {
        Box::pin(async move {
            let req_id = info.header("x-request-id").unwrap_or("-");
            println!(
                "  [on_response] model={model} status={} x-request-id={req_id}",
                info.status
            );
        })
    });

    // 用 `GenaiStreamFn::with_exec_hooks` 把钩子装到 client 上。
    let builder = genai::Client::builder();
    let stream_fn = GenaiStreamFn::with_exec_hooks(builder, Some(on_payload), Some(on_response));

    let state = AgentState {
        model: MODEL.into(),
        system_prompt: "你是一个简洁的助手。".into(),
        ..AgentState::default()
    };

    let agent = Agent::new(
        AgentConfig::default()
            .with_initial_state(state)
            .with_stream_fn(Arc::new(stream_fn)),
    );

    let subscription = agent.subscribe_fn(|event, _cancel| async move {
        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = event
        {
            print!("{delta}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    });

    agent.prompt("用一句话说明什么是幂等。").await?;
    println!();

    drop(subscription);
    Ok(())
}

/// 状态 demo：同一个 agent 连续多轮对话，状态是累积的。
///
/// `Agent` 是**有状态 facade**：
///
/// - `state().messages` —— 已提交的历史，跨 `prompt()` 调用累积
/// - `state().tools` —— 工具集
/// - `state().system_prompt` / `state().model` —— 可用 setter 改
/// - `agent.reset()` —— 清空历史回到初始状态
///
/// 对比：模型层的 `genai::Client` 是**无状态**的，历史得自己带。
pub async fn run_state_demo() -> Result<()> {
    println!("\n=== 9. Agent 状态 ===");

    let agent = create_agent(false)?;

    let subscription = agent.subscribe_fn(|event, _cancel| async move {
        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = event
        {
            print!("{delta}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    });

    // 第一轮
    agent.prompt("我叫 Rodriguez。").await?;
    println!();
    println!("  [状态] 消息数 = {}", agent.state().messages.len());

    // 第二轮：agent 记得上一轮说了什么
    agent.prompt("我叫什么名字？").await?;
    println!();
    println!("  [状态] 消息数 = {}", agent.state().messages.len());

    // 运行时改系统提示
    agent.set_system_prompt("你是一个只用一句话回答的助手。");
    println!("  [状态] 新系统提示 = {}", agent.state().system_prompt);

    // 重置：清空历史
    agent.reset()?;
    println!("  [状态] reset 后消息数 = {}", agent.state().messages.len());

    drop(subscription);
    Ok(())
}
