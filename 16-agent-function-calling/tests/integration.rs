//! Function Calling 循环的集成测试 —— 零网络、零延迟、完全确定。
//!
//! ## 为什么要用 `ScriptedFunctionLlm`
//!
//! 本项目的循环有**四个**必须锁住的行为（都是实测踩出来的坑），
//! 它们都需要**控制模型返回什么**才能测试：
//!
//! 1. 并行调用（一次多个 `tool_calls`）
//! 2. 工具失败后模型自我纠正
//! 3. 每个 `call_id` 都必须有回执
//! 4. 推理模型的 `reasoning_content` 必须回传
//!
//! 这些行为用真实模型测是**不可靠**的 ——
//! 模型今天返回 2 个调用，明天返回 1 个，测试就会随机失败。
//!
//! 所以用 [`ScriptedFunctionLlm`] 把模型输出**钉死**。
//!
//! ## 测试分层
//!
//! | 层 | 测什么 | 需要模型？ |
//! |---|---|---|
//! | 本文件 | 循环的**控制流**（并行、回执、自我纠正、降级） | ❌ |
//! | `api` 测试 | HTTP 契约（状态码、响应形状） | ❌ |
//! | 真实模型 | 端到端是否真的能跑通 | ✅（手动） |

use agent_16::agent::Agent;
use agent_16::api::AppStateInner;
use agent_16::api::routes::router;
use agent_16::config::AppConfig;
use agent_16::llm::{ScriptedFunctionLlm, ScriptedStep, ScriptedToolCall, ToolChoice};
use agent_16::tool::{FunctionTool, ToolContext, ToolFuture, ToolOutcome, ToolRegistry};
use agent_16::tools::ToolSpec;

use serde_json::{Value, json};
use std::sync::Arc;

/// 一个确定性的假工具：把 `value` 原样返回。
struct EchoTool {
    name: &'static str,
}

impl FunctionTool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            self.name,
            "回显传入的值",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string", "description": "要回显的值" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let value = args
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(ToolOutcome::ok(self.name, value))
        })
    }
}

/// 一个总是失败的工具。
struct FailingTool {
    name: &'static str,
}

impl FunctionTool for FailingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            self.name,
            "总是失败",
            json!({ "type": "object", "properties": {} }),
        )
    }

    fn call<'a>(&'a self, _args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move { Err(agent_16::error::ToolError::execution(self.name, "工具炸了")) })
    }
}

fn registry() -> ToolRegistry {
    ToolRegistry::new(vec![
        Arc::new(EchoTool { name: "echo" }),
        Arc::new(FailingTool { name: "boom" }),
    ])
}

fn config() -> AppConfig {
    AppConfig::default()
}

fn agent_with(steps: Vec<ScriptedStep>) -> Agent {
    let llm = Arc::new(ScriptedFunctionLlm::new(steps));
    Agent::new(llm, registry(), config())
}

// ─────────────────────────────────────────────────────────────────────
// 循环控制流
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn returns_the_final_answer_when_the_model_does_not_call_tools() {
    let agent = agent_with(vec![ScriptedStep::Text("答案是 42".to_string())]);

    let run = agent.run("多少个").await.unwrap();

    assert_eq!(run.answer, "答案是 42");
    assert_eq!(run.tool_calls, 0);
    assert_eq!(run.steps.len(), 1);
    assert!(run.steps[0].final_answer);
    assert!(!run.hit_max_steps);
}

#[tokio::test]
async fn executes_the_tool_and_feeds_the_result_back() {
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "echo", json!({ "value": "你好" })),
        ScriptedStep::Text("你说的是你好".to_string()),
    ]);

    let run = agent.run("打个招呼").await.unwrap();

    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.tools_used, vec!["echo".to_string()]);
    assert_eq!(run.steps[0].tool_calls[0].observation, "你好");
    assert!(run.steps[0].tool_calls[0].ok);
    assert_eq!(run.answer, "你说的是你好");
}

#[tokio::test]
async fn handles_parallel_tool_calls_in_one_turn() {
    // ⚠️ 这是 DeepSeek 的真实行为 —— 一轮返回多个 tool_calls。
    //    实测：「北京和上海天气」一次返回 2 个调用。
    let agent = agent_with(vec![
        ScriptedStep::calls(vec![
            ScriptedToolCall::new("c1", "echo", json!({ "value": "北京" })),
            ScriptedToolCall::new("c2", "echo", json!({ "value": "上海" })),
        ]),
        ScriptedStep::Text("北京和上海".to_string()),
    ]);

    let run = agent.run("两个城市").await.unwrap();

    assert_eq!(run.tool_calls, 2);
    assert_eq!(run.steps[0].tool_calls.len(), 2);
    assert_eq!(run.steps[0].tool_calls[0].call_id, "c1");
    assert_eq!(run.steps[0].tool_calls[1].call_id, "c2");
    assert_eq!(run.steps[0].tool_calls[0].observation, "北京");
    assert_eq!(run.steps[0].tool_calls[1].observation, "上海");
}

#[tokio::test]
async fn a_failing_tool_still_produces_a_receipt() {
    // ⚠️ 坑 1：工具失败**也必须有回执**。
    //
    //    DeepSeek 的报错原文：
    //    「An assistant message with 'tool_calls' must be followed by tool
    //     messages responding to each 'tool_call_id'.」
    //
    //    少一条回执，下一轮请求直接 400。
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "boom", json!({})),
        ScriptedStep::Text("工具失败了".to_string()),
    ]);

    let run = agent.run("调一个会炸的工具").await.unwrap();

    assert_eq!(run.tool_calls, 1);
    assert!(!run.steps[0].tool_calls[0].ok);
    // 错误文本必须包含可执行的信息。
    assert!(run.steps[0].tool_calls[0].observation.contains("工具炸了"));
}

#[tokio::test]
async fn a_failing_tool_does_not_break_the_loop() {
    // 工具失败后模型应该还能继续调工具。
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "boom", json!({})),
        ScriptedStep::call("c2", "echo", json!({ "value": "重试成功" })),
        ScriptedStep::Text("好了".to_string()),
    ]);

    let run = agent.run("先失败再成功").await.unwrap();

    assert_eq!(run.tool_calls, 2);
    assert!(!run.steps[0].tool_calls[0].ok);
    assert!(run.steps[1].tool_calls[0].ok);
    assert_eq!(run.answer, "好了");
}

#[tokio::test]
async fn invalid_arguments_are_reported_as_a_failure() {
    // 参数校验失败 —— 不该执行工具。
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "echo", json!({ "wrong": "字段" })),
        ScriptedStep::Text("参数错了".to_string()),
    ]);

    let run = agent.run("传错参数").await.unwrap();

    assert!(!run.steps[0].tool_calls[0].ok);
    assert!(run.steps[0].tool_calls[0].observation.contains("value"));
}

#[tokio::test]
async fn unknown_tool_is_reported_as_a_failure() {
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "不存在的工具", json!({})),
        ScriptedStep::Text("没有这个工具".to_string()),
    ]);

    let run = agent.run("调一个不存在的工具").await.unwrap();

    assert!(!run.steps[0].tool_calls[0].ok);
    // 错误里要列出可用工具 —— 模型才知道能调什么。
    assert!(run.steps[0].tool_calls[0].observation.contains("echo"));
}

#[tokio::test]
async fn string_encoded_arguments_are_normalized() {
    // ⚠️ 有些模型把 arguments 序列化成字符串。
    //    不转换的话本地校验会直接报「必须是 JSON 对象」。
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "echo", json!("{\"value\": \"解出来了\"}")),
        ScriptedStep::Text("好".to_string()),
    ]);

    let run = agent.run("参数是字符串").await.unwrap();

    assert!(run.steps[0].tool_calls[0].ok);
    assert_eq!(run.steps[0].tool_calls[0].observation, "解出来了");
}

#[tokio::test]
async fn degrades_when_the_step_limit_is_reached() {
    // ⚠️ 达到步数上限时**不能报错** ——
    //    前面几步辛苦收集的工具结果就白费了。
    let agent = agent_with(vec![
        ScriptedStep::call("c1", "echo", json!({ "value": "一" })),
        ScriptedStep::call("c2", "echo", json!({ "value": "二" })),
        ScriptedStep::call("c3", "echo", json!({ "value": "三" })),
        ScriptedStep::call("c4", "echo", json!({ "value": "四" })),
        ScriptedStep::call("c5", "echo", json!({ "value": "五" })),
    ]);

    let run = agent.run("一直调").await.unwrap();

    assert!(run.hit_max_steps);
    assert_eq!(run.tool_calls, 5);
    // 降级返回最后一步的观察，而不是报错。
    assert_eq!(run.answer, "五");
}

#[tokio::test]
async fn rejects_an_empty_question() {
    let agent = agent_with(vec![]);
    let err = agent.run("   ").await.unwrap_err();
    assert_eq!(err.status_code(), 400);
}

#[tokio::test]
async fn records_token_usage() {
    let agent = agent_with(vec![ScriptedStep::Text("答案".to_string())]);
    let run = agent.run("问题").await.unwrap();

    // ScriptedLlm 不报 token —— 但字段要存在且可累加。
    assert_eq!(run.usage.total_tokens, 0);
    assert_eq!(run.usage.prompt_tokens, 0);
}

#[tokio::test]
async fn reasoning_content_is_carried_in_the_step_record() {
    // ⚠️ 坑 2：推理模型返回的 reasoning_content 必须被记录，
    //    并且下一轮原样带回（否则 DeepSeek 会 400）。
    let agent = agent_with(vec![
        ScriptedStep::ReasoningAndCalls(
            "我需要调用工具".to_string(),
            vec![ScriptedToolCall::new(
                "c1",
                "echo",
                json!({ "value": "好" }),
            )],
        ),
        ScriptedStep::Text("完成".to_string()),
    ]);

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.steps[0].reasoning.as_deref(), Some("我需要调用工具"));
}

// ─────────────────────────────────────────────────────────────────────
// HTTP API 层
// ─────────────────────────────────────────────────────────────────────

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

fn app(steps: Vec<ScriptedStep>) -> axum::Router {
    let llm = Arc::new(ScriptedFunctionLlm::new(steps));
    let agent = Arc::new(Agent::new(llm.clone(), registry(), config()));
    let state = Arc::new(AppStateInner::new(config(), agent, llm));
    router(state)
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_endpoint_works() {
    let response = app(vec![]).oneshot(request_get("/health")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["status"], json!("ok"));
}

#[tokio::test]
async fn tools_endpoint_lists_tools() {
    let response = app(vec![])
        .oneshot(request_get("/api/tools"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["count"], json!(2));
    assert_eq!(json["tools"][0]["name"], json!("echo"));
    // 工具契约必须可检视 —— Schema 要在里面。
    assert!(json["tools"][0]["schema"]["properties"]["value"].is_object());
}

#[tokio::test]
async fn run_endpoint_returns_the_answer() {
    let app = app(vec![ScriptedStep::Text("最终答案".to_string())]);

    let response = app
        .oneshot(request_post("/api/run", json!({ "question": "问题" })))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["answer"], json!("最终答案"));
    assert_eq!(json["tool_calls"], json!(0));
}

#[tokio::test]
async fn run_endpoint_rejects_an_empty_question() {
    let app = app(vec![]);

    let response = app
        .oneshot(request_post("/api/run", json!({ "question": "  " })))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let json = body_json(response).await;
    assert_eq!(json["error"], json!("Bad Request"));
}

#[tokio::test]
async fn run_endpoint_rejects_a_malformed_body() {
    let app = app(vec![]);

    let response = app
        .oneshot(request_post_raw("/api/run", "{ 不是 json"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn call_tool_endpoint_calls_the_tool_directly() {
    let app = app(vec![]);

    let response = app
        .oneshot(request_post(
            "/api/tools/call",
            json!({ "tool": "echo", "args": { "value": "直接调用" } }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["ok"], json!(true));
    assert_eq!(json["observation"], json!("直接调用"));
}

#[tokio::test]
async fn call_tool_endpoint_returns_400_for_an_unknown_tool() {
    // ⚠️ 未知工具是**调用方**的问题 → 400，不是 500。
    let app = app(vec![]);

    let response = app
        .oneshot(request_post(
            "/api/tools/call",
            json!({ "tool": "nope", "args": {} }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn call_tool_endpoint_returns_400_for_bad_arguments() {
    let app = app(vec![]);

    let response = app
        .oneshot(request_post(
            "/api/tools/call",
            json!({ "tool": "echo", "args": { "wrong": 1 } }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn compare_endpoint_returns_four_modes() {
    // 四档对照：同一个模型实例，只有 tool_choice 变。
    let llm = Arc::new(ScriptedFunctionLlm::always(ScriptedStep::Text(
        "答案".to_string(),
    )));
    let agent = Arc::new(Agent::new(llm.clone(), registry(), config()));
    let state = Arc::new(AppStateInner::new(config(), agent, llm));
    let app = router(state);

    let response = app
        .oneshot(request_post("/api/compare", json!({ "question": "问题" })))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    let results = json["results"].as_array().unwrap();
    // auto / none / required / forced = 4 档。
    assert_eq!(results.len(), 4);
    assert_eq!(results[0]["mode"], json!("auto"));
    assert_eq!(results[1]["mode"], json!("none"));
    assert_eq!(results[2]["mode"], json!("required"));
    assert_eq!(results[3]["mode"], json!("forced"));
}

// ─────────────────────────────────────────────────────────────────────
// 辅助
// ─────────────────────────────────────────────────────────────────────

fn request_get(path: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap()
}

fn request_post(path: &str, payload: Value) -> Request<Body> {
    request_post_raw(path, &payload.to_string())
}

fn request_post_raw(path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[test]
fn tool_choice_modes_cover_the_api_contract() {
    // 四档必须和 API 文档一致。
    assert_eq!(agent_16::llm::tool_choice_name(&ToolChoice::Auto), "auto");
    assert_eq!(agent_16::llm::tool_choice_name(&ToolChoice::None), "none");
    assert_eq!(
        agent_16::llm::tool_choice_name(&ToolChoice::Required),
        "required"
    );
    assert_eq!(
        agent_16::llm::tool_choice_name(&ToolChoice::tool("x")),
        "forced"
    );
}
