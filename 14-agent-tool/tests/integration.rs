//! 集成测试 —— 验证工具契约与 Agent 循环的**完整行为**。
//!
//! ## 为什么这些测试能离线跑
//!
//! Agent 循环是一个**状态机**：每一步的输出决定下一步的输入。
//! 有了 [`ScriptedLlm`]，整个循环可以在**零网络、零延迟、完全确定**的
//! 条件下被测试 ——
//! 包括那些靠真实模型根本触发不了的边界路径
//! （比如「模型连续 6 次要求调工具」）。
//!
//! 工具层同理 —— 塞一个 `FakeTool` 进去，不碰网络。
//!
//! ## 覆盖的场景
//!
//! | 场景 | 验证什么 |
//! |---|---|
//! | 单次工具调用后作答 | 主路径 |
//! | 多次工具调用 | 轨迹累积正确 |
//! | 工具失败 | 错误作为 Observation 喂回模型 |
//! | 未知工具 | 同上，且提示可用工具列表 |
//! | 参数校验失败 | 错误可执行（指出哪个参数错） |
//! | 达到步数上限 | 兜底再问一次，不报错 |
//! | 直通工具 | 跳过最后一次模型调用 |
//! | 模式差异 | `direct` 不挂工具 |
//! | MCP 往返 | 客户端 ↔ 服务端全链路打通 |

use std::sync::Arc;

use agent_14::agent::Agent;
use agent_14::config::{AppConfig, ToolMode};
use agent_14::error::AppError;
use agent_14::llm::ScriptedLlm;
use agent_14::mcp::McpClient;
use agent_14::tool::{Tool, ToolContext, ToolFuture, ToolOutcome, ToolRegistry, ToolSpec};
use serde_json::{Value, json};

/// 一个确定性的假工具：回显。
struct EchoTool;

impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "echo",
            "回显文本。",
            json!({
                "type": "object",
                "properties": { "text": { "type": "string", "description": "要回显的文本" } },
                "required": ["text"]
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Box::pin(async move { Ok(ToolOutcome::ok("echo", format!("回显: {text}"))) })
    }
}

/// 一个总是失败的工具。
struct FailingTool;

impl Tool for FailingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("boom", "总是失败", json!({ "type": "object" }))
    }

    fn call<'a>(&'a self, _args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move { Err(agent_14::error::ToolError::execution("boom", "炸了")) })
    }
}

/// 一个带枚举参数的��具（用于测试 Schema 校验）。
struct EnumTool;

impl Tool for EnumTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "pick",
            "选一个颜色。",
            json!({
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": ["red", "green"],
                        "description": "颜色"
                    }
                },
                "required": ["color"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        let color = args
            .get("color")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Box::pin(async move { Ok(ToolOutcome::ok("pick", format!("选了 {color}"))) })
    }
}

/// 建一个带假工具的 Agent。
fn agent_with(answers: Vec<String>, max_steps: usize) -> (Agent, Arc<ScriptedLlm>) {
    let llm = Arc::new(ScriptedLlm::new(answers));

    let mut config = AppConfig::default();
    config.agent.max_steps = max_steps;

    let registry = ToolRegistry::new(vec![Arc::new(EchoTool), Arc::new(FailingTool)]);

    (Agent::new(llm.clone(), registry, config), llm)
}

#[tokio::test]
async fn single_tool_call_then_final_answer() {
    let (agent, _llm) = agent_with(
        vec![
            "Thought: 需要回显\nAction: echo\nAction Input: {\"text\": \"你好\"}".to_string(),
            "Thought: 信息足够了\nFinal Answer: 已经回显过了".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.steps.len(), 2);
    assert!(run.answer.contains("已经回显过了"));
    assert!(!run.hit_max_steps);
    assert_eq!(run.tools_used, vec!["echo".to_string()]);
}

#[tokio::test]
async fn trajectory_accumulates_across_steps() {
    let (agent, _llm) = agent_with(
        vec![
            "Thought: 先回显\nAction: echo\nAction Input: {\"text\": \"a\"}".to_string(),
            "Thought: 再回显\nAction: echo\nAction Input: {\"text\": \"b\"}".to_string(),
            "Final Answer: 完成".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.tool_calls, 2);
    assert_eq!(run.steps.len(), 3);

    let trajectory = run.trajectory();
    assert!(trajectory.contains("Thought: 先回显"));
    assert!(trajectory.contains("Action: echo"));
    assert!(trajectory.contains("Observation: 回显: a"));
    assert!(trajectory.contains("Final Answer: 完成"));
}

#[tokio::test]
async fn observation_is_fed_back_to_the_model() {
    let (agent, llm) = agent_with(
        vec![
            "Thought: 回显\nAction: echo\nAction Input: {\"text\": \"内容\"}".to_string(),
            "Final Answer: 答案".to_string(),
        ],
        5,
    );

    agent.run("问题").await.unwrap();

    // 第二轮调用的 user 提示里必须包含第一轮的 Observation。
    let calls = llm.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].1.contains("Observation:"));
    assert!(
        calls[1].1.contains("回显: 内容"),
        "Observation 必须带上工具返回的实际内容"
    );
}

#[tokio::test]
async fn tool_failure_is_reported_as_observation_not_error() {
    let (agent, llm) = agent_with(
        vec![
            // 缺少 text 参数 → 工具失败。
            "Thought: 回显\nAction: echo\nAction Input: {}".to_string(),
            "Final Answer: 我看到工具报错了".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    // 关键：整个运行**没有失败**。
    assert_eq!(run.steps[0].tool_ok, Some(false));
    let observation = run.steps[0].observation.as_ref().unwrap();
    assert!(observation.contains("text"));
    assert!(observation.contains("必填"));

    // 错误文本必须进了下一轮的提示词。
    let calls = llm.calls();
    assert!(calls[1].1.contains("必填"));
}

#[tokio::test]
async fn unknown_tool_lists_available_tools() {
    let (agent, llm) = agent_with(
        vec![
            "Thought: 用不存在的工具\nAction: nonexistent\nAction Input: {}".to_string(),
            "Final Answer: 好".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.steps[0].tool_ok, Some(false));
    let observation = run.steps[0].observation.as_ref().unwrap();
    assert!(observation.contains("未知工具"));
    // 必须告诉模型有哪些工具可用 —— 它要靠这个自我纠正。
    assert!(observation.contains("echo"));
    assert!(observation.contains("boom"));

    let calls = llm.calls();
    assert!(calls[1].1.contains("未知工具"));
}

#[tokio::test]
async fn enum_violation_becomes_actionable_observation() {
    // ⚠️ 工具契约的核心：参数错了，模型要能**立刻**知道怎么改。
    let llm = Arc::new(ScriptedLlm::new(vec![
        "Thought: 选颜色\nAction: pick\nAction Input: {\"color\": \"blue\"}".to_string(),
        "Final Answer: 我改用 red".to_string(),
    ]));

    let mut config = AppConfig::default();
    config.agent.max_steps = 5;

    let registry = ToolRegistry::new(vec![Arc::new(EnumTool)]);
    let agent = Agent::new(llm.clone(), registry, config);

    let run = agent.run("选个颜色").await.unwrap();

    assert_eq!(run.steps[0].tool_ok, Some(false));
    let observation = run.steps[0].observation.as_ref().unwrap();
    assert!(observation.contains("red"));
    assert!(observation.contains("green"));
    assert!(observation.contains("blue"));
}

#[tokio::test]
async fn hitting_max_steps_triggers_a_final_graceful_answer() {
    // 模型一直要求调工具，永远不给 Final Answer。
    let (agent, llm) = agent_with(
        vec![
            "Action: echo\nAction Input: {\"text\": \"a\"}".to_string(),
            "Action: echo\nAction Input: {\"text\": \"b\"}".to_string(),
            "Action: echo\nAction Input: {\"text\": \"c\"}".to_string(),
        ],
        3,
    );

    let run = agent.run("问题").await.unwrap();

    // 达到上限 → 但**不报错**。
    assert!(run.hit_max_steps);
    assert_eq!(run.tool_calls, 3);
    assert!(!run.answer.is_empty());

    // 兜底会**再问一次** —— 共 4 次模型调用（3 步 + 1 次收尾）。
    assert_eq!(llm.call_count(), 4);
}

#[tokio::test]
async fn direct_mode_never_calls_tools() {
    let llm = Arc::new(ScriptedLlm::new(vec!["Final Answer: 直接回答".to_string()]));
    let mut config = AppConfig::default();
    config.agent.max_steps = 5;

    let agent = Agent::new(llm.clone(), ToolRegistry::new(Vec::new()), config);

    let run = agent.run_with_mode("问题", ToolMode::Direct).await.unwrap();

    assert_eq!(run.tool_calls, 0);
    assert_eq!(run.answer, "直接回答");

    // 提示词里不能出现工具。
    let calls = llm.calls();
    assert!(!calls[0].1.contains("Action:"));
    assert!(!calls[0].0.contains("echo"));
}

#[tokio::test]
async fn act_only_mode_has_no_thought_in_prompt() {
    let (agent, llm) = agent_with(vec!["Final Answer: 答案".to_string()], 5);

    agent
        .run_with_mode("问题", ToolMode::ActOnly)
        .await
        .unwrap();

    let calls = llm.calls();
    assert!(!calls[0].1.contains("Thought:"));
    assert!(calls[0].1.contains("Action:"));
}

#[tokio::test]
async fn react_mode_has_thought_in_prompt() {
    let (agent, llm) = agent_with(vec!["Final Answer: 答案".to_string()], 5);

    agent.run_with_mode("问题", ToolMode::ReAct).await.unwrap();

    let calls = llm.calls();
    assert!(calls[0].1.contains("Thought:"));
}

#[tokio::test]
async fn empty_question_is_rejected() {
    let (agent, _llm) = agent_with(vec![], 5);

    let err = agent.run("   ").await.unwrap_err();
    assert!(matches!(err, AppError::BadRequest(_)));
}

#[tokio::test]
async fn tool_call_can_be_made_directly_without_the_model() {
    // ⚠️ 直接调工具：不碰模型、不花 token、结果确定。
    let (agent, llm) = agent_with(vec![], 5);

    let outcome = agent
        .registry()
        .call_strict("echo", &json!({ "text": "直连" }), &ToolContext::empty())
        .await
        .unwrap();

    assert!(outcome.ok);
    assert!(outcome.observation.contains("直连"));

    // 模型一次都没被调用。
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn direct_tool_call_surfaces_real_errors() {
    // ⚠️ 直接调工具时，错误**原样返回** —— 因为用户扮演了模型角色。
    let (agent, _llm) = agent_with(vec![], 5);

    let err = agent
        .registry()
        .call_strict("echo", &json!({}), &ToolContext::empty())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("text"));
}

// ── MCP 往返测试 ─────────────────────────────────────────────────────
//
// 起一个**真的** MCP 服务器（axum），用**真的** MCP 客户端连上去。
// 这是本项目对 MCP 协议实现正确性的端到端验证。

/// 起一个内置 MCP 服务器，返回它的地址。
async fn spawn_mcp_server() -> (String, tokio::task::JoinHandle<()>) {
    use agent_14::mcp_server::McpServerState;

    let state = McpServerState::new();
    let app = agent_14::mcp_server::router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{}/mcp", addr), handle)
}

#[tokio::test]
async fn mcp_client_lists_tools_from_a_real_server() {
    let (endpoint, _server) = spawn_mcp_server().await;

    let client = McpClient::new(&endpoint);
    let tools = client.list_tools().await.unwrap();

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"echo"));
    assert!(names.contains(&"get_weather"));

    // 会话 ID 应该在握手里拿到了。
    assert!(client.session_id().is_some());
}

#[tokio::test]
async fn mcp_client_calls_a_tool_and_gets_text_back() {
    let (endpoint, _server) = spawn_mcp_server().await;

    let client = McpClient::new(&endpoint);
    let result = client
        .call_tool("echo", &json!({ "text": "你好" }))
        .await
        .unwrap();

    assert!(!result.is_error);
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(text, "回显: 你好");
}

#[tokio::test]
async fn mcp_tool_error_result_is_reported_as_error() {
    let (endpoint, _server) = spawn_mcp_server().await;

    let client = McpClient::new(&endpoint);
    // 缺少必填参数 → 服务端返回 `isError: true`。
    let result = client.call_tool("echo", &json!({})).await.unwrap();

    assert!(result.is_error);
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("text"));
}

#[tokio::test]
async fn mcp_tools_are_interchangeable_with_local_tools() {
    // ⚠️ 这是本项目最核心的断言：
    //    MCP 工具和本地工具在 Agent 眼里**完全一样**。
    let (endpoint, _server) = spawn_mcp_server().await;

    let client = Arc::new(McpClient::new(&endpoint));
    client.list_tools().await.unwrap();

    let mcp_tools = client.as_tools();
    assert_eq!(mcp_tools.len(), 2);

    // MCP 工具的 Schema 和本地天气工具**完全一致**。
    let weather_spec = mcp_tools
        .iter()
        .find(|t| t.spec().name == "get_weather")
        .map(|t| t.spec())
        .unwrap();

    assert_eq!(weather_spec.schema, agent_14::weather::weather_schema());
    assert!(!weather_spec.description.is_empty());

    // 合并进注册表 —— 和本地工具并列。
    //
    // ⚠️ 注意：本地也有一个叫 `echo` 的工具，MCP 那边也有。
    //    `merge` 的语义是**同名时后者覆盖前者** ——
    //    所以结果还是 2 个工具（`echo` 被 MCP 版本替换了）。
    let registry = ToolRegistry::new(vec![Arc::new(EchoTool)]).merge(&ToolRegistry::new(mcp_tools));

    let names = registry.names();
    assert!(names.iter().any(|n| n == "echo"));
    assert!(names.iter().any(|n| n == "get_weather"));
    assert_eq!(registry.len(), 2, "同名工具被覆盖，不应重复");

    // 覆盖后生效的是 MCP 版本的 `echo`（它来自远端）。
    let echo = registry.get("echo").unwrap();
    assert_eq!(
        echo.spec().description,
        "回显一段文本。用于验证 MCP 链路是否通畅。"
    );
}

#[tokio::test]
async fn mcp_tool_can_be_called_through_the_registry() {
    let (endpoint, _server) = spawn_mcp_server().await;

    let client = Arc::new(McpClient::new(&endpoint));
    client.list_tools().await.unwrap();

    let registry = ToolRegistry::new(client.as_tools());

    let outcome = registry
        .call(
            "echo",
            &json!({ "text": "通过注册表" }),
            &ToolContext::empty(),
        )
        .await;

    assert!(outcome.ok);
    assert!(outcome.observation.contains("通过注册表"));
}

// ── HTTP API 层测试 ─────────────────────────────────────────────────
//
// 用 `tower::ServiceExt::oneshot` 直接调路由 —— 不开端口，零 IO 开销。

/// 造一个带假工具的 API 状态。
fn api_state(answers: Vec<String>) -> agent_14::api::AppState {
    use agent_14::api::AppStateInner;

    let llm = Arc::new(ScriptedLlm::new(answers));

    let mut config = AppConfig::default();
    config.agent.max_steps = 5;

    let registry = ToolRegistry::new(vec![Arc::new(EchoTool), Arc::new(FailingTool)]);
    let agent = Agent::new(llm, registry, config.clone());

    Arc::new(AppStateInner::new(config, Arc::new(agent), None))
}

/// 发一个请求，返回状态码和响应体。
async fn request(
    state: agent_14::api::AppState,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (u16, Value) {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let app = agent_14::api::routes::router(state);

    let mut builder = Request::builder().method(method).uri(path);
    let request = match body {
        Some(value) => {
            builder = builder.header("content-type", "application/json");
            builder.body(Body::from(value.to_string())).unwrap()
        }
        None => builder.body(Body::empty()).unwrap(),
    };

    let response = app.oneshot(request).await.unwrap();
    let status = response.status().as_u16();

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);

    (status, value)
}

#[tokio::test]
async fn health_endpoint_responds_ok() {
    let state = api_state(vec![]);
    let (status, body) = request(state, "GET", "/health", None).await;

    assert_eq!(status, 200);
    assert_eq!(body["status"], json!("ok"));
}

#[tokio::test]
async fn tools_endpoint_exposes_the_full_contract() {
    let state = api_state(vec![]);
    let (status, body) = request(state, "GET", "/api/tools", None).await;

    assert_eq!(status, 200);
    assert_eq!(body["count"], json!(2));

    let tools = body["tools"].as_array().unwrap();
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .collect();
    assert!(names.contains(&"echo"));

    // 每个工具都要有完整契约：描述 / Schema / 渲染后的参数说明。
    for tool in tools {
        assert!(!tool["description"].as_str().unwrap().is_empty());
        assert!(tool["schema"].is_object());
        assert!(tool["parameters"].is_string());
    }
}

#[tokio::test]
async fn call_tool_endpoint_returns_observation() {
    let state = api_state(vec![]);
    let (status, body) = request(
        state,
        "POST",
        "/api/tools/call",
        Some(json!({ "tool": "echo", "args": { "text": "你好" } })),
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(body["ok"], json!(true));
    assert!(body["observation"].as_str().unwrap().contains("你好"));
}

#[tokio::test]
async fn call_tool_endpoint_maps_argument_errors_to_400() {
    let state = api_state(vec![]);
    let (status, body) = request(
        state,
        "POST",
        "/api/tools/call",
        Some(json!({ "tool": "echo", "args": {} })),
    )
    .await;

    // ⚠️ 参数填错是**调用方**的问题 → 400（不是 500）。
    assert_eq!(status, 400);
    assert!(body["message"].as_str().unwrap().contains("text"));
}

#[tokio::test]
async fn call_tool_endpoint_maps_unknown_tool_to_400() {
    let state = api_state(vec![]);
    let (status, body) = request(
        state,
        "POST",
        "/api/tools/call",
        Some(json!({ "tool": "nope", "args": {} })),
    )
    .await;

    assert_eq!(status, 400);
    let message = body["message"].as_str().unwrap();
    assert!(message.contains("未知工具"));
    // 必须列出可用工具。
    assert!(message.contains("echo"));
}

#[tokio::test]
async fn run_endpoint_returns_full_trajectory() {
    let state = api_state(vec![
        "Thought: 回显\nAction: echo\nAction Input: {\"text\": \"hi\"}".to_string(),
        "Final Answer: 好了".to_string(),
    ]);

    let (status, body) = request(
        state,
        "POST",
        "/api/run",
        Some(json!({ "question": "问题" })),
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(body["answer"], json!("好了"));
    assert_eq!(body["tool_calls"], json!(1));
    assert_eq!(body["tools_used"], json!(["echo"]));
    assert!(
        body["trajectory"]
            .as_str()
            .unwrap()
            .contains("Observation:")
    );
    // 每一步都要有详细记录。
    assert_eq!(body["steps"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn run_endpoint_rejects_empty_question() {
    let state = api_state(vec![]);
    let (status, body) = request(
        state,
        "POST",
        "/api/run",
        Some(json!({ "question": "   " })),
    )
    .await;

    assert_eq!(status, 400);
    assert!(body["message"].as_str().unwrap().contains("问题"));
}

#[tokio::test]
async fn run_endpoint_rejects_unknown_mode() {
    let state = api_state(vec![]);
    let (status, body) = request(
        state,
        "POST",
        "/api/run",
        Some(json!({ "question": "问题", "mode": "bogus" })),
    )
    .await;

    assert_eq!(status, 400);
    assert!(body["message"].as_str().unwrap().contains("bogus"));
}

#[tokio::test]
async fn compare_endpoint_runs_all_three_modes() {
    // ⚠️ 三模式共用**同一个** `ScriptedLlm` —— 脚本要按顺序覆盖所有调用：
    //   - `direct`   : 1 次调用（直接给 Final Answer）
    //   - `act_only` : 2 次调用（1 次动作 + 1 次收尾）
    //   - `react`    : 2 次调用（1 次动作 + 1 次收尾）
    let state = api_state(vec![
        "Final Answer: 直答".to_string(),
        "Action: echo\nAction Input: {\"text\": \"a\"}".to_string(),
        "Final Answer: act_only 答案".to_string(),
        "Action: echo\nAction Input: {\"text\": \"b\"}".to_string(),
        "Final Answer: react 答案".to_string(),
    ]);

    let (status, body) = request(
        state,
        "POST",
        "/api/compare",
        Some(json!({ "question": "问题" })),
    )
    .await;

    assert_eq!(status, 200);
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    // 顺序：direct → act_only → react。
    assert_eq!(results[0]["mode"], json!("direct"));
    assert_eq!(results[1]["mode"], json!("act_only"));
    assert_eq!(results[2]["mode"], json!("react"));

    // `direct` 不调工具。
    assert_eq!(results[0]["tool_calls"], json!(0));
    // 另外两个调了。
    assert_eq!(results[1]["tool_calls"], json!(1));
    assert_eq!(results[2]["tool_calls"], json!(1));

    // 摘要要有人类可读的结论。
    let summary = body["summary"].as_array().unwrap();
    assert_eq!(summary.len(), 3);
    assert!(summary[0].as_str().unwrap().contains("direct"));
}

#[tokio::test]
async fn parse_endpoint_classifies_model_output() {
    let state = api_state(vec![]);

    // Action。
    let (status, body) = request(
        state.clone(),
        "POST",
        "/api/parse",
        Some(json!({
            "output": "Thought: 查\nAction: echo\nAction Input: {\"text\": \"x\"}"
        })),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["kind"], json!("action"));
    assert_eq!(body["tool"], json!("echo"));
    assert_eq!(body["args"]["text"], json!("x"));

    // Final Answer。
    let (_, body) = request(
        state.clone(),
        "POST",
        "/api/parse",
        Some(json!({ "output": "Final Answer: 42" })),
    )
    .await;
    assert_eq!(body["kind"], json!("final_answer"));
    assert_eq!(body["answer"], json!("42"));

    // Unstructured。
    let (_, body) = request(
        state,
        "POST",
        "/api/parse",
        Some(json!({ "output": "我觉得是 42" })),
    )
    .await;
    assert_eq!(body["kind"], json!("unstructured"));
    assert!(body["parse_error"].is_string());
}

#[tokio::test]
async fn malformed_json_body_returns_400() {
    let state = api_state(vec![]);

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let app = agent_14::api::routes::router(state);
    let request = Request::builder()
        .method("POST")
        .uri("/api/run")
        .header("content-type", "application/json")
        .body(Body::from("{ 不是 JSON"))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 400);
}
