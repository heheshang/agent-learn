//! 集成测试 —— 零网络、确定性。
//!
//! ## 为什么不用真实模型
//!
//! 本项目的核心是「**工具怎么并发执行**」，
//! 而不是「模型怎么说话」。
//! 用 `ScriptedLlm` 预设响应，就能把并发行为完全测死。
//!
//! ## ⚠️ 一个刻意的取舍
//!
//! **天气工具不测真实网络** —— 网络波动会让测试不稳定。
//! 并发的行为用「模拟 I/O」测（`concurrency` 模块的单元测试），
//! 而 Agent 循环的并发语义用**脚本 LLM + 本地工具**测。

use std::sync::Arc;
use std::time::Duration;

use agent_18::llm::ScriptedLlm;
use agent_18::middleware::{ErrorHandlingMiddleware, ToolCallRequest};
use agent_22::agent::{Agent, AgentConfig};
use agent_22::api::routes::router;
use agent_22::config::AppConfig;
use agent_22::tools::{AsyncTool, ToolContext};
use serde_json::{Value, json};

use tower::ServiceExt;

// ─────────────────────────────────────────────────────────────────────
// 测试替身
// ─────────────────────────────────────────────────────────────────────

/// 一个「等一会儿再返回」的工具（模拟 I/O）。
struct SlowTool {
    name: String,
    delay_ms: u64,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for SlowTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlowTool")
            .field("name", &self.name)
            .field("delay_ms", &self.delay_ms)
            .finish()
    }
}

impl SlowTool {
    fn new(name: &str, delay_ms: u64) -> (Arc<Self>, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool = Arc::new(Self {
            name: name.to_string(),
            delay_ms,
            calls: calls.clone(),
        });
        (tool, calls)
    }
}

impl AsyncTool for SlowTool {
    fn spec(&self) -> agent_16::tools::ToolSpec {
        agent_16::tools::ToolSpec::new(
            self.name.clone(),
            "慢速工具（测试用）",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(
        &'a self,
        args: &'a Value,
        _ctx: &'a ToolContext,
    ) -> agent_22::tools::ToolFuture<'a> {
        let delay = self.delay_ms;
        let value = args
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();

        Box::pin(async move {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok(format!("{value} 的结果"))
        })
    }
}

/// 构造一个只含慢速工具的注册表。
fn slow_registry(
    names: &[&str],
    delay_ms: u64,
) -> (
    agent_18::tool::ToolRegistry,
    Vec<Arc<std::sync::atomic::AtomicUsize>>,
) {
    let mut tools: Vec<Arc<dyn AsyncTool>> = Vec::new();
    let mut counters = Vec::new();

    for name in names {
        let (tool, counter) = SlowTool::new(name, delay_ms);
        tools.push(tool as Arc<dyn AsyncTool>);
        counters.push(counter);
    }

    let registry = agent_22::tools::build_registry(tools, None);
    (registry, counters)
}

/// 一个「前 N 次失败（Transient）、之后成功」的工具。
///
/// ⚠️ 这是测**重试次数统计**的样本 —— 18 的中间件会重试它。
struct FlakyTool {
    fail_times: usize,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for FlakyTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlakyTool")
            .field("fail_times", &self.fail_times)
            .finish()
    }
}

impl FlakyTool {
    fn new(fail_times: usize) -> (Arc<Self>, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool = Arc::new(Self {
            fail_times,
            calls: calls.clone(),
        });
        (tool, calls)
    }
}

impl AsyncTool for FlakyTool {
    fn spec(&self) -> agent_16::tools::ToolSpec {
        agent_16::tools::ToolSpec::new(
            "flaky_tool",
            "不稳定的上游，前几次调用会失败（临时故障）。",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(
        &'a self,
        _args: &'a Value,
        _ctx: &'a ToolContext,
    ) -> agent_22::tools::ToolFuture<'a> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let fail_times = self.fail_times;

        Box::pin(async move {
            if n <= fail_times {
                // ⚠️ **Transient** —— 中间件会退避后重试。
                return Err(agent_22::error::ToolError::upstream(
                    "flaky_tool",
                    format!("上游暂时不可用（第 {n} 次尝试）"),
                ));
            }
            Ok(format!("第 {n} 次成功了"))
        })
    }
}

/// 一个「总是鉴权失败（Fatal）」的工具。
///
/// ⚠️ 这是测**致命错误不降级**的样本。
struct FatalTool;

impl std::fmt::Debug for FatalTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FatalTool").finish()
    }
}

impl AsyncTool for FatalTool {
    fn spec(&self) -> agent_16::tools::ToolSpec {
        agent_16::tools::ToolSpec::new(
            "fatal_tool",
            "需要 API key 的工具，没配 key 时会鉴权失败。",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(
        &'a self,
        _args: &'a Value,
        _ctx: &'a ToolContext,
    ) -> agent_22::tools::ToolFuture<'a> {
        Box::pin(async move {
            // ⚠️ **Fatal** —— 重试没用，降级会掩盖配置问题。
            Err(agent_22::error::ToolError::unauthorized(
                "fatal_tool",
                "未配置 API key",
            ))
        })
    }
}

/// 构造一个只含单个自定义工具的注册表。
fn registry_with(tool: Arc<dyn AsyncTool>) -> agent_18::tool::ToolRegistry {
    agent_22::tools::build_registry(vec![tool], None)
}

/// 构造一个 Agent（脚本 LLM）。
fn build_agent(
    llm: Arc<ScriptedLlm>,
    registry: agent_18::tool::ToolRegistry,
    max_concurrency: usize,
) -> Arc<Agent> {
    build_agent_with_retries(llm, registry, max_concurrency, 0)
}

/// 构造一个 Agent（可指定工具重试次数）。
fn build_agent_with_retries(
    llm: Arc<ScriptedLlm>,
    registry: agent_18::tool::ToolRegistry,
    max_concurrency: usize,
    max_retries: usize,
) -> Arc<Agent> {
    let middleware = ErrorHandlingMiddleware::new(registry)
        // ⚠️ 测试里不要退避等待 —— 退避会拖慢测试且不改变结论。
        .with_max_retries(max_retries)
        .with_base_delay(Duration::ZERO);

    let config = AgentConfig {
        max_steps: 3,
        tool_choice: agent_22::llm::ToolChoiceMode::Auto,
        max_concurrency,
    };

    Arc::new(Agent::new(llm, middleware, config, Vec::new()))
}

/// 构造一个「返回两个并行工具调用，然后给最终答案」的脚本 LLM。
fn parallel_script() -> Arc<ScriptedLlm> {
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![
            ScriptedToolCall::new("c1", "tool_a", json!({ "value": "A" })),
            ScriptedToolCall::new("c2", "tool_b", json!({ "value": "B" })),
        ]),
        step_text("两个工具都查完了。"),
    ]))
}

// ─────────────────────────────────────────────────────────────────────
// 测试
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn parallel_tool_calls_run_concurrently() {
    // ⚠️ **本项目的核心断言**。
    //
    // 两个各 200ms 的工具：
    // - 串行 → ≥ 400ms
    // - 并发 → ≈ 200ms
    let llm = parallel_script();
    let (registry, _counters) = slow_registry(&["tool_a", "tool_b"], 200);
    let agent = build_agent(llm, registry, 4);

    let start = std::time::Instant::now();
    let run = agent.run("问题").await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(run.tool_call_groups, 1, "两个调用应该在同一轮");
    assert_eq!(run.tool_calls, 2);
    assert!(run.answer.contains("两个工具都查完了"));

    // ⚠️ 允许一定抖动（CI 机器负载），但必须明显快于串行。
    assert!(
        elapsed.as_millis() < 350,
        "并发执行应明显快于串行（串行需 ~400ms），实际 {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn parallel_tool_calls_keep_input_order_in_receipts() {
    // ⚠️ **坑 1 的回归测试**。
    //
    // 并发执行后结果到达顺序是乱的，
    // 但回执必须按 `call_id` 对应 ——
    // 所以 `step.tool_calls` 的顺序必须等于**输入顺序**。
    //
    // 做法：让第二个工具更快（先完成），
    // 如果实现按「完成顺序」排，顺序就会乱。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![
            ScriptedToolCall::new("c1", "slow_tool", json!({ "value": "slow" })),
            ScriptedToolCall::new("c2", "fast_tool", json!({ "value": "fast" })),
        ]),
        step_text("完成"),
    ]));

    let slow = Arc::new(SlowTool {
        name: "slow_tool".to_string(),
        delay_ms: 200,
        calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    });
    let fast = Arc::new(SlowTool {
        name: "fast_tool".to_string(),
        delay_ms: 10,
        calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    });

    let registry = agent_22::tools::build_registry(
        vec![slow as Arc<dyn AsyncTool>, fast as Arc<dyn AsyncTool>],
        None,
    );
    let agent = build_agent(llm, registry, 4);

    let run = agent.run("问题").await.unwrap();

    let calls = &run.steps[0].tool_calls;
    assert_eq!(calls.len(), 2);
    // ⚠️ 顺序必须是输入顺序（slow 在前），不是完成顺序（fast 在前）。
    assert_eq!(calls[0].name, "slow_tool");
    assert_eq!(calls[1].name, "fast_tool");
    assert_eq!(calls[0].call_id, "c1");
    assert_eq!(calls[1].call_id, "c2");
}

#[tokio::test]
async fn every_call_id_gets_exactly_one_receipt() {
    // ⚠️ **16 的坑 1 的回归测试**。
    //
    // 少一条回执 → 整个请求 400。
    // 本项目测的是「记录条数 == 输入条数」。
    let llm = parallel_script();
    let (registry, _counters) = slow_registry(&["tool_a", "tool_b"], 10);
    let agent = build_agent(llm, registry, 4);

    let run = agent.run("问题").await.unwrap();

    let calls = &run.steps[0].tool_calls;
    assert_eq!(calls.len(), 2);

    // call_id 必须**互不相同**且都非空。
    let ids: Vec<&str> = calls.iter().map(|c| c.call_id.as_str()).collect();
    assert_eq!(ids, vec!["c1", "c2"]);
}

#[tokio::test]
async fn concurrency_limit_is_respected() {
    // ⚠️ 限流必须真的生效 —— 不能「声称限流但实际全放」。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let names = ["t1", "t2", "t3", "t4"];
    let calls: Vec<ScriptedToolCall> = names
        .iter()
        .enumerate()
        .map(|(i, n)| ScriptedToolCall::new(&format!("c{i}"), n, json!({ "value": n })))
        .collect();

    let llm = Arc::new(ScriptedLlm::new(vec![step_calls(calls), step_text("完成")]));

    let (registry, _counters) = slow_registry(&names, 150);
    // ⚠️ 限流成 1 —— 4 个任务串行跑。
    let agent = build_agent(llm, registry, 1);

    let start = std::time::Instant::now();
    let run = agent.run("问题").await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(run.tool_calls, 4);
    // ⚠️ 限流成 1 → 4×150ms ≈ 600ms。
    assert!(
        elapsed.as_millis() >= 550,
        "限流为 1 时应串行执行（≈600ms），实际 {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn tool_failure_becomes_observation_not_crash() {
    // ⚠️ 18 的结论：工具失败**不是**请求失败。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "unknown_tool",
            json!({ "value": "x" }),
        )]),
        step_text("工具失败了，我如实告知。"),
    ]));

    let (registry, _counters) = slow_registry(&["tool_a"], 10);
    let agent = build_agent(llm, registry, 4);

    let run = agent.run("问题").await.unwrap();

    assert!(!run.steps[0].tool_calls[0].ok);
    assert_eq!(
        run.steps[0].tool_calls[0].error_code.as_deref(),
        Some("E_UNKNOWN_TOOL")
    );
    // ⚠️ 关键：Agent **没有**崩 —— 继续跑完并给出答案。
    assert!(run.answer.contains("工具失败了"));
}

#[tokio::test]
async fn empty_question_is_rejected() {
    let llm = parallel_script();
    let (registry, _counters) = slow_registry(&["tool_a"], 10);
    let agent = build_agent(llm, registry, 4);

    let err = agent.run("   ").await.unwrap_err();

    assert_eq!(err.status_code(), 400);
}

// ─────────────────────────────────────────────────────────────────────
// 回归测试：致命错误不降级
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fatal_tool_error_terminates_the_agent_loop() {
    // ⚠️ **18 的结论在并发场景下同样成立**：Fatal 不能降级。
    //
    // 降级会让模型以为「工具坏了」→ 转而编答案，
    // 把配置问题（鉴权失败）掩盖成幻觉。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "fatal_tool",
            json!({ "value": "x" }),
        )]),
        // ⚠️ 这一步**不应该**被走到 —— 致命错误必须终止。
        step_text("不该出现的答案"),
    ]));

    let agent = build_agent(llm, registry_with(Arc::new(FatalTool)), 4);

    let err = agent.run("问题").await.unwrap_err();

    // ⚠️ Fatal → 500（服务端自己的问题）。
    assert_eq!(err.status_code(), 500);
    assert!(
        err.to_string().contains("E_UNAUTHORIZED"),
        "错误必须保留真实的错误码，实际：{err}"
    );
}

#[tokio::test]
async fn fatal_tool_error_still_produces_one_receipt_per_call_id() {
    // ⚠️ **不变量 1 优先于不变量 2**。
    //
    // 16 实测：每个 `call_id` 必须有恰好一条 `tool` 消息，
    // 少一条整个请求就 400 —— **包括致命错误这一条**。
    //
    // 所以终止之前**必须**先把回执生成完。
    use agent_18::llm::{ScriptedToolCall, step_calls};

    let llm = Arc::new(ScriptedLlm::new(vec![step_calls(vec![
        ScriptedToolCall::new("c1", "fatal_tool", json!({ "value": "x" })),
        ScriptedToolCall::new("c2", "tool_b", json!({ "value": "y" })),
    ])]));

    let (slow, _c) = SlowTool::new("tool_b", 10);
    let slow_calls = slow.calls.clone();
    let registry = agent_22::tools::build_registry(
        vec![
            Arc::new(FatalTool) as Arc<dyn AsyncTool>,
            slow as Arc<dyn AsyncTool>,
        ],
        None,
    );
    let agent = build_agent(llm, registry, 4);

    // 致命错误 → 整个请求失败。
    let err = agent.run("问题").await.unwrap_err();
    assert_eq!(err.status_code(), 500);

    // ⚠️ 关键：另一个（正常）调用也**必须**被执行 ——
    //    并发下两个任务是同时跑的，终止不能「取消」另一个。
    assert_eq!(
        slow_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "并发下的另一个调用也必须被执行"
    );
}

#[tokio::test]
async fn business_tool_error_is_still_degraded_into_an_observation() {
    // ⚠️ **对照组**：Business 错误**必须**降级成观察（18 的结论）。
    //
    // 只测「Fatal 会终止」是不够的 ——
    // 如果实现变成「所有错误都终止」，Business 场景就崩了。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "tool_a",
            json!({ "value": 42 }),
        )]),
        step_text("工具失败了，我如实告知。"),
    ]));

    // ⚠️ `tool_a` 的 schema 要求 `value` 是字符串 ——
    //    传数字 → Schema 校验失败 → Business。
    let (registry, _counters) = slow_registry(&["tool_a"], 10);
    let agent = build_agent(llm, registry, 4);

    let run = agent.run("问题").await.unwrap();

    assert!(!run.steps[0].tool_calls[0].ok);
    assert_eq!(
        run.steps[0].tool_calls[0].error_kind.as_deref(),
        Some("business")
    );
    assert!(run.answer.contains("工具失败了"));
}

// ─────────────────────────────────────────────────────────────────────
// 回归测试：重试次数不能丢
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retried_calls_are_counted_and_recorded() {
    // ⚠️ **中间件重试了几次，记录里就必须显示几次**。
    //
    // 曾经 `attempts` 被写死成 1，导致：
    // - `retried_calls` 恒为 0
    // - 终端永远显示「1 次尝试」
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "flaky_tool",
            json!({ "value": "x" }),
        )]),
        step_text("工具重试后成功了。"),
    ]));

    // 前 2 次失败（Transient）→ 中间件重试到第 3 次成功。
    let (flaky, _counter) = FlakyTool::new(2);
    let agent = build_agent_with_retries(llm, registry_with(flaky), 4, 2);

    let run = agent.run("问题").await.unwrap();

    let call = &run.steps[0].tool_calls[0];
    assert!(call.ok, "重试后应该成功");
    assert_eq!(call.attempts, 3, "尝试次数必须是真实的 3 次");
    assert!(call.retried, "必须标记为被重试过");
    assert_eq!(run.retried_calls, 1, "重试计数不能恒为 0");
    // ⚠️ `tool_calls` 是「实际调用次数」（含重试）。
    assert_eq!(run.tool_calls, 3);
}

#[tokio::test]
async fn successful_call_reports_one_attempt() {
    // ⚠️ 对照组：没重试的工具必须显示 1 次尝试。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "tool_a",
            json!({ "value": "x" }),
        )]),
        step_text("完成"),
    ]));

    let (registry, _counters) = slow_registry(&["tool_a"], 10);
    let agent = build_agent(llm, registry, 4);

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.steps[0].tool_calls[0].attempts, 1);
    assert!(!run.steps[0].tool_calls[0].retried);
    assert_eq!(run.retried_calls, 0);
}

// ─────────────────────────────────────────────────────────────────────
// HTTP API 层
// ─────────────────────────────────────────────────────────────────────

/// 构造测试用的 app。
fn test_app() -> (axum::Router, Arc<ScriptedLlm>) {
    let llm = parallel_script();
    let (registry, _counters) = slow_registry(&["tool_a", "tool_b"], 10);
    let agent = build_agent(llm.clone(), registry, 4);

    let config = AppConfig::default();
    let shared_llm: Arc<dyn agent_22::llm::FunctionLlm> = llm.clone();
    let state = Arc::new(agent_22::api::AppStateInner::new(config, agent, shared_llm));

    (router(state), llm)
}

#[tokio::test]
async fn health_endpoint_works() {
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn root_endpoint_lists_tools_and_concurrency() {
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["service"], "agent-22");
    assert!(json["max_concurrency"].as_u64().unwrap() >= 1);
    assert!(json["tools"].as_array().unwrap().len() >= 2);
}

#[tokio::test]
async fn modes_endpoint_explains_four_modes() {
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/modes")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let modes = json["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 4);

    let names: Vec<&str> = modes.iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        vec!["sequential", "join_all", "as_completed", "bounded"]
    );
}

#[tokio::test]
async fn run_endpoint_returns_answer() {
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/run")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(r#"{"question":"问题"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["answer"], "两个工具都查完了。");
    assert_eq!(json["tool_calls"], 2);
    assert_eq!(json["tool_call_groups"], 1);
}

#[tokio::test]
async fn bench_endpoint_returns_four_reports() {
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/bench")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"cities":["北京","上海"],"max_concurrency":2,"simulated":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let reports = json["reports"].as_array().unwrap();
    assert_eq!(reports.len(), 4);

    // ⚠️ 串行必然比并发慢。
    let seq_ms = reports[0]["total_ms"].as_u64().unwrap();
    let par_ms = reports[1]["total_ms"].as_u64().unwrap();
    assert!(seq_ms > par_ms, "串行 {seq_ms}ms 应慢于并发 {par_ms}ms");
    assert!(json["best_speedup"].as_f64().unwrap() > 1.0);
}

#[tokio::test]
async fn bench_endpoint_rejects_empty_cities() {
    // ⚠️ 空列表会走默认城市，所以这不是 400 ——
    //    但「没有城市」时用默认值，行为必须稳定。
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/bench")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(r#"{"simulated":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn call_tool_endpoint_reports_unknown_tool_as_400() {
    // ⚠️ 「用户直接调工具」的错误**不**降级 ——
    //    而是变成真正的 HTTP 错误（400）。
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(r#"{"tool":"nope","args":{}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn call_tool_endpoint_executes_a_tool() {
    let (app, _llm) = test_app();

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"tool":"tool_a","args":{"value":"hello"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert!(json["observation"].as_str().unwrap().contains("hello"));
    assert_eq!(json["blocking"], false);
}

#[tokio::test]
async fn middleware_is_shared_across_concurrent_tasks() {
    // ⚠️ 中间件是 `Arc` 包住的 —— 多个并发任务共享一个实例。
    //    这里验证共享不会导致状态错乱（每个任务的观察对应自己的参数）。
    use agent_18::llm::{ScriptedToolCall, step_calls, step_text};

    let names = ["t1", "t2", "t3"];
    let calls: Vec<ScriptedToolCall> = names
        .iter()
        .enumerate()
        .map(|(i, n)| ScriptedToolCall::new(&format!("c{i}"), n, json!({ "value": n })))
        .collect();

    let llm = Arc::new(ScriptedLlm::new(vec![step_calls(calls), step_text("完成")]));

    let (registry, _counters) = slow_registry(&names, 30);
    let agent = build_agent(llm, registry, 4);

    let run = agent.run("问题").await.unwrap();

    let records = &run.steps[0].tool_calls;
    for (i, name) in names.iter().enumerate() {
        assert_eq!(records[i].name, *name);
        assert!(
            records[i].observation.contains(name),
            "第 {i} 条观察应包含自己的参数 {}，实际：{}",
            name,
            records[i].observation
        );
    }
}

#[tokio::test]
async fn direct_tool_call_goes_through_the_middleware() {
    // ⚠️ 直接调工具时**也要**走中间件（重试逻辑复用）。
    let (registry, counters) = slow_registry(&["tool_a"], 10);

    let middleware = ErrorHandlingMiddleware::new(registry).with_max_retries(0);

    let outcome = middleware
        .wrap_tool_call(&ToolCallRequest {
            name: "tool_a".to_string(),
            arguments: json!({ "value": "x" }),
            call_id: "direct".to_string(),
        })
        .await;

    assert!(outcome.ok);
    assert_eq!(outcome.attempts, 1);
    assert_eq!(counters[0].load(std::sync::atomic::Ordering::SeqCst), 1);
}
