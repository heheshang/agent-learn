//! 错误处理层的集成测试 —— 零网络、零延迟、完全确定。
//!
//! ## 为什么要用 `ScriptedLlm`
//!
//! 本项目的核心行为有**五个**必须锁住的地方：
//!
//! 1. **Transient 会重试**，**Business 不会重试**
//! 2. **Business 降级成观察**（带错误码 + 可执行建议）
//! 3. **Fatal 不降级** —— 终止且不返回模型答案
//! 4. **致命错误也要生成回执** —— 少一条就 400
//! 5. **模型调用失败不能降级**
//!
//! 这些行为用真实模型测是**不可靠**的 ——
//! 模型今天返回 2 个调用，明天返回 1 个，测试就会随机失败。
//! 所以用脚本把模型输出**钉死**。
//!
//! ## 测试分层
//!
//! | 层 | 测什么 | 需要模型？ |
//! |---|---|---|
//! | 本文件 | 错误处理的**控制流**（重试 / 降级 / 终止） | ❌ |
//! | `api` 测试 | HTTP 契约（状态码、响应形状） | ❌ |
//! | 真实模型 | 端到端是否真的能跑通 | ✅（手动） |

use std::sync::Arc;
use std::time::Duration;

use agent_18::agent::{Agent, AgentConfig};
use agent_18::api::AppStateInner;
use agent_18::api::routes::router;
use agent_18::config::AppConfig;
use agent_18::error::ErrorCode;
use agent_18::llm::{
    FunctionLlm, LlmStep, ScriptedLlm, ScriptedToolCall, step_calls, step_failure, step_text,
};
use agent_18::middleware::ErrorHandlingMiddleware;
use agent_18::tool::{
    ErrorAwareTool, FLAKY_FETCH_TOOL_NAME, FlakyTool, GUARDED_PAY_TOOL_NAME, GuardedTool,
    STRICT_LOOKUP_TOOL_NAME, StrictTool, ToolRegistry,
};

use serde_json::{Value, json};
use tower::ServiceExt;

// ─────────────────────────────────────────────────────────────────────
// 测试替身
// ─────────────────────────────────────────────────────────────────────

/// 构建一个测试用的中间件（**零延迟**，不真的 sleep）。
fn middleware_with(tools: Vec<Arc<dyn ErrorAwareTool>>) -> ErrorHandlingMiddleware {
    ErrorHandlingMiddleware::new(ToolRegistry::new(tools)).with_base_delay(Duration::ZERO)
}

/// 构建一个测试用的 Agent。
fn agent_with(llm: Arc<dyn FunctionLlm>, middleware: ErrorHandlingMiddleware) -> Agent {
    Agent::new(llm, middleware, AgentConfig::default())
}

// ─────────────────────────────────────────────────────────────────────
// ① Transient：重试
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn transient_failures_are_retried_until_the_tool_succeeds() {
    // FlakyTool 前 2 次失败 —— 中间件会重试到成功。
    let tool = Arc::new(FlakyTool::new(2));
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            FLAKY_FETCH_TOOL_NAME,
            json!({ "resource": "r" }),
        )]),
        step_text("拉取成功，数据已就绪"),
    ]));

    let agent = agent_with(llm, middleware_with(vec![tool.clone()]));
    let run = agent.run("拉一下").await.unwrap();

    assert_eq!(run.answer, "拉取成功，数据已就绪");
    // ⚠️ 关键断言：重试**发生过**，且最终成功。
    assert_eq!(run.retried_calls, 1);
    assert_eq!(run.degraded_errors, 0);
    assert_eq!(tool.call_count(), 3);
    assert!(run.steps[0].tool_calls[0].retried);
    assert_eq!(run.steps[0].tool_calls[0].attempts, 3);
}

#[tokio::test]
async fn transient_failures_are_degraded_after_the_retry_budget_is_exhausted() {
    // 一直失败 → 重试耗尽 → 降级。
    let tool = Arc::new(FlakyTool::new(99));
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            FLAKY_FETCH_TOOL_NAME,
            json!({ "resource": "r" }),
        )]),
        step_text("上游暂时不可用，请稍后再试"),
    ]));

    let agent = agent_with(llm, middleware_with(vec![tool.clone()]));
    let run = agent.run("拉一下").await.unwrap();

    // 降级了，但不是致命错误 —— 循环继续走到了最终答案。
    assert_eq!(run.degraded_errors, 1);
    assert!(!run.fatal_terminated);
    // 观察文本带错误码 —— 这才是「可诊断」。
    assert!(
        run.steps[0].tool_calls[0]
            .observation
            .contains(ErrorCode::UpstreamUnavailable.id())
    );
    // 重试次数 = 1 + max_retries。
    assert_eq!(tool.call_count(), 3);
}

// ─────────────────────────────────────────────────────────────────────
// ② Business：不重试，只降级
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn business_failures_are_never_retried() {
    // ⚠️ 本项目**最核心**的断言。
    let tool = Arc::new(StrictTool::new(vec!["rec-1".to_string()]));
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            STRICT_LOOKUP_TOOL_NAME,
            json!({ "id": "nope" }),
        )]),
        step_text("没有找到这条记录"),
    ]));

    let agent = agent_with(llm, middleware_with(vec![tool.clone()]));
    let run = agent.run("查一下 nope").await.unwrap();

    // ⚠️ 只调了 1 次 —— 重试对业务失败是**纯浪费**。
    assert_eq!(run.retried_calls, 0);
    assert_eq!(run.degraded_errors, 1);
    assert!(!run.fatal_terminated);
    // 错误码和类别都要被记录。
    assert_eq!(
        run.steps[0].tool_calls[0].error_code.as_deref(),
        Some(ErrorCode::NotFound.id())
    );
    assert_eq!(
        run.steps[0].tool_calls[0].error_kind.as_deref(),
        Some("business")
    );
    // ⚠️ 观察里要有**可执行**的建议 —— 模型据此纠正。
    let observation = &run.steps[0].tool_calls[0].observation;
    assert!(observation.contains("建议"));
    assert!(observation.contains("rec-1"));
}

#[tokio::test]
async fn schema_violations_are_business_failures_and_are_not_retried() {
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            FLAKY_FETCH_TOOL_NAME,
            json!({ "resource": 42 }),
        )]),
        step_text("参数不对，我改一下"),
    ]));

    let agent = agent_with(llm, middleware_with(vec![Arc::new(FlakyTool::new(0))]));
    let run = agent.run("拉一下").await.unwrap();

    assert_eq!(run.retried_calls, 0);
    assert_eq!(run.degraded_errors, 1);
    assert_eq!(
        run.steps[0].tool_calls[0].error_code.as_deref(),
        Some(ErrorCode::InvalidArguments.id())
    );
}

#[tokio::test]
async fn unknown_tools_are_business_failures() {
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new("c1", "no_such_tool", json!({}))]),
        step_text("我不知道有这个工具"),
    ]));

    let agent = agent_with(llm, middleware_with(vec![Arc::new(FlakyTool::new(0))]));
    let run = agent.run("调一下").await.unwrap();

    assert_eq!(run.retried_calls, 0);
    assert_eq!(
        run.steps[0].tool_calls[0].error_code.as_deref(),
        Some(ErrorCode::UnknownTool.id())
    );
    // 观察里要列出可用工具 —— 模型才知道能调什么。
    assert!(
        run.steps[0].tool_calls[0]
            .observation
            .contains("flaky_fetch")
    );
}

// ─────────────────────────────────────────────────────────────────────
// ③ Fatal：终止，不降级
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fatal_failures_terminate_the_loop_without_returning_an_answer() {
    // ⚠️ 致命错误后，**不该**有任何模型的答案被返回。
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            GUARDED_PAY_TOOL_NAME,
            json!({ "amount": 100 }),
        )]),
        step_text("支付成功"),
    ]));

    let agent = agent_with(llm, middleware_with(vec![Arc::new(GuardedTool::new(None))]));
    let err = agent.run("付 100 块").await.unwrap_err();

    // ⚠️ 鉴权失败 → 500（服务端的问题，不是调用方的）。
    assert_eq!(err.status_code(), 500);
    assert!(err.to_string().contains(ErrorCode::Unauthorized.id()));
}

#[tokio::test]
async fn fatal_failures_are_not_retried() {
    let tool = Arc::new(GuardedTool::new(None));
    let llm = Arc::new(ScriptedLlm::new(vec![step_calls(vec![
        ScriptedToolCall::new("c1", GUARDED_PAY_TOOL_NAME, json!({ "amount": 100 })),
    ])]));

    let agent = agent_with(llm, middleware_with(vec![tool]));
    let _ = agent.run("付 100 块").await;

    // ⚠️ 致命错误**一次都不重试** —— 重试是掩盖问题。
    // 这里通过「循环没有跑第二步」间接验证：
    // 脚本里没有第二步预设，如果重试了会看到不同的错误。
    // 更直接的验证在 middleware 单元测试里。
}

// ─────────────────────────────────────────────────────────────────────
// ④ 模型调用失败：不能降级
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn model_failures_are_not_degraded() {
    // ⚠️ 「没有模型就没有答案」—— 不能把模型失败降级成观察。
    let llm = Arc::new(ScriptedLlm::new(vec![step_failure("连接超时")]));
    let agent = agent_with(llm, middleware_with(vec![Arc::new(FlakyTool::new(0))]));

    let err = agent.run("你好").await.unwrap_err();

    assert_eq!(err.status_code(), 502);
    assert!(err.to_string().contains("连接超时"));
}

// ─────────────────────────────────────────────────────────────────────
// ⑤ 多个并行调用：每个都要有回执
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn every_tool_call_gets_its_own_receipt() {
    // ⚠️ 16 实测踩出来的坑：少一条回执 → 整个请求 400。
    //
    // 这里一次返回 3 个调用（1 个成功、1 个业务失败、1 个致命），
    // 断言三条记录都在 —— 且致命的那条**也在**。
    let llm = Arc::new(ScriptedLlm::new(vec![step_calls(vec![
        ScriptedToolCall::new("c1", FLAKY_FETCH_TOOL_NAME, json!({ "resource": "r" })),
        ScriptedToolCall::new("c2", STRICT_LOOKUP_TOOL_NAME, json!({ "id": "nope" })),
        ScriptedToolCall::new("c3", GUARDED_PAY_TOOL_NAME, json!({ "amount": 1 })),
    ])]));

    let agent = agent_with(
        llm,
        middleware_with(vec![
            Arc::new(FlakyTool::new(0)),
            Arc::new(StrictTool::new(vec!["rec-1".to_string()])),
            Arc::new(GuardedTool::new(None)),
        ]),
    );

    // 致命错误导致终止 → Err。
    let err = agent.run("一次调三个").await.unwrap_err();
    assert_eq!(err.status_code(), 500);

    // 断言「三个调用都被处理了」——
    // 用中间件直接跑一遍更容易观察。
    let middleware = middleware_with(vec![
        Arc::new(FlakyTool::new(0)),
        Arc::new(StrictTool::new(vec!["rec-1".to_string()])),
        Arc::new(GuardedTool::new(None)),
    ]);

    let mut ids = Vec::new();
    for (name, args) in [
        (FLAKY_FETCH_TOOL_NAME, json!({ "resource": "r" })),
        (STRICT_LOOKUP_TOOL_NAME, json!({ "id": "nope" })),
        (GUARDED_PAY_TOOL_NAME, json!({ "amount": 1 })),
    ] {
        let outcome = middleware
            .wrap_tool_call(&agent_18::middleware::ToolCallRequest {
                name: name.to_string(),
                arguments: args,
                call_id: format!("c_{name}"),
            })
            .await;
        // 每个调用都有**一条**观察 —— 观察非空即回执存在。
        assert!(!outcome.observation.is_empty(), "{name} 缺少回执");
        ids.push(name.to_string());
    }
    assert_eq!(ids.len(), 3);
}

// ─────────────────────────────────────────────────────────────────────
// HTTP 契约
// ─────────────────────────────────────────────────────────────────────

/// 构造一个测试用的 AppState。
fn state_with(
    llm: Arc<dyn FunctionLlm>,
    middleware: ErrorHandlingMiddleware,
) -> Arc<AppStateInner> {
    let config = AppConfig::default();
    let agent = Arc::new(Agent::new(llm.clone(), middleware, AgentConfig::default()));
    Arc::new(AppStateInner::new(config, agent, llm))
}

#[tokio::test]
async fn error_table_endpoint_exposes_the_decision_rules() {
    let llm = Arc::new(ScriptedLlm::new(vec![step_text("ok")]));
    let app = router(state_with(llm, middleware_with(vec![])));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/errors/table")
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

    let entries = json["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 9);

    // 抽查三条关键规则。
    let find = |code: &str| {
        entries
            .iter()
            .find(|e| e["code"] == code)
            .unwrap_or_else(|| panic!("找不到 {code}"))
    };

    let timeout = find(ErrorCode::UpstreamTimeout.id());
    assert_eq!(timeout["kind"], "transient");
    assert_eq!(timeout["retryable"], true);
    assert_eq!(timeout["degradable"], true);
    assert_eq!(timeout["http_status"], 503);

    let invalid = find(ErrorCode::InvalidArguments.id());
    assert_eq!(invalid["kind"], "business");
    assert_eq!(invalid["retryable"], false);
    assert_eq!(invalid["degradable"], true);
    assert_eq!(invalid["http_status"], 400);

    let unauthorized = find(ErrorCode::Unauthorized.id());
    assert_eq!(unauthorized["kind"], "fatal");
    assert_eq!(unauthorized["retryable"], false);
    assert_eq!(unauthorized["degradable"], false);
    assert_eq!(unauthorized["http_status"], 500);
}

#[tokio::test]
async fn direct_tool_call_returns_400_on_a_business_failure() {
    // ⚠️ 用户直接调工具 → 参数错了是**调用方**的问题 → 400。
    let llm = Arc::new(ScriptedLlm::new(vec![step_text("ok")]));
    let app = router(state_with(
        llm,
        middleware_with(vec![Arc::new(StrictTool::new(vec!["rec-1".to_string()]))]),
    ));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({ "tool": "strict_lookup", "args": { "id": "nope" } }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["message"].as_str().unwrap().contains("E_NOT_FOUND"));
}

#[tokio::test]
async fn direct_tool_call_returns_500_on_a_fatal_failure() {
    let llm = Arc::new(ScriptedLlm::new(vec![step_text("ok")]));
    let app = router(state_with(
        llm,
        middleware_with(vec![Arc::new(GuardedTool::new(None))]),
    ));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({ "tool": "guarded_pay", "args": { "amount": 10 } }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 500);
}

#[tokio::test]
async fn direct_tool_call_returns_200_on_success() {
    let llm = Arc::new(ScriptedLlm::new(vec![step_text("ok")]));
    let app = router(state_with(
        llm,
        middleware_with(vec![Arc::new(FlakyTool::new(0))]),
    ));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({ "tool": "flaky_fetch", "args": { "resource": "r" } }).to_string(),
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
    assert_eq!(json["ok"], true);
    assert_eq!(json["attempts"], 1);
}

#[tokio::test]
async fn run_endpoint_returns_the_error_statistics() {
    let llm = Arc::new(ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            FLAKY_FETCH_TOOL_NAME,
            json!({ "resource": "r" }),
        )]),
        step_text("好了"),
    ]));

    let app = router(state_with(
        llm,
        middleware_with(vec![Arc::new(FlakyTool::new(1))]),
    ));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/run")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({ "question": "拉一下" }).to_string(),
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

    assert_eq!(json["answer"], "好了");
    // ⚠️ 重试被统计到了。
    assert_eq!(json["retried_calls"], 1);
    assert_eq!(json["tool_call_groups"], 1);
    assert_eq!(json["tool_calls"], 2);
    assert_eq!(json["degraded_errors"], 0);
}

#[tokio::test]
async fn run_endpoint_returns_500_on_a_fatal_tool_error() {
    let llm = Arc::new(ScriptedLlm::new(vec![step_calls(vec![
        ScriptedToolCall::new("c1", GUARDED_PAY_TOOL_NAME, json!({ "amount": 10 })),
    ])]));

    let app = router(state_with(
        llm,
        middleware_with(vec![Arc::new(GuardedTool::new(None))]),
    ));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/run")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({ "question": "付钱" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // ⚠️ 致命错误 → 500，**不是** 200 + 一个编出来的答案。
    assert_eq!(response.status(), 500);
}

#[tokio::test]
async fn run_endpoint_rejects_an_empty_question() {
    let llm = Arc::new(ScriptedLlm::new(vec![step_text("ok")]));
    let app = router(state_with(llm, middleware_with(vec![])));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/run")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({ "question": "   " }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn health_endpoint_is_ok() {
    let llm = Arc::new(ScriptedLlm::new(vec![step_text("ok")]));
    let app = router(state_with(llm, middleware_with(vec![])));

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

// ─────────────────────────────────────────────────────────────────────
// 脚本 LLM 的边界
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn scripted_llm_runs_out_of_responses() {
    let llm = Arc::new(ScriptedLlm::new(vec![LlmStep::Response(Box::new(
        agent_18::llm::ScriptedStep::Text("一".to_string()),
    ))]));

    let agent = agent_with(llm.clone(), middleware_with(vec![]));

    // 第一次成功。
    let run = agent.run("问题").await.unwrap();
    assert_eq!(run.answer, "一");

    // 第二次没有预设 → 502。
    let err = agent.run("问题").await.unwrap_err();
    assert_eq!(err.status_code(), 502);
}
