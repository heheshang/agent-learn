//! 集成测试 —— 全链路，**零网络**。
//!
//! ## 为什么能零网络
//!
//! 两个「外部依赖」都被替换掉了：
//!
//! | 依赖 | 真实实现 | 测试替身 |
//! |---|---|---|
//! | LLM | `GenaiFunctionLlm`（DeepSeek API） | `ScriptedLlm`（按脚本返回） |
//! | Web 搜索 | `BingBackend` / `GithubBackend` | `MockBackend`（固定结果） |
//!
//! 所以整个测试**不碰网络、不花 token、可重复**。
//!
//! ## 测什么
//!
//! 1. 完整周报流程（研究 → 成文 → 校验）
//! 2. 完整项目方案流程
//! 3. **接地校验**的三态（接地 / 部分接地 / 未接地）
//! 4. **工具失败 → 降级 → 模型自我纠正**的完整链路
//! 5. **致命错误 → 终止**（不返回模型的答案）
//! 6. HTTP API 层（`tower::ServiceExt::oneshot`，不开端口）

use std::sync::Arc;

use agent_18::llm::{ScriptedLlm, ScriptedToolCall, step_calls, step_failure, step_text};
use agent_18::middleware::ErrorHandlingMiddleware;
use agent_18::tool::ToolRegistry;
use agent_19::agent::{Agent, AgentConfig};
use agent_19::api::AppStateInner;
use agent_19::config::AppConfig;
use agent_19::error::ErrorCode;
use agent_19::evidence::{EvidenceCollector, EvidenceKind};
use agent_19::llm::ToolChoiceMode;
use agent_19::search::{MockBackend, SearchResult, WebBackend};
use agent_19::tools::{MemoryTool, default_registry};

use serde_json::json;

/// 造一个 BM25 索引。
fn index() -> Arc<rag_08::bm25::Bm25Index> {
    let docs = vec![
        rag_08::bm25::Bm25Document {
            id: "1".to_string(),
            source: "notes.md".to_string(),
            heading_path: vec!["错误处理".to_string()],
            text: "错误处理的降级策略是把工具失败转成观察喂回模型，让模型自我纠正".to_string(),
        },
        rag_08::bm25::Bm25Document {
            id: "2".to_string(),
            source: "notes.md".to_string(),
            heading_path: vec!["检索".to_string()],
            text: "BM25 是关键词倒排检索，不理解语义".to_string(),
        },
    ];
    Arc::new(rag_08::bm25::Bm25Index::build(&docs))
}

/// 造一个 mock 后端。
fn mock_backend() -> Arc<dyn WebBackend> {
    Arc::new(MockBackend::new(vec![
        SearchResult {
            title: "Rust 官方文档".to_string(),
            url: "https://doc.rust-lang.org/book/".to_string(),
            snippet: "The Rust Programming Language".to_string(),
        },
        SearchResult {
            title: "Tokio 文档".to_string(),
            url: "https://docs.rs/tokio/latest/tokio/".to_string(),
            snippet: "An asynchronous runtime for Rust".to_string(),
        },
    ]))
}

/// 造记忆工具。
fn memory_tool() -> MemoryTool {
    let store: Arc<dyn agent_17::longterm::MemoryStore> =
        Arc::new(agent_17::longterm::InMemoryMemoryStore::new());
    let retriever: Arc<dyn agent_17::longterm::MemoryRetriever> =
        Arc::new(agent_17::longterm::Bm25MemoryRetriever::new(store.clone()));
    MemoryTool::new(store, retriever, "tester", 5, 20)
}

/// 造注册表。
fn registry() -> ToolRegistry {
    default_registry(index(), mock_backend(), memory_tool(), 5, 8)
}

/// 造中间件（重试次数设为 0，让测试更快）。
fn middleware(max_retries: usize) -> ErrorHandlingMiddleware {
    ErrorHandlingMiddleware::new(registry())
        .with_max_retries(max_retries)
        .with_base_delay(std::time::Duration::from_millis(1))
        .with_max_delay(std::time::Duration::from_millis(2))
}

/// 造 Agent。
///
/// ## ⚠️ 为什么参数是 `ScriptedLlm` 而不是 `Arc<dyn FunctionLlm>`
///
/// 因为测试里 `ScriptedLlm` 还要用来断言调用次数 ——
/// 提成 trait object 就丢掉了具体类型。
fn agent(llm: ScriptedLlm, max_steps: usize) -> Agent {
    Agent::new(
        Arc::new(llm),
        middleware(0),
        AgentConfig {
            max_steps,
            tool_choice: ToolChoiceMode::Auto,
        },
    )
}

// ─────────────────────────────────────────────────────────────────────
// ① 完整周报流程
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn weekly_pipeline_runs_end_to_end() {
    // 第 1 步：模型调两个工具（并行）。
    // 第 2 步：模型给出最终答案。
    // 第 3 步：成文调用 —— 输出带证据引用的 JSON。
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![
            ScriptedToolCall::new("c1", "kb_search", json!({ "query": "错误处理 降级" })),
            ScriptedToolCall::new("c2", "web_search", json!({ "query": "rust 错误处理" })),
        ]),
        step_text("本周完成了错误处理模块的设计。".to_string()),
        step_text(
            r#"```json
{
  "overview": "本周完成错误处理模块与 Web 搜索接入",
  "highlights": [
    { "text": "实现了重试与降级策略", "evidence": ["E1"] }
  ],
  "risks": [
    { "text": "外部搜索服务不稳定", "evidence": ["E3"] }
  ],
  "next_steps": [
    { "text": "接入更多证据来源", "evidence": ["E1"] }
  ]
}
```"#
                .to_string(),
        ),
    ]);

    let pipeline = agent_19::pipeline::WeeklyPipeline::new(
        Arc::new(agent(llm, 5)),
        Arc::new(ScriptedLlm::new(vec![step_text(
            r#"{"overview":"完成错误处理","highlights":[{"text":"做了降级","evidence":["E1"]}],"risks":[],"next_steps":[]}"#
                .to_string(),
        )])),
        8,
    );

    let result = pipeline.run("本周错误处理进展").await.unwrap();

    // 证据被收集到了。
    assert!(result.run.evidence.len() >= 2, "应该至少收集 2 条证据");

    // 周报被解析成结构体。
    assert!(!result.summary.overview.is_empty());
    assert_eq!(result.summary.highlights.len(), 1);

    // Markdown 渲染包含引用编号。
    assert!(result.markdown.contains("[E1]"));
    assert!(result.markdown.contains("接地状态"));
}

#[tokio::test]
async fn weekly_pipeline_collects_evidence_from_tools() {
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "kb_search",
            json!({ "query": "错误处理 降级" }),
        )]),
        step_text("查到了错误处理的设计。".to_string()),
    ]);

    let a = agent(llm, 3);
    let system = agent_19::prompt::research_system();
    let (run, collector) = a.research("错误处理怎么设计的", &system).await.unwrap();

    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.tool_call_groups, 1);
    assert_eq!(run.degraded_errors, 0);
    assert!(!collector.is_empty(), "kb_search 成功后应该有证据");
    assert!(run.tools_used.contains(&"kb_search".to_string()));
}

// ─────────────────────────────────────────────────────────────────────
// ② 完整项目方案流程
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn plan_pipeline_runs_end_to_end() {
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "web_search",
            json!({ "query": "Rust axum RAG" }),
        )]),
        step_text("查到了一些 Rust 异步运行时的资料。".to_string()),
    ]);

    let plan_llm = ScriptedLlm::new(vec![step_text(
        r#"{
  "title": "研究助手",
  "goal": "整合 RAG 与 Web 搜索",
  "tech_stack": ["Rust", "axum"],
  "modules": [
    { "name": "检索层", "purpose": "召回证据", "tech": ["BM25"], "tasks": ["构建索引"], "evidence": ["E1"] }
  ],
  "milestones": [
    { "text": "端到端跑通", "evidence": ["E1"] }
  ],
  "risks": []
}"#
        .to_string(),
    )]);

    let pipeline =
        agent_19::pipeline::PlanPipeline::new(Arc::new(agent(llm, 3)), Arc::new(plan_llm), 8);

    let result = pipeline.run("做一个研究助手").await.unwrap();

    assert_eq!(result.plan.title, "研究助手");
    assert_eq!(result.plan.tech_stack.len(), 2);
    assert_eq!(result.plan.modules.len(), 1);
    assert_eq!(result.plan.modules[0].tasks.len(), 1);
    assert!(result.markdown.contains("### 检索层"));
    assert!(result.markdown.contains("- [ ] 构建索引"));
}

#[tokio::test]
async fn plan_pipeline_rejects_empty_requirement() {
    let llm = ScriptedLlm::new(vec![step_text("不会用到")]);
    let pipeline = agent_19::pipeline::PlanPipeline::new(
        Arc::new(agent(llm, 3)),
        Arc::new(ScriptedLlm::new(vec![step_text("不会用到")])),
        8,
    );

    let err = pipeline.run("   ").await.unwrap_err();

    assert_eq!(err.status_code(), 400);
}

// ─────────────────────────────────────────────────────────────────────
// ③ 接地校验三态
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn grounding_has_three_states() {
    let mut collector = EvidenceCollector::new();
    collector.add(EvidenceKind::Kb, "kb_search", "notes.md", "内容");

    // 接地。
    let grounded = agent_19::report::parse_weekly_summary(
        r#"{"overview":"o","highlights":[{"text":"t","evidence":["E1"]}],"risks":[],"next_steps":[]}"#,
        &collector,
    )
    .unwrap();
    assert_eq!(grounded.grounding, agent_19::report::Grounding::Grounded);
    assert!(grounded.grounding.is_grounded());

    // 部分接地（引用了不存在的编号）。
    let partial = agent_19::report::parse_weekly_summary(
        r#"{"overview":"o","highlights":[{"text":"t","evidence":["E1","E9"]}],"risks":[],"next_steps":[]}"#,
        &collector,
    )
    .unwrap();
    assert_eq!(
        partial.grounding,
        agent_19::report::Grounding::PartiallyGrounded
    );
    assert_eq!(
        partial.grounding_report.dangling_refs,
        vec!["E9".to_string()]
    );

    // 未接地（一条都没引用）。
    let ungrounded = agent_19::report::parse_weekly_summary(
        r#"{"overview":"o","highlights":[{"text":"t","evidence":[]}],"risks":[],"next_steps":[]}"#,
        &collector,
    )
    .unwrap();
    assert_eq!(
        ungrounded.grounding,
        agent_19::report::Grounding::Ungrounded
    );
    assert!(!ungrounded.grounding.is_grounded());
}

#[tokio::test]
async fn dangling_reference_is_the_real_hallucination_signal() {
    // ⚠️ 关键区别：
    // 「没引用」= 模型什么都没查（证据为空，也是诚实的）
    // 「引用了不存在的编号」= 模型**编造**了引用（幻觉）
    let mut collector = EvidenceCollector::new();
    collector.add(EvidenceKind::Kb, "kb_search", "notes.md", "内容");

    let summary = agent_19::report::parse_weekly_summary(
        r#"{"overview":"o","highlights":[{"text":"编的","evidence":["E7"]}],"risks":[],"next_steps":[]}"#,
        &collector,
    )
    .unwrap();

    assert_eq!(
        summary.grounding,
        agent_19::report::Grounding::PartiallyGrounded
    );
    assert_eq!(summary.grounding_report.dangling_refs.len(), 1);
    assert!(summary.grounding_report.used_refs.is_empty());
}

// ─────────────────────────────────────────────────────────────────────
// ④ 工具失败 → 降级 → 自我纠正
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tool_failure_degrades_into_observation_and_model_recovers() {
    // 第 1 步：模型调 `kb_search`，但查询词没有共同字符 → NotFound（Business）。
    // 第 2 步：模型**换个词**再查 → 成功。
    // 第 3 步：模型给出最终答案。
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "kb_search",
            json!({ "query": "xyzzy" }),
        )]),
        step_calls(vec![ScriptedToolCall::new(
            "c2",
            "kb_search",
            json!({ "query": "错误处理 降级" }),
        )]),
        step_text("查到了错误处理的降级策略。".to_string()),
    ]);

    let a = agent(llm, 5);
    let system = agent_19::prompt::research_system();
    let (run, collector) = a.research("错误处理怎么设计的", &system).await.unwrap();

    // 第一次失败被降级（不是终止）。
    assert_eq!(run.tool_call_groups, 2);
    assert_eq!(run.degraded_errors, 1);
    assert!(!run.fatal_terminated);

    // 失败的那一步有错误码，成功的那一步没有。
    let first = &run.steps[0].tool_calls[0];
    assert!(!first.ok);
    assert_eq!(first.error_code.as_deref(), Some("E_NOT_FOUND"));
    assert_eq!(first.error_kind.as_deref(), Some("business"));
    // ⚠️ 失败的调用**没有**证据。
    assert!(first.evidence.is_empty());

    // 第二次成功，收集到了证据。
    assert!(!collector.is_empty());
    assert!(run.answer.contains("降级"));
}

#[tokio::test]
async fn unknown_tool_is_business_error_and_does_not_retry() {
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new("c1", "no_such_tool", json!({}))]),
        step_text("换了个方式回答。".to_string()),
    ]);

    let a = agent(llm, 3);
    let system = agent_19::prompt::research_system();
    let (run, _) = a.research("随便问问", &system).await.unwrap();

    let call = &run.steps[0].tool_calls[0];
    assert!(!call.ok);
    // ⚠️ Business 错误**不重试** —— attempts 必须是 1。
    assert_eq!(call.attempts, 1);
    assert_eq!(call.error_kind.as_deref(), Some("business"));
}

// ─────────────────────────────────────────────────────────────────────
// ⑤ 致命错误 → 终止
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fatal_error_terminates_without_returning_model_answer() {
    // 模型调用失败 —— 按 18 的不变量 3，**不能**降级，必须终止。
    let llm = ScriptedLlm::new(vec![step_failure("鉴权失败")]);

    let a = agent(llm, 3);
    let system = agent_19::prompt::research_system();
    let err = a.research("问题", &system).await.unwrap_err();

    assert_eq!(err.status_code(), 502);
}

// ─────────────────────────────────────────────────────────────────────
// ⑥ 证据收集的确定性
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn evidence_ids_are_deterministic_across_runs() {
    async fn collect() -> Vec<String> {
        let llm = ScriptedLlm::new(vec![
            step_calls(vec![
                ScriptedToolCall::new("c1", "kb_search", json!({ "query": "错误处理" })),
                ScriptedToolCall::new("c2", "web_search", json!({ "query": "rust" })),
            ]),
            step_text("答案".to_string()),
        ]);

        let a = agent(llm, 3);
        let system = agent_19::prompt::research_system();
        let (_, collector) = a.research("问题", &system).await.unwrap();

        collector.all().iter().map(|e| e.id.clone()).collect()
    }

    let first = collect().await;
    let second = collect().await;

    // ⚠️ 同样的输入必须给出同样的证据编号 —— 否则报告里的引用没法复现。
    assert_eq!(first, second);
    assert_eq!(first[0], "E1");
}

#[tokio::test]
async fn data_line_is_stripped_before_model_sees_it() {
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "kb_search",
            json!({ "query": "错误处理" }),
        )]),
        step_text("答案".to_string()),
    ]);

    let a = agent(llm, 3);
    let system = agent_19::prompt::research_system();
    let (run, _) = a.research("问题", &system).await.unwrap();

    let observation = &run.steps[0].tool_calls[0].observation;
    // ⚠️ 模型看到的是**干净**的观察文本。
    assert!(!observation.contains("@@DATA:"));
    assert!(!observation.contains("\"hits\""));
    assert!(observation.contains("错误处理"));
}

// ─────────────────────────────────────────────────────────────────────
// ⑦ HTTP API 层
// ─────────────────────────────────────────────────────────────────────

fn test_state() -> Arc<AppStateInner> {
    let config = AppConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        ..AppConfig::default()
    };

    // 用脚本 LLM 造状态 —— 零网络。
    let llm = ScriptedLlm::new(vec![
        step_calls(vec![ScriptedToolCall::new(
            "c1",
            "kb_search",
            json!({ "query": "错误处理 降级" }),
        )]),
        step_text("查到了降级策略。".to_string()),
    ]);

    let agent = Arc::new(agent(llm, 3));

    Arc::new(AppStateInner::new(
        config,
        agent,
        Arc::new(ScriptedLlm::new(vec![step_text(
            r#"{"overview":"o","highlights":[{"text":"t","evidence":["E1"]}],"risks":[],"next_steps":[]}"#,
        )])),
    ))
}

#[tokio::test]
async fn api_health_returns_ok() {
    use tower::ServiceExt;

    let app = agent_19::api::routes::router(test_state());

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
async fn api_root_lists_tools() {
    use tower::ServiceExt;

    let app = agent_19::api::routes::router(test_state());

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
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["service"], "agent-19");
    assert!(v["tools"].as_array().unwrap().len() >= 5);
}

#[tokio::test]
async fn api_research_returns_evidence() {
    use tower::ServiceExt;

    let app = agent_19::api::routes::router(test_state());

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/research")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"question":"错误处理怎么设计的"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(!v["evidence"].as_array().unwrap().is_empty());
    assert_eq!(v["evidence"][0]["id"], "E1");
}

#[tokio::test]
async fn api_call_tool_reports_business_error_as_400() {
    use tower::ServiceExt;

    let app = agent_19::api::routes::router(test_state());

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"tool":"kb_search","args":{"query":"xyzzy"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // ⚠️ 用户直接调工具 → 参数错 / 查无此项 → **400**，不是 500。
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn api_call_unknown_tool_is_400() {
    use tower::ServiceExt;

    let app = agent_19::api::routes::router(test_state());

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/tools/call")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"tool":"no_such_tool","args":{}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn api_evidence_endpoint_documents_sources() {
    use tower::ServiceExt;

    let app = agent_19::api::routes::router(test_state());

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/evidence")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["sources"].as_array().unwrap().len(), 5);
    assert!(v["principle"].as_str().unwrap().contains("先查本地"));
}

// ─────────────────────────────────────────────────────────────────────
// ⑧ 搜索后端的错误分类
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn offline_backend_search_is_transient() {
    let backend = MockBackend::new(Vec::new()).failing();

    let err = backend.search("query", 5).await.unwrap_err();

    // ⚠️ 网络故障 = Transient —— 重试可能成功。
    assert_eq!(err.code, ErrorCode::UpstreamUnavailable);
    assert!(err.retryable());
    assert!(err.degradable());
}

#[tokio::test]
async fn http_error_is_classified_as_not_found() {
    use agent_19::error::ToolError;

    let err = ToolError::new(ErrorCode::NotFound, "web_fetch", "网页返回 HTTP 404")
        .with_hint("换一个可访问的链接。");

    // ⚠️ 404 = Business —— 重试同一个链接没用。
    assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    assert!(!err.retryable());
}
