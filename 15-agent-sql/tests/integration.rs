//! HTTP API 层集成测试。
//!
//! ## 测试方式
//!
//! 用 `tower::ServiceExt::oneshot` **直接调路由** —— 不开端口、零 IO 开销。
//!
//! 为什么不用真的起服务器：
//!
//! 1. 快（每个用例几毫秒）
//! 2. 不会因为端口占用而随机失败
//! 3. 能精确构造请求（状态码、请求头、body）
//!
//! ## 零网络确定性
//!
//! Agent 用 [`ScriptedLlm`] —— 预设模型输出，
//! 所以「护栏拦截 → 模型改写 → 重新执行」这条链路能被**完全确定**地测到。

use std::sync::Arc;

use agent_14::tool::ToolRegistry;
use agent_15::agent::Agent;
use agent_15::api::{AppStateInner, routes};
use agent_15::checker::StaticChecker;
use agent_15::config::AppConfig;
use agent_15::db::{Database, load_seed};
use agent_15::llm::ScriptedLlm;
use agent_15::tools::{ListTablesTool, QueryCheckerTool, QueryTool, SchemaTool};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// 构造一个带种子数据库的测试环境。
fn setup_db() -> (tempfile::TempDir, Arc<Database>) {
    let dir = tempfile::tempdir().unwrap();
    let seed = dir.path().join("seed.sql");
    std::fs::write(
        &seed,
        r#"
        CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
        CREATE TABLE Track (
            TrackId INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            GenreId INTEGER NOT NULL REFERENCES Genre(GenreId),
            Milliseconds INTEGER NOT NULL
        );
        INSERT INTO Genre (GenreId, Name) VALUES (1, '摇滚'), (2, '金属');
        INSERT INTO Track (TrackId, Name, GenreId, Milliseconds) VALUES
            (1, '夜色', 1, 300000),
            (2, '回声', 1, 200000),
            (3, '侧影', 2, 400000);
        "#,
    )
    .unwrap();
    let db_path = dir.path().join("test.db");
    load_seed(&seed, &db_path).unwrap();
    (dir, Arc::new(Database::open_readonly(&db_path).unwrap()))
}

/// 构造应用（用脚本化 LLM，零网络）。
fn setup_app(answers: Vec<String>) -> (tempfile::TempDir, axum::Router) {
    let (dir, db) = setup_db();

    let registry = ToolRegistry::new(vec![
        Arc::new(ListTablesTool::new(db.clone())),
        Arc::new(SchemaTool::new(db.clone())),
        Arc::new(QueryTool::new(db.clone())),
        Arc::new(QueryCheckerTool::new(Arc::new(StaticChecker::new()))),
    ]);

    let llm = Arc::new(ScriptedLlm::new(answers));

    let mut config = AppConfig::default();
    config.agent.max_steps = 6;

    let agent = Arc::new(Agent::new(llm, registry, config.clone()));
    let state = Arc::new(AppStateInner::new(config, agent, db));

    (dir, routes::router(state))
}

/// 发一个 JSON POST 请求。
async fn post_json(
    app: &axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("Content-Type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);

    (status, json)
}

/// 发一个 GET 请求。
async fn get(app: &axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);

    (status, json)
}

// ── 基础端点 ──────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = get(&app, "/health").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn root_lists_tools_and_database_info() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = get(&app, "/").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["database"]["read_only"], true);
    assert_eq!(body["database"]["dialect"], "sqlite");

    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 4);

    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"sql_db_list_tables"));
    assert!(names.contains(&"sql_db_schema"));
    assert!(names.contains(&"sql_db_query"));
    assert!(names.contains(&"sql_db_query_checker"));
}

#[tokio::test]
async fn tools_endpoint_returns_schemas() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = get(&app, "/api/tools").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 4);

    // 每个工具都要有描述和参数说明 —— 那是模型行为的一部分。
    for tool in body["tools"].as_array().unwrap() {
        assert!(!tool["description"].as_str().unwrap().is_empty());
        assert!(tool["schema"].is_object());
    }
}

#[tokio::test]
async fn schema_endpoint_describes_the_database() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = get(&app, "/api/schema").await;

    assert_eq!(status, StatusCode::OK);
    let tables = body["tables"].as_array().unwrap();
    assert_eq!(tables.len(), 2);

    let track = tables
        .iter()
        .find(|t| t["name"] == "Track")
        .expect("应该有 Track 表");
    assert_eq!(track["row_count"], 3);
    assert!(track["ddl"].as_str().unwrap().contains("CREATE TABLE"));
    assert!(track["columns"].as_array().unwrap().len() == 4);
}

// ── 直接调工具 ────────────────────────────────────────────────────

#[tokio::test]
async fn call_tool_executes_a_query() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = post_json(
        &app,
        "/api/tools/call",
        serde_json::json!({
            "tool": "sql_db_query",
            "args": { "query": "SELECT Name FROM Genre LIMIT 5" }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ok"], true);
    assert!(body["observation"].as_str().unwrap().contains("摇滚"));
}

#[tokio::test]
async fn call_tool_returns_guard_error_as_400() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = post_json(
        &app,
        "/api/tools/call",
        serde_json::json!({
            "tool": "sql_db_query",
            "args": { "query": "DELETE FROM Track" }
        }),
    )
    .await;

    // ⚠️ 护栏拦截是**调用方**的问题 → 400，不是 500。
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"].as_str().unwrap().contains("只读"));
}

#[tokio::test]
async fn call_tool_with_unknown_tool_returns_400() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = post_json(
        &app,
        "/api/tools/call",
        serde_json::json!({ "tool": "nope", "args": {} }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"].as_str().unwrap().contains("未知工具"));
}

#[tokio::test]
async fn call_tool_with_empty_name_returns_400() {
    let (_dir, app) = setup_app(vec![]);
    let (status, _) = post_json(
        &app,
        "/api/tools/call",
        serde_json::json!({ "tool": "  ", "args": {} }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ── 护栏端点 ──────────────────────────────────────────────────────

#[tokio::test]
async fn guard_endpoint_allows_read_only_queries() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = post_json(
        &app,
        "/api/guard",
        serde_json::json!({ "query": "SELECT Name FROM Genre LIMIT 5" }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["allowed"], true);
    assert!(body.get("reason").is_none());
}

#[tokio::test]
async fn guard_endpoint_blocks_writes_with_a_reason() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) = post_json(
        &app,
        "/api/guard",
        serde_json::json!({ "query": "DELETE FROM Track" }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["allowed"], false);
    assert!(body["reason"].as_str().unwrap().contains("只读"));
}

#[tokio::test]
async fn guard_endpoint_blocks_multiple_statements() {
    let (_dir, app) = setup_app(vec![]);
    let (_status, body) = post_json(
        &app,
        "/api/guard",
        serde_json::json!({ "query": "SELECT 1; DROP TABLE Track" }),
    )
    .await;

    assert_eq!(body["allowed"], false);
    assert!(body["reason"].as_str().unwrap().contains("单条"));
}

// ── Agent 循环 ────────────────────────────────────────────────────

#[tokio::test]
async fn run_executes_the_full_discovery_chain() {
    let (_dir, app) = setup_app(vec![
        "Thought: 先看表\nAction: sql_db_list_tables\nAction Input: {}".to_string(),
        "Thought: 看 schema\nAction: sql_db_schema\nAction Input: {\"table_names\": \"Genre\"}".to_string(),
        "Thought: 查\nAction: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}".to_string(),
        "Final Answer: 有摇滚和金属两个流派".to_string(),
    ]);

    let (status, body) = post_json(
        &app,
        "/api/run",
        serde_json::json!({ "question": "有哪些流派？" }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["tool_calls"], 3);
    assert!(body["answer"].as_str().unwrap().contains("摇滚"));
    assert_eq!(body["executed_sql"].as_array().unwrap().len(), 1);
    assert!(body["blocked_sql"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn run_records_blocked_sql_for_audit() {
    let (_dir, app) = setup_app(vec![
        "Thought: 删掉\nAction: sql_db_query\nAction Input: {\"query\": \"DELETE FROM Track\"}"
            .to_string(),
        "Final Answer: 数据库是只读的，我不能删".to_string(),
    ]);

    let (status, body) = post_json(
        &app,
        "/api/run",
        serde_json::json!({ "question": "删掉所有曲目" }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let blocked = body["blocked_sql"].as_array().unwrap();
    assert_eq!(blocked.len(), 1);
    assert!(blocked[0].as_str().unwrap().contains("DELETE"));

    // 执行过的 SQL 应该是空的 —— 护栏确实没让它落地。
    assert!(body["executed_sql"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn run_feeds_guard_error_back_to_the_model() {
    let (_dir, app) = setup_app(vec![
        "Thought: 试试\nAction: sql_db_query\nAction Input: {\"query\": \"DROP TABLE Track\"}"
            .to_string(),
        "Final Answer: 被拒绝了".to_string(),
    ]);

    let (status, body) =
        post_json(&app, "/api/run", serde_json::json!({ "question": "删表" })).await;

    assert_eq!(status, StatusCode::OK);
    // 第一步是失败的（护栏拦截）。
    assert_eq!(body["steps"][0]["tool_ok"], false);
    assert!(
        body["steps"][0]["observation"]
            .as_str()
            .unwrap()
            .contains("只读")
    );
}

#[tokio::test]
async fn run_with_empty_question_returns_400() {
    let (_dir, app) = setup_app(vec![]);
    let (status, body) =
        post_json(&app, "/api/run", serde_json::json!({ "question": "   " })).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"].as_str().unwrap().contains("问题"));
}

#[tokio::test]
async fn run_with_malformed_body_returns_400() {
    let (_dir, app) = setup_app(vec![]);

    let request = Request::builder()
        .method("POST")
        .uri("/api/run")
        .header("Content-Type", "application/json")
        .body(Body::from("{ 不是 JSON"))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn run_response_contains_the_full_trajectory() {
    let (_dir, app) = setup_app(vec![
        "Action: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}"
            .to_string(),
        "Final Answer: 好了".to_string(),
    ]);

    let (_status, body) = post_json(
        &app,
        "/api/run",
        serde_json::json!({ "question": "有哪些流派" }),
    )
    .await;

    let trajectory = body["trajectory"].as_str().unwrap();
    assert!(trajectory.contains("Action: sql_db_query"));
    assert!(trajectory.contains("Observation:"));
    assert!(trajectory.contains("Final Answer:"));
}

#[tokio::test]
async fn run_rejects_an_answer_that_was_never_grounded_in_a_query() {
    // ⚠️ 回归测试（实测踩过的坑）：
    // 模型只查了 schema 就直接给答案，数字全是编的。
    //
    // 修法：机制化兜底 —— 没查过库就把答案退回，要求先查。
    let (_dir, app) = setup_app(vec![
        "Thought: 我直接答\nFinal Answer: 编造的答案：平均 393600 毫秒".to_string(),
        "Action: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}"
            .to_string(),
        "Final Answer: 真实答案：摇滚".to_string(),
    ]);

    let (status, body) = post_json(
        &app,
        "/api/run",
        serde_json::json!({ "question": "哪个流派平均最长" }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // 第一步被退回（tool_ok = false，且 observation 里说明了原因）。
    assert_eq!(body["steps"][0]["tool_ok"], false);
    let obs = body["steps"][0]["observation"].as_str().unwrap();
    assert!(
        obs.contains("没有数据来源"),
        "退回原因应说明「数字没有来源」"
    );

    // 最终答案是**查过库之后**给出的。
    let answer = body["answer"].as_str().unwrap();
    assert!(answer.contains("真实答案"));
    assert!(!answer.contains("编造的答案"));

    // 而且那条查询真的执行了。
    assert_eq!(body["executed_sql"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn run_strips_a_fabricated_observation_and_executes_the_real_action() {
    // ⚠️ 回归测试（实测踩过的坑）：
    // 模型一口气把整条轨迹（Observation + Final Answer）都编了出来。
    //
    // 修法：剥离伪造段 → 只执行其中的 Action → 拿真实的观察。
    let (_dir, app) = setup_app(vec![
        "Thought: 查一下\n\
         Action: sql_db_query\n\
         Action Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}\n\
         Observation: 列: Name\n\
         行数: 5\n\
         ---\n\
         编造的行\n\
         Final Answer: 编造的答案"
            .to_string(),
        "Final Answer: 真实答案：摇滚".to_string(),
    ]);

    let (status, body) = post_json(
        &app,
        "/api/run",
        serde_json::json!({ "question": "有哪些流派" }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // ⚠️ 那条 SQL 必须真的执行过。
    let executed = body["executed_sql"].as_array().unwrap();
    assert_eq!(executed.len(), 1);
    assert!(
        executed[0]
            .as_str()
            .unwrap()
            .contains("SELECT Name FROM Genre")
    );

    // 编造的答案不能出现。
    let answer = body["answer"].as_str().unwrap();
    assert!(answer.contains("真实答案"));
    assert!(!answer.contains("编造的答案"));
}
