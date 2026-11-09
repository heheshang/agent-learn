//! 路由表。
//!
//! | 中间件 | 作用 |
//! |---|---|
//! | `TimeoutLayer` | LLM 调用 + 外部 HTTP 都可能挂起 |
//! | `RequestBodyLimitLayer` | 防止超大请求体打爆内存 |
//! | `trace_layer` | 统一请求追踪：请求 ID、方法、路径、耗时、状态码 |

use std::time::Duration;

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{get, post};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::api::AppState;
use crate::api::handlers;

/// 请求超时。
///
/// ⚠️ 180 秒 —— 研究阶段可能要跑多轮工具，
/// 每次工具调用最多 3 次尝试（重试 + 退避），
/// 外加一次成文调用。
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// 请求体大小上限（1 MB）。
pub const MAX_BODY_SIZE: usize = 1024 * 1024;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route("/api/research", post(handlers::research))
        .route("/api/weekly", post(handlers::weekly))
        .route("/api/plan", post(handlers::plan))
        .route("/api/tools/call", post(handlers::call_tool))
        .route("/api/evidence", get(handlers::evidence))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(trace_layer())
        .with_state(state)
}
