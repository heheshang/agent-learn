//! 路由表。
//!
//! ## 中间件
//!
//! | 中间件 | 作用 |
//! |---|---|
//! | `TimeoutLayer` | LLM 调用 + 外部 API 都可能挂起 |
//! | `RequestBodyLimitLayer` | 防止超大请求体打爆内存 |
//! | `trace_layer` | 统一请求追踪 |

use std::time::Duration;

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{delete, get, post};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::api::AppState;
use crate::api::handlers;

/// 请求超时。
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// 请求体大小上限（1 MB）。
pub const MAX_BODY_SIZE: usize = 1024 * 1024;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route("/api/tools", get(handlers::list_tools))
        .route("/api/tools/call", post(handlers::call_tool))
        .route("/api/run", post(handlers::run))
        .route("/api/cache/stats", get(handlers::cache_stats))
        .route("/api/cache/entries", get(handlers::cache_entries))
        .route("/api/cache", delete(handlers::clear_cache))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(trace_layer())
        .with_state(state)
}
