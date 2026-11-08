//! 路由表。
//!
//! ## 中间件
//!
//! | 中间件 | 作用 |
//! |---|---|
//! | `TimeoutLayer` | PDF 解析可能很慢（几百页的大文件），没有超时会拖垮服务 |
//! | `RequestBodyLimitLayer` | 防止超大请求体打爆内存 |
//! | `trace_layer` | 统一请求追踪：请求 ID、方法、路径、耗时、状态码 |
//!
//! ## ⚠️ 超时给得比别处长
//!
//! 06-rag 的超时是 120 秒（一次 LLM 往返），
//! 这里给 300 秒 —— 因为 `/api/parse` 可能要处理**几百页**的 PDF，
//! 而且纯几何路线虽然快，但大文件的解析仍然是秒级的。

use std::time::Duration;

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{get, post};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::api::AppState;
use crate::api::handlers;

/// 请求超时。
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// 请求体大小上限（1 MB）。
pub const MAX_BODY_SIZE: usize = 1024 * 1024;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route("/api/parse", post(handlers::parse))
        .route("/api/compare", post(handlers::compare))
        .route("/api/chunk", post(handlers::chunk))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(trace_layer())
        .with_state(state)
}
