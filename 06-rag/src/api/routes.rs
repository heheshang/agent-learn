//! 路由表。
//!
//! 用 `Router` 组合路由，`route()` 绑定路径 + 方法。
//!
//! ## 中间件
//!
//! | 中间件 | 作用 |
//! |---|---|
//! | `TimeoutLayer` | LLM 调用可能挂起，没有超时会拖垮整个服务 |
//! | `RequestBodyLimitLayer` | 防止超大请求体打爆内存 |
//! | `trace_layer` | 打印每个请求的方法、路径、耗时、状态码，并分配 `request_id` |
//!
//! ⚠️ 超时设成 120 秒 —— LLM 调用（尤其推理模型）本来就慢，
//! 但也不能无限等，否则一个挂死的请求会占着连接不放。
//!
//! ## 统一错误格式
//!
//! `fallback` 让「请求体解析失败」也走同一套错误响应。
//! 不加的话，axum 默认返回纯文本，而其它错误返回 JSON ——
//! 客户端要处理两种格式，很别扭。

use std::time::Duration;

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{get, post};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::api::AppState;
use crate::api::handlers;

/// LLM 请求超时。
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// 请求体大小上限（1 MB）。
pub const MAX_BODY_SIZE: usize = 1024 * 1024;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route("/api/ask", post(handlers::ask))
        .route("/api/agent", post(handlers::agent))
        .route("/api/search", post(handlers::search))
        .route("/api/reindex", post(handlers::reindex))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        // 请求追踪：给每个请求分配 request_id，
        // handler 内部的日志会自动带上它。
        .layer(trace_layer())
        .with_state(state)
}
