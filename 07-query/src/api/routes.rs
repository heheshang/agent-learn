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
//! | `TraceLayer` | 打印每个请求的方法、路径、耗时 |
//!
//! ⚠️ 超时设成 180 秒 —— 比 06-rag 的 120 秒长。
//!
//! 因为**改写 + 生成是两次 LLM 调用**：
//!
//! ```text
//! 问题 ──改写(LLM)──> 查询 ──检索──> 上下文 ──生成(LLM)──> 答案
//! ```
//!
//! Multi-Query 更极端：3 路改写（虽然它们是 1 次 LLM 调用）+ 1 次生成。
//! 两次串行的 LLM 调用，120 秒不够用。
//!
//! ## 统一错误格式
//!
//! `fallback` 让「请求体解析失败」也走同一套错误响应。
//! 不加的话，axum 默认返回纯文本，而其它错误返回 JSON ——
//! 客户端要处理两种格式，很别扭。

use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::api::handlers;
use crate::api::AppState;

/// LLM 请求超时。
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// 请求体大小上限（1 MB）。
pub const MAX_BODY_SIZE: usize = 1024 * 1024;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route("/api/ask", post(handlers::ask))
        .route("/api/search", post(handlers::search))
        .route("/api/transform", post(handlers::transform))
        .route("/api/compare", post(handlers::compare))
        .route("/api/reindex", post(handlers::reindex))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
