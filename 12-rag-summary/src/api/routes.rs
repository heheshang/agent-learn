//! 路由定义。
//!
//! ## 分层
//!
//! ```text
//! routes.rs   —— 路由表 + 中间件
//! handlers.rs —— 处理器（薄）
//! models.rs  —— 请求/响应类型（纯数据）
//! extract.rs —— JSON 提取器
//! ```
//!
//! 处理器**只做三件事**：取参数、调核心层、组装响应。
//! 任何业务判断都不该出现在这里 —— 否则同一个逻辑会被
//! CLI 和 HTTP 走两遍，两边慢慢就会走偏。

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;

use crate::api::AppState;
use crate::api::handlers;

/// 构建路由表。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::index))
        .route("/health", get(handlers::health))
        .route("/api/search", post(handlers::search))
        .route("/api/ask", post(handlers::ask))
        .route("/api/compare", post(handlers::compare))
        .route("/api/eval", post(handlers::eval))
        .route("/api/reindex", post(handlers::reindex))
        .layer(CorsLayer::permissive())
        .layer(trace_layer())
        .with_state(state)
}
