//! 路由表。
//!
//! ## 中间件
//!
//! | 中间件 | 作用 |
//! |---|---|
//! | `TimeoutLayer` | 评估要调很多次 LLM，没有超时会拖垮整个服务 |
//! | `RequestBodyLimitLayer` | 防止超大请求体打爆内存 |
//! | `trace_layer` | 统一请求追踪：请求 ID、方法、路径、耗时、状态码 |
//!
//! ## ⚠️ 超时必须给得很长
//!
//! 一次 `/api/eval` 可能是：
//!
//! ```text
//! 10 个样本 × 4 个变体 × (1 次生成 + N 次 judge 调用)
//! ```
//!
//! 每个 judge 调用是一次完整的 LLM 往返（几百毫秒到几秒）。
//! 全部跑完可能要**几分钟**。
//!
//! 所以这里给 600 秒 —— 和 08-rag 的 180 秒不同，
//! 因为这里的负载重得多。

use std::time::Duration;

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{get, post};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::api::AppState;
use crate::api::handlers;

/// 请求超时。
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// 请求体大小上限（1 MB）。
pub const MAX_BODY_SIZE: usize = 1024 * 1024;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route("/api/dataset", get(handlers::dataset))
        .route("/api/eval", post(handlers::eval))
        .route("/api/quick-eval", post(handlers::quick_eval))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(trace_layer())
        .with_state(state)
}
