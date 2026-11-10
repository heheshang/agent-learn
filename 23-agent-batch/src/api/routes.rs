//! 路由定义。

use agent_logging::trace_layer;
use axum::Router;
use axum::routing::{get, post};

use crate::api::AppState;
use crate::api::handlers;

/// 构造路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/", get(handlers::root))
        .route("/api/embed", post(handlers::embed))
        .route("/api/rerank", post(handlers::rerank))
        .route("/api/bench/embed", post(handlers::bench_embed))
        .route("/api/bench/rerank", post(handlers::bench_rerank))
        .route("/api/bench/remote", post(handlers::bench_remote))
        .route("/api/bench/sweep", post(handlers::bench_sweep))
        .route("/api/index", post(handlers::index))
        .route("/api/search", post(handlers::search))
        // ⚠️ 用 workspace 的 trace_layer（带 request_id / elapsed_ms），
        //    不要用 tower 的通用 TraceLayer。
        .layer(trace_layer())
        .with_state(state)
}
