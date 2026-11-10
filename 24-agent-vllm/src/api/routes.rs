//! 路由装配。

use axum::Router;

use crate::api::handlers::{self, AppState};

/// 装配全部路由。
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", axum::routing::get(handlers::info))
        .route("/health", axum::routing::get(handlers::health))
        .route("/api/modes", axum::routing::get(handlers::modes))
        .route("/api/env", axum::routing::get(handlers::env_info))
        .route("/api/simulate", axum::routing::post(handlers::simulate))
        .route("/api/bench", axum::routing::post(handlers::bench))
        .with_state(state)
}
