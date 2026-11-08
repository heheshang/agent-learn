use axum::{Router, routing::get};

use crate::handlers::{self, AppState};
use agent_logging::trace_layer;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route(
            "/users",
            get(handlers::list_users).post(handlers::create_user),
        )
        .route(
            "/users/{id}",
            get(handlers::get_user).delete(handlers::delete_user),
        )
        // 请求追踪中间件：给每个请求分配 request_id，
        // 记录方法 / 路径 / 状态码 / 耗时。
        .layer(trace_layer())
        .with_state(state)
}
