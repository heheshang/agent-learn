use axum::{
    routing::get,
    Router,
};

use crate::handlers::{self, AppState};
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/health", get(handlers::health))
        .route(
            "/users",
            get(handlers::list_users).post(handlers::create_user),
        )
        .route("/users/{id}", get(handlers::get_user).delete(handlers::delete_user))
        .with_state(state)
}
