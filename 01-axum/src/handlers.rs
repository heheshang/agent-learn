use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 统一错误响应
#[derive(Debug)]
pub struct AppError {
    code: u16,
    message: String,
}

impl AppError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            code: 400,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: 404,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (
            status,
            Json(serde_json::json!({
                "error": status.canonical_reason(),
                "message": self.message,
            })),
        )
            .into_response()
    }
}

/// 创建用户请求体
#[derive(Debug, Deserialize)]
pub struct CreateUser {
    pub name: String,
    pub age: u32,
}

/// 用户响应体
#[derive(Debug, Clone, Serialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub age: u32,
}

/// 查询参数
#[derive(Debug, Deserialize)]
pub struct Pagination {
    #[serde(default = "default_page")]
    pub page: u32,
    #[serde(default = "default_size")]
    pub size: u32,
}

fn default_page() -> u32 {
    1
}

fn default_size() -> u32 {
    10
}

/// 内部存储的用户数据
#[derive(Debug, Default)]
pub struct UserStore {
    pub users: HashMap<u64, User>,
    pub next_id: u64,
}

impl UserStore {
    fn seeded() -> Self {
        let mut users = HashMap::new();
        users.insert(
            1,
            User {
                id: 1,
                name: "Alice".to_string(),
                age: 30,
            },
        );
        users.insert(
            2,
            User {
                id: 2,
                name: "Bob".to_string(),
                age: 25,
            },
        );
        Self { users, next_id: 3 }
    }
}

/// 应用共享状态（多请求间共享同一份数据）
pub type AppState = Arc<Mutex<UserStore>>;

pub fn app_state() -> AppState {
    Arc::new(Mutex::new(UserStore::seeded()))
}

/// GET / —— 根路由
pub async fn root() -> &'static str {
    "Hello, axum!"
}

/// GET /health —— 健康检查
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET /users —— 查询用户列表，支持分页查询参数 ?page=1&size=10
pub async fn list_users(
    State(state): State<AppState>,
    Query(pagination): Query<Pagination>,
) -> Json<Vec<User>> {
    let store = state.lock().expect("获取用户存储锁失败");
    let mut users: Vec<User> = store.users.values().cloned().collect();
    users.sort_by_key(|u| u.id);

    let start = ((pagination.page.saturating_sub(1)) * pagination.size) as usize;
    Json(users.into_iter().skip(start).take(pagination.size as usize).collect())
}

/// GET /users/{id} —— 按 id 查询单个用户
pub async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<User>, AppError> {
    let store = state.lock().expect("获取用户存储锁失败");
    store
        .users
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| AppError::not_found(format!("用户 {id} 不存在")))
}

/// POST /users —— 创建用户
pub async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUser>,
) -> Result<(StatusCode, Json<User>), AppError> {
    if payload.name.trim().is_empty() {
        return Err(AppError::bad_request("name 不能为空"));
    }

    let mut store = state.lock().expect("获取用户存储锁失败");
    let id = store.next_id;
    store.next_id += 1;
    let user = User {
        id,
        name: payload.name,
        age: payload.age,
    };
    store.users.insert(id, user.clone());

    Ok((StatusCode::CREATED, Json(user)))
}

/// DELETE /users/{id} —— 删除用户
pub async fn delete_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<StatusCode, AppError> {
    let mut store = state.lock().expect("获取用户存储锁失败");
    store
        .users
        .remove(&id)
        .map(|_| StatusCode::NO_CONTENT)
        .ok_or_else(|| AppError::not_found(format!("用户 {id} 不存在")))
}
