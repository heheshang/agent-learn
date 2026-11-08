//! 自定义提取器（Extractor）—— 把 JSON 请求体解析失败也变成统一错误。
//!
//! 与 06~12 完全相同的做法：解析失败时返回 `AgentError::BadRequest`，
//! 于是**所有**错误（参数错误、解析错误、后端故障）都是同一种 JSON 格式。
//!
//! ```json
//! { "error": "Bad Request", "message": "请求体解析失败: ..." }
//! ```

use axum::Json;
use axum::extract::{FromRequest, Request};

use crate::error::AgentError;

/// 带错误处理的 JSON 请求体。
///
/// ```ignore
/// async fn handler(body: JsonBody<RunRequest>) -> Result<Json<RunResponse>, AgentError> {
///     let payload = body.0;   // 一定是解析成功的
///     // ...
/// }
/// ```
pub struct JsonBody<T>(pub T);

impl<T, S> FromRequest<S> for JsonBody<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = AgentError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(JsonBody(value)),
            Err(rejection) => Err(AgentError::BadRequest(format!(
                "请求体解析失败: {}",
                rejection.body_text()
            ))),
        }
    }
}
