//! 自定义提取器 —— 把 JSON 请求体解析失败也变成统一错误。
//!
//! 和 08-rag 的做法一样：axum 默认返回纯文本错误，
//! 而业务错误返回 JSON —— 客户端要处理两种格式，很别扭。
//!
//! 这里包一层 `Json<T>`，解析失败就转成 [`crate::error::EvalError::BadRequest`]。

use axum::Json;
use axum::extract::{FromRequest, Request};

use crate::error::EvalError;

/// 带错误处理的 JSON 请求体。
///
/// ```ignore
/// async fn handler(body: JsonBody<EvalRequest>) -> Result<Json<EvalResponse>, EvalError> {
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
    type Rejection = EvalError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(JsonBody(value)),
            Err(rejection) => Err(EvalError::BadRequest(format!(
                "请求体解析失败: {}",
                rejection.body_text()
            ))),
        }
    }
}
