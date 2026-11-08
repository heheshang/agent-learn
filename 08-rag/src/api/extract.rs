//! 自定义提取器（Extractor）—— 把 JSON 请求体解析失败也变成统一错误。
//!
//! ## 为什么需要它
//!
//! axum 的 `Json<T>` 提取器在解析失败时会**直接短路**，返回一个纯文本响应：
//!
//! ```text
//! Failed to deserialize the JSON body into the target type: missing field `question`
//! ```
//!
//! 而业务错误（空问题、非法模式名）走的是我们的 `RagError`，返回的是 JSON。
//!
//! 于是客户端要处理**两种格式**的错误 —— 这在写 SDK 时很烦。
//!
//! ## 做法
//!
//! 定义一个新类型包住 `Json<T>`，并实现 `FromRequest`：
//! 解析成功就交出 `T`，失败就把 `JsonRejection` 转成 `RagError::BadRequest`。
//!
//! 这样**所有**错误都是同一种 JSON 格式。

use axum::Json;
use axum::extract::{FromRequest, Request};

use crate::error::RagError;

/// 带错误处理的 JSON 请求体。
///
/// ```ignore
/// async fn handler(body: JsonBody<AskRequest>) -> Result<Json<AskResponse>, RagError> {
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
    type Rejection = RagError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(JsonBody(value)),
            Err(rejection) => Err(RagError::BadRequest(format!(
                "请求体解析失败: {}",
                rejection.body_text()
            ))),
        }
    }
}
