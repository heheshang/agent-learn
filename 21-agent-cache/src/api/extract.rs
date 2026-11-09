//! 自定义提取器 —— 把 JSON 请求体解析失败也变成统一错误。
//!
//! 与 06~20 完全相同的做法：解析失败时返回 `AppError::BadRequest`，
//! 于是**所有**错误都是同一种 JSON 格式。

use axum::Json;
use axum::extract::{FromRequest, Request};

use crate::error::AppError;

/// 带错误处理的 JSON 请求体。
pub struct JsonBody<T>(pub T);

impl<T, S> FromRequest<S> for JsonBody<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(JsonBody(value)),
            Err(rejection) => Err(AppError::BadRequest(format!(
                "请求体解析失败: {}",
                rejection.body_text()
            ))),
        }
    }
}
