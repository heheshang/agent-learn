//! Middleware —— 在请求/响应的边界上插一手。
//!
//! Rig 里的 middleware 作用在**传输层**（HTTP），不是 agent 层。
//! `HttpMiddleware` 有三个钩子：
//!
//! | 钩子 | 时机 | 能做什么 |
//! |------|------|----------|
//! | `before_request_headers` | 请求头发出去前 | 改 header（如加追踪 id） |
//! | `before_request_body` | body 序列化后 | 观察或改写 body |
//! | `after_response` | 响应头刚到 | 观察状态码、耗时 |
//!
//! 用 `DynHttpClient::with_middleware` 挂到客户端上，然后 `connect` 给
//! provider 配置。多个 middleware 按**挂载顺序**执行：先跑所有 header 钩子，
//! 再跑所有 body 钩子，最后是响应钩子。
//!
//! 与 `AgentHook` 的区别：`AgentHook` 工作在 agent 的**语义层**
//! （选模型、改请求、拦截工具调用），`HttpMiddleware` 工作在**线缆层**
//! （每个 HTTP 请求）。两者互不替代。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rig::http_client::{
    DynHttpClient, Error, Exchange, HeaderMap, HeaderValue, HttpMiddleware, ReqwestClient,
    StatusCode,
};
use rig::providers::openai::{OpenAI, wire::DEEPSEEK};
use rig::wasm_compat::WasmBoxedFuture;

/// 记录每个请求的状态码的 middleware。
///
/// 这是一个纯观察者：它不改请求也不改响应，只在响应到达时记一笔账。
#[derive(Clone, Default)]
pub struct LatencyLogger {
    requests: Arc<AtomicU64>,
}

impl LatencyLogger {
    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::SeqCst)
    }
}

impl HttpMiddleware for LatencyLogger {
    fn after_response<'a>(
        &'a self,
        exchange: Exchange<'a>,
        status: StatusCode,
        _headers: &'a HeaderMap,
    ) -> WasmBoxedFuture<'a, Result<(), Error>> {
        let count = self.requests.fetch_add(1, Ordering::SeqCst) + 1;
        println!(
            "[middleware] #{} {} {} -> {}",
            count,
            exchange.method,
            exchange.uri.path(),
            status.as_u16()
        );
        Box::pin(async { Ok(()) })
    }
}

/// 给每个请求加一个追踪头的 middleware。
///
/// 演示「请求侧」钩子：header 变更对所有后续 body 钩子可见。
#[derive(Clone)]
pub struct TraceHeader {
    tag: String,
}

impl HttpMiddleware for TraceHeader {
    fn before_request_headers<'a>(
        &'a self,
        _exchange: Exchange<'a>,
        headers: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, Result<(), Error>> {
        if let Ok(value) = HeaderValue::from_str(&self.tag) {
            headers.insert("x-trace-tag", value);
        }
        Box::pin(async { Ok(()) })
    }
}

/// 构造一个挂了 middleware 的 DeepSeek 客户端。
///
/// `OpenAIConfig::from_env_with(&DEEPSEEK)` 读 `DEEPSEEK_API_KEY`，
/// `connect(http)` 把它挂到这个传输上，返回一个能建模型的客户端。
pub fn client_with_middleware() -> anyhow::Result<(OpenAI, LatencyLogger)> {
    let logger = LatencyLogger::default();

    let http = DynHttpClient::new(ReqwestClient::default())
        .with_middleware(logger.clone())
        .with_middleware(TraceHeader {
            tag: "02-rig-demo".to_string(),
        });

    let config = rig::providers::openai::OpenAIConfig::from_env_with(&DEEPSEEK)?;
    Ok((config.connect(http), logger))
}

/// 演示：middleware 观察到的请求数。
pub async fn middleware_demo() -> anyhow::Result<()> {
    let (client, logger) = client_with_middleware()?;
    let model = client.completion(rig::providers::deepseek::DEEPSEEK_V4_FLASH);

    let request = rig::completion::CompletionRequest::new("用一句话说明什么是所有权。");
    let response = model.call(request).await?;

    println!("=== Middleware ===");
    println!("回复: {}", response.text());
    println!("middleware 观察到的请求数: {}", logger.requests());
    Ok(())
}
