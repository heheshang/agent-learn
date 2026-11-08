//! # agent-logging —— workspace 共享日志设施
//!
//! 本仓库有 14 个学习项目，每一个都是独立的可执行服务。
//! 如果每个项目各自实现一套日志初始化，日志格式、级别语义、
//! 请求追踪方式就会各写各的，跨项目对照实验时也没法把日志放一起看。
//!
//! 所以日志设施抽成**独立的 workspace 成员 crate**，所有项目共用。
//!
//! ## 三件事
//!
//! | 能力 | 说明 |
//! |---|---|
//! | [`init`] | 初始化全局 subscriber，一次性设置格式、级别、来源 |
//! | [`trace_layer`] | HTTP 请求追踪中间件：请求 ID、方法、路径、耗时、状态码 |
//! | 宏再导出 | [`info!`] / [`debug!`] / [`warn!`] / [`error!`] 直接从本 crate 导入 |
//!
//! ## 级别约定
//!
//! | 级别 | 用途 | 典型场景 |
//! |---|---|---|
//! | `ERROR` | 请求失败、服务不可用、数据不一致 | LLM 调用失败、索引加载失败 |
//! | `WARN` | 能兜住但不该发生 | 参数非法、工具执行失败、降级 |
//! | `INFO` | 关键业务节点 | 服务启动、索引就绪、一次请求的最终结果 |
//! | `DEBUG` | 过程细节 | 召回了哪些块、Prompt 全文、重排分数 |
//! | `TRACE` | 逐行细节 | 每条候选块的原始分值 |
//!
//! ## 过滤
//!
//! 用环境变量 `RUST_LOG` 控制，遵循 `tracing-subscriber` 的 `EnvFilter` 语法：
//!
//! ```bash
//! RUST_LOG=info                          # 全局 info
//! RUST_LOG=debug,rag_06=trace            # rag_06 单独开到 trace
//! RUST_LOG=warn,rag_12::retriever=debug  # 只对某个模块开 debug
//! ```
//!
//! 不设 `RUST_LOG` 时用各项目传入的默认级别（一般是 `info`）。
//!
//! ## 为什么不用 `println!`
//!
//! 1. **可过滤**：`println!` 没法按模块级别关掉，debug 输出一旦写进去
//!    就永远在生产路径上（每次 LLM 调用都刷一遍 Prompt，是很大的噪音）。
//! 2. **有结构**：`info!(chunk_count = 12, ...)` 出来的字段可以被
//!    日志采集器解析，而 `println!("{}", ...)` 只有一坨文本。
//! 3. **带 span 上下文**：HTTP 请求里的日志自动带上 `request_id`，
//!    一次请求触发的所有日志能串起来。
//!
//! ## 用法
//!
//! ```rust,no_run
//! use agent_logging::{info, init, trace_layer};
//!
//! #[tokio::main]
//! async fn main() {
//!     init("rag-06");
//!     let app: axum::Router = axum::Router::new()
//!         .route("/", axum::routing::get(|| async { "ok" }))
//!         .layer(trace_layer());
//!     info!(addr = "127.0.0.1:3000", "服务已启动");
//!     // ...
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tracing::Instrument;
use tracing::Level;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::time::ChronoLocal;

/// 默认日志级别（未设置 `RUST_LOG` 时使用）。
pub const DEFAULT_LEVEL: Level = Level::INFO;

/// 请求 ID 的全局计数器。
///
/// 请求 ID 需要在**多个线程**之间唯一（HTTP 服务是多线程的），
/// 所以用原子计数器而不是 `rand`。
static REQUEST_SEQ: AtomicU64 = AtomicU64::new(1);

/// 生成一个进程内唯一的请求 ID。
///
/// 形如 `req-7`：前缀 + 进程内自增序号。
///
/// 之所以不引入 uuid 依赖：日志里的请求 ID 只需要
/// **在一次进程运行内唯一**，不需要跨进程/跨机器唯一。
/// 跨机器唯一是分布式追踪（OpenTelemetry）该解决的问题，
/// 那是本仓库不打算引入的复杂度。
pub fn next_request_id() -> String {
    let n = REQUEST_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("req-{n}")
}

/// 初始化全局日志 subscriber。
///
/// 只能在进程里调用一次（第二次调用会被忽略），
/// 因为 `set_global_default` 失败是设计如此 —— 测试里可能会重复调用。
///
/// # 参数
///
/// - `service`：**crate 名**（连字符形式，如 `rag-06`）。
///   它会出现在每一行日志的 `service=` 字段里，便于多项目日志
///   混在一起时区分来源。
pub fn init(service: &str) {
    init_with_level(service, DEFAULT_LEVEL);
}

/// 初始化全局日志 subscriber，指定默认级别。
pub fn init_with_level(service: &str, default_level: Level) {
    // RUST_LOG 优先；没设就用项目传入的默认级别。
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "{default_level},{}={default_level}",
            module_target(service)
        ))
    });

    let result = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_level(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .with_timer(ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".to_string()))
        .try_init();

    if let Err(e) = result {
        // 重复初始化不算错误（测试里常会重复调），
        // 但要让人知道「日志配置没生效」。
        eprintln!("[logging] 全局 subscriber 初始化失败（可能已初始化过）: {e}");
        return;
    }

    tracing::info!(
        service,
        level = %default_level,
        "日志系统已就绪（用 RUST_LOG 环境变量可调整过滤规则）"
    );
}

/// 把服务名转成 tracing 的模块 target 前缀。
///
/// `rag-06` → `rag_06`：Rust 的 crate 名在源码里是下划线形式
/// （`rag_06`），目录名和包名则是连字符形式。统一转成下划线，
/// target 才对得上，`RUST_LOG=rag_06=debug` 才能精确命中。
fn module_target(service: &str) -> String {
    service.replace('-', "_")
}

/// 请求追踪中间件的返回类型别名。
///
/// `FromFnLayer` 有三个泛型参数（回调、状态、抽取器元组），
/// 直接写出来会触发 clippy 的 `type_complexity` 告警，
/// 所以抽成别名。
pub type TraceLayer = axum::middleware::FromFnLayer<
    fn(Request, Next) -> Pin<Box<dyn Future<Output = Response> + Send>>,
    (),
    (Request,),
>;

/// 构造 HTTP 请求追踪中间件。
///
/// 每次请求会开一个 `http_request` span，字段包含：
///
/// | 字段 | 含义 |
/// |---|---|
/// | `request_id` | 进程内唯一请求 ID |
/// | `method` | HTTP 方法 |
/// | `path` | 请求路径 |
/// | `status` | 响应状态码（响应后写入） |
/// | `elapsed_ms` | 耗时毫秒（响应后写入） |
///
/// span 内的日志（handler 里的 `info!` 等）会自动带上这些字段 ——
/// 这正是「一次请求触发的所有日志能串起来」的机制。
///
/// 状态码分级：
/// - `>= 500` → `ERROR`
/// - `>= 400` → `WARN`
/// - 其余 → `INFO`
///
/// 用法：`router.layer(trace_layer())`。
///
/// 返回类型里的 `(Request,)` 是 `axum::middleware::FromFnLayer` 的
/// 「抽取器元组」参数 —— 中间件从请求里抽取 `Request`（整个请求），
/// 剩下的由 `Next` 处理。这个元组必须写对，否则中间件装不上。
pub fn trace_layer() -> TraceLayer {
    axum::middleware::from_fn(trace_handler)
}

/// 请求追踪中间件的实际实现。见 [`trace_layer`]。
///
/// 单独拆成 `fn`（而不是闭包）是为了让返回类型是具体的
/// `FromFnLayer<...>` 而不是一堆泛型参数，调用方写起来简单。
///
/// 之所以手工装箱 future：`axum::middleware::from_fn` 要求回调是
/// `fn(Request, Next) -> impl Future`，而 `impl Future` 在函数返回类型
/// 位置不能写成具体类型（除非用 `impl Trait` 关联类型，那会引入
/// 更复杂的泛型传播）。装箱是最省事的写法。
fn trace_handler(request: Request, next: Next) -> Pin<Box<dyn Future<Output = Response> + Send>> {
    Box::pin(async move {
        let request_id = next_request_id();
        let method = request.method().clone();
        let path = request.uri().path().to_string();
        let query = request.uri().query().unwrap_or("").to_string();

        let span = tracing::info_span!(
            "http_request",
            request_id = %request_id,
            method = %method,
            path = %path,
        );

        // 用 `instrument` 而不是 `span.enter()`：
        // `Entered` 守卫跨 `.await` 是不 Send 的，在多线程运行时里
        // 会让 future 变成 !Send（axum 要求 handler future 是 Send）。
        // `instrument` 把 span 附着到整个 future 上，语义等价且是 Send 的。
        let span_for_response = span.clone();
        async move {
            tracing::debug!(query = %query, "收到请求");

            let start = Instant::now();
            let response = next.run(request).await;
            let elapsed_ms = start.elapsed().as_millis() as u64;

            let status = response.status().as_u16();
            span_for_response.record("status", status);
            span_for_response.record("elapsed_ms", elapsed_ms);

            if status >= 500 {
                tracing::error!(status, elapsed_ms, "请求失败（服务端错误）");
            } else if status >= 400 {
                tracing::warn!(status, elapsed_ms, "请求被拒绝（客户端错误）");
            } else {
                tracing::info!(status, elapsed_ms, "请求完成");
            }

            response
        }
        .instrument(span)
        .await
    })
}

/// 打印一条「阶段开始」日志，并返回一个守卫。
///
/// 守卫 drop 时打印「阶段完成」并带上耗时 —— 这样即使中途
/// `?` 返回错误，耗时日志也一定会打出来（drop 总是执行）。
///
/// ```rust,ignore
/// let _g = stage_guard("加载索引");
/// // ... 干活 ...
/// // drop 时自动打印：完成: 加载索引 (耗时 1234ms)
/// ```
pub fn stage_guard(name: &'static str) -> StageGuard {
    tracing::info!("开始: {name}");
    StageGuard {
        name,
        start: Instant::now(),
    }
}

/// 阶段计时守卫。见 [`stage_guard`]。
pub struct StageGuard {
    name: &'static str,
    start: Instant,
}

impl StageGuard {
    /// 当前阶段已耗时的毫秒数。
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        tracing::info!(
            stage = self.name,
            elapsed_ms = self.elapsed_ms(),
            "完成: {}",
            self.name
        );
    }
}

/// 宏再导出 —— 让各项目 `use agent_logging::info;` 就能用。
///
/// 再导出而不是让每个项目直接依赖 `tracing`，
/// 是为了让「日志从哪来」只有一个答案。
pub use tracing::{debug, error, info, trace, warn};

/// 记录一条错误并附带上下文（`error!` 的语法糖）。
///
/// ```rust,ignore
/// agent_logging::log_error(&err, "加载索引失败", path = %path.display());
/// ```
#[macro_export]
macro_rules! log_error {
    ($err:expr, $msg:literal $(, $($field:tt)*)?) => {{
        tracing::error!(
            error = %$err,
            $($($field)*)?
            $msg
        );
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_is_unique_and_prefixed() {
        let a = next_request_id();
        let b = next_request_id();
        assert_ne!(a, b);
        assert!(a.starts_with("req-"), "请求 ID 应带前缀: {a}");
    }

    #[test]
    fn module_target_replaces_dash_with_underscore() {
        assert_eq!(module_target("rag-06"), "rag_06");
        assert_eq!(module_target("agent-13"), "agent_13");
        assert_eq!(module_target("plain"), "plain");
    }

    #[test]
    fn init_is_idempotent() {
        // 重复初始化不应 panic —— 测试里每个用例都可能调 init。
        init("rag-test");
        init("rag-test");
    }

    #[test]
    fn stage_guard_reports_elapsed_time() {
        let guard = stage_guard("测试阶段");
        assert!(guard.elapsed_ms() < 1000);
        drop(guard);
    }

    #[tokio::test]
    async fn trace_layer_passes_request_through_and_keeps_status() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::util::ServiceExt;

        let app = Router::new()
            .route(
                "/ok",
                get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
            )
            .route("/boom", get(|| async { "boom" }))
            .layer(trace_layer());

        // 中间件不能吞掉响应，也不能改状态码。
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let resp = app
            .oneshot(Request::builder().uri("/boom").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}
