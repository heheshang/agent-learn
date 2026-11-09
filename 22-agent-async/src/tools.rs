//! 工具层 —— 三种**执行轨道**。
//!
//! # ⚠️ 本项目最重要的一条认知
//!
//! FastAPI 教程的「In a hurry?」一节说：
//!
//! > 如果第三方库支持 `await`，就用 `async def`；
//! > 如果它**不支持**（大多数数据库库），就用普通 `def` ——
//! > FastAPI 会把它放进**外部线程池**再等待。
//!
//! 这在 Rust 里**完全对应**：
//!
//! | 轨道 | FastAPI | Rust / tokio | 什么时候用 |
//! |---|---|---|---|
//! | **异步轨道** | `async def` | `async fn` + `.await` | 等 I/O（HTTP / 数据库 / 队列） |
//! | **线程池轨道** | `def` | `spawn_blocking` | **同步**的阻塞调用（老数据库驱动、文件 IO） |
//! | **CPU 轨道** | `def` + 多进程 | `rayon` / 独立进程 | 纯计算（矩阵、图像） |
//!
//! ## ⚠️ 最容易踩的坑
//!
//! **把同步阻塞代码直接塞进 `async fn` 里。**
//!
//! ```ignore
//! async fn bad() {
//!     let rows = blocking_db_query(); // ❌ 阻塞整个事件循环
//! }
//! ```
//!
//! 这段时间里**整个 tokio 运行时都卡住** ——
//! 别的 HTTP 请求全部排队。
//!
//! 正确写法是 `tokio::task::spawn_blocking` ——
//! 它把这段同步代码丢到**独立的阻塞线程池**上，
//! 事件循环继续跑别的任务。
//!
//! 本项目的 [`BlockingTool`] 就是这条轨道的样本，
//! 并用回归测试锁住「阻塞工具**不会**卡住其它任务」。
//!
//! # 与 18 的关系
//!
//! **工具契约与错误分类直接复用 18**：
//!
//! - `agent_18::tool::ToolRegistry` —— 分发 + 填默认值 + Schema 校验
//! - `agent_18::tool::ErrorAwareTool` —— 声明失败类别的 trait
//! - `agent_18::middleware::ErrorHandlingMiddleware` —— 重试 / 降级 / 终止
//!
//! 本项目只加一层薄薄的适配器 [`AsyncToolAdapter`]，
//! 把「本项目的异步工具」翻译成「18 的工具」。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use agent_16::tools::ToolSpec;
use agent_18::error::ToolError;
use serde_json::{Value, json};

use agent_13::tools::calculate as calc_expression;

/// 工具执行返回给调用方的 future。
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;

/// 一个「可被模型调用」的异步工具。
pub trait AsyncTool: Send + Sync {
    /// 工具定义。
    fn spec(&self) -> ToolSpec;

    /// 执行工具。
    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a>;

    /// 这个工具是否**占用阻塞线程**（走了线程池轨道）。
    ///
    /// 默认 `false`（纯异步轨道）。
    fn is_blocking(&self) -> bool {
        false
    }
}

/// 工具运行时上下文。
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// 单次调用的超时。
    ///
    /// ⚠️ 只有**异步**工具真正用它。
    /// 阻塞工具的「超时」做不到真取消 ——
    /// `spawn_blocking` 的任务一旦跑起来就停不下来。
    pub timeout: Option<Duration>,
}

impl ToolContext {
    /// 空上下文。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 带超时的上下文。
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 适配器：把「异步工具」翻译成「18 的工具」
// ─────────────────────────────────────────────────────────────────────

/// 把 [`AsyncTool`] 适配成 18 的 [`agent_18::tool::ErrorAwareTool`]。
///
/// ## 为什么需要这一层
///
/// 18 的中间件（重试 / 降级 / 终止）吃的是 18 的 trait，
/// 而本项目要额外表达「这个工具走的是哪条轨道」。
///
/// 适配器是**最薄**的一层：只翻译类型，不加逻辑。
pub struct AsyncToolAdapter {
    tool: Arc<dyn AsyncTool>,
    timeout: Option<Duration>,
}

impl std::fmt::Debug for AsyncToolAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncToolAdapter")
            .field("tool", &self.tool.spec().name)
            .finish()
    }
}

impl AsyncToolAdapter {
    /// 包装一个异步工具。
    pub fn new(tool: Arc<dyn AsyncTool>, timeout: Option<Duration>) -> Self {
        Self { tool, timeout }
    }

    /// 底层工具。
    pub fn inner(&self) -> &Arc<dyn AsyncTool> {
        &self.tool
    }
}

impl agent_18::tool::ErrorAwareTool for AsyncToolAdapter {
    fn spec(&self) -> ToolSpec {
        self.tool.spec()
    }

    fn call<'a>(
        &'a self,
        args: &'a Value,
        _ctx: &'a agent_18::tool::ToolContext,
    ) -> agent_18::tool::ToolFuture<'a> {
        let ctx = ToolContext {
            timeout: self.timeout,
        };

        Box::pin(async move {
            let name = self.tool.spec().name;
            let observation = self.tool.call(args, &ctx).await?;
            Ok(agent_18::tool::ToolOutcome::ok(name, observation))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ① 天气工具 —— 异步轨道（真·HTTP）
// ─────────────────────────────────────────────────────────────────────

/// `get_weather` 工具名。
pub const GET_WEATHER_TOOL_NAME: &str = "get_weather";

/// 天气工具 —— 复用 14 的 `WeatherTool`。
///
/// ## 为什么复用
///
/// 14 的 `WeatherTool` 打的是 Open-Meteo 的**真实**接口（免 key），
/// 内部是**两跳**（地名 → 经纬度 → 预报）。
/// 这是本项目演示「I/O 密集型」的**最佳载体**：
/// 每次调用都要真的等网络。
///
/// 14 修过的东西（Schema 与实现对齐、错误分类）这里自动继承。
pub struct WeatherTool {
    inner: Arc<agent_14::weather::WeatherTool>,
}

impl std::fmt::Debug for WeatherTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeatherTool").finish_non_exhaustive()
    }
}

impl WeatherTool {
    /// 创建天气工具（连接真实的 Open-Meteo 服务）。
    pub fn new(default_days: usize, max_days: usize) -> Self {
        Self {
            inner: Arc::new(agent_14::weather::WeatherTool::new(default_days, max_days)),
        }
    }
}

impl AsyncTool for WeatherTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            GET_WEATHER_TOOL_NAME,
            "查询某个城市的实时天气和未来几天的预报。\
             需要实时的天气信息（温度、湿度、风力、降水）时必须调用这个工具，\
             不要凭记忆回答天气问题。如果工具返回错误，请根据错误提示修正参数后重试。",
            agent_14::weather::weather_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a> {
        // ⚠️ 14 的 `Tool::call` 用的错误类型是 14 的 `ToolError`，
        //    这里翻译成本项目的错误类型（= 18 的 `ToolError`）。
        //
        //    14 的四种错误 → 18 的三分类：
        //    - InvalidArgs / UnknownTool → Business（重试没用）
        //    - Upstream                  → Transient（重试可能好）
        //    - Execution                 → Fatal（内部逻辑挂了）
        let timeout = ctx.timeout;
        let tool_ctx = agent_14::tool::ToolContext::empty();

        Box::pin(async move {
            let fut = agent_14::tool::Tool::call(&*self.inner, args, &tool_ctx);

            match timeout {
                Some(d) => match tokio::time::timeout(d, fut).await {
                    Ok(Ok(outcome)) => Ok(outcome.observation),
                    Ok(Err(err)) => Err(translate_tool_error(&err)),
                    Err(_) => Err(ToolError::timeout(
                        GET_WEATHER_TOOL_NAME,
                        format!("天气接口在 {} ms 内没有响应", d.as_millis()),
                    )
                    .with_hint("上游超时了，可以稍后重试，或换一个城市名。")),
                },
                None => match fut.await {
                    Ok(outcome) => Ok(outcome.observation),
                    Err(err) => Err(translate_tool_error(&err)),
                },
            }
        })
    }
}

/// 把 14 的工具错误翻译成 18 的三分类错误。
fn translate_tool_error(err: &agent_14::error::ToolError) -> ToolError {
    match err {
        agent_14::error::ToolError::UnknownTool { name, available } => ToolError::unknown_tool(
            GET_WEATHER_TOOL_NAME,
            format!("{name}（可用：{available}）"),
        ),
        agent_14::error::ToolError::InvalidArgs { issues } => {
            ToolError::invalid_args(GET_WEATHER_TOOL_NAME, issues.clone())
        }
        agent_14::error::ToolError::Execution { message, .. } => {
            ToolError::internal(GET_WEATHER_TOOL_NAME, message.clone())
        }
        agent_14::error::ToolError::Upstream { message } => {
            ToolError::upstream(GET_WEATHER_TOOL_NAME, message.clone())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ② 计算工具 —— CPU 轨道（纯计算，零 I/O）
// ─────────────────────────────────────────────────────────────────────

/// `calculate` 工具名。
pub const CALCULATE_TOOL_NAME: &str = "calculate";

/// 计算工具 —— 复用 13 的算术求值器。
///
/// ## ⚠️ 为什么它值得单独放一条「CPU 轨道」
///
/// 算术求值是**纯 CPU** 的，耗时与 I/O 无关。
/// 把它当成「I/O 密集型」来并发**没有意义** ——
/// 并发只能提升**等待**的吞吐，不能提升**计算**的吞吐。
///
/// 这是本项目要澄清的一个常见误解：
/// **并发 ≠ 并行**（FastAPI 教程的核心论点）。
/// 真正的并行要靠 `rayon`（多线程）或独立进程。
pub struct CalculateTool;

impl std::fmt::Debug for CalculateTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculateTool").finish_non_exhaustive()
    }
}

impl AsyncTool for CalculateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            CALCULATE_TOOL_NAME,
            "计算一个算术表达式。只支持 + - * / ( ) 和数字，\
             例如 \"(12+34)*2\"。需要精确数字时必须调用这个工具，\
             不要自己心算。表达式非法时工具会告诉你哪里错了。",
            agent_13::tools::calculate_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let expression = args
                .get("expression")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();

            if expression.is_empty() {
                return Err(
                    ToolError::invalid_args(CALCULATE_TOOL_NAME, "`expression` 不能为空")
                        .with_hint("传入一个算术表达式，例如 \"(12+34)*2\"。"),
                );
            }

            match calc_expression(&expression) {
                Ok(value) => Ok(format!("{expression} = {value}")),
                Err(msg) => Err(ToolError::invalid_args(CALCULATE_TOOL_NAME, msg)
                    .with_hint("只支持 + - * / ( ) 和数字，请修正表达式。")),
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ③ 阻塞工具 —— 线程池轨道（FastAPI 的 `def`）
// ─────────────────────────────────────────────────────────────────────

/// `slow_report` 工具名。
pub const SLOW_REPORT_TOOL_NAME: &str = "slow_report";

/// 一个**故意阻塞**的工具 —— 用来演示线程池轨道。
///
/// ## 它模拟什么
///
/// 老式数据库驱动（同步）、同步文件 IO、遗留 gRPC 客户端 ——
/// 这些都是「不支持 `await`」的库。
/// FastAPI 的处置是「丢进线程池」，Rust 的处置是 `spawn_blocking`。
///
/// ## ⚠️ 关键设计
///
/// 这个工具**不是**用 `tokio::time::sleep` 模拟的 ——
/// 它用 `std::thread::sleep`，这才是**真阻塞**：
/// 它会把所在的 OS 线程占住不放。
///
/// 而且它**正确地**用 `spawn_blocking` 把自己丢进阻塞线程池，
/// 所以事件循环**不受影响**。
///
/// 回归测试 [`blocking_tool_does_not_stall_the_event_loop`] 锁住了这一点。
pub struct BlockingTool {
    /// 阻塞时长（毫秒）。
    pub block_ms: u64,
}

impl std::fmt::Debug for BlockingTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockingTool")
            .field("block_ms", &self.block_ms)
            .finish()
    }
}

impl BlockingTool {
    /// 创建阻塞工具。
    pub fn new(block_ms: u64) -> Self {
        Self { block_ms }
    }
}

impl AsyncTool for BlockingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            SLOW_REPORT_TOOL_NAME,
            "生成一份（故意很慢的）报表。这是一个**同步阻塞**的实现，\
             它会占用一个阻塞线程池线程，但不会卡住事件循环。",
            json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "报表主题，例如 \"sales\"。"
                    }
                },
                "required": ["topic"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        let block_ms = self.block_ms;

        Box::pin(async move {
            let topic = args
                .get("topic")
                .and_then(Value::as_str)
                .unwrap_or("未命名")
                .to_string();

            // ⚠️ **真阻塞** —— 占住当前 OS 线程。
            //
            //    所以必须用 `spawn_blocking` 把它丢到阻塞线程池，
            //    否则这 500ms 内整个事件循环都停摆。
            let result = tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(block_ms));
                format!("报表 `{topic}` 生成完毕（阻塞 {block_ms} ms）")
            })
            .await;

            match result {
                Ok(text) => Ok(text),
                Err(err) => Err(ToolError::internal(
                    SLOW_REPORT_TOOL_NAME,
                    format!("阻塞任务失败: {err}"),
                )),
            }
        })
    }

    fn is_blocking(&self) -> bool {
        true
    }
}

// ─────────────────────────────────────────────────────────────────────
// 注册表构建
// ─────────────────────────────────────────────────────────────────────

/// 构建默认的演示注册表（复用 18 的 `ToolRegistry`）。
///
/// 三个工具分别代表三条轨道：
/// - `get_weather` —— 异步轨道（真 HTTP）
/// - `calculate` —— CPU 轨道（纯计算）
/// - `slow_report` —— 线程池轨道（同步阻塞）
pub fn demo_registry(block_ms: u64, timeout: Option<Duration>) -> agent_18::tool::ToolRegistry {
    let tools: Vec<Arc<dyn AsyncTool>> = vec![
        Arc::new(WeatherTool::new(1, 3)),
        Arc::new(CalculateTool),
        Arc::new(BlockingTool::new(block_ms)),
    ];

    build_registry(tools, timeout)
}

/// 用一组异步工具构建 18 的注册表。
pub fn build_registry(
    tools: Vec<Arc<dyn AsyncTool>>,
    timeout: Option<Duration>,
) -> agent_18::tool::ToolRegistry {
    let adapted: Vec<Arc<dyn agent_18::tool::ErrorAwareTool>> = tools
        .into_iter()
        .map(|t| {
            Arc::new(AsyncToolAdapter::new(t, timeout)) as Arc<dyn agent_18::tool::ErrorAwareTool>
        })
        .collect();

    agent_18::tool::ToolRegistry::new(adapted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calculate_tool_computes_expressions() {
        let tool = CalculateTool;

        let out = tool
            .call(&json!({ "expression": "(12+34)*2" }), &ToolContext::empty())
            .await
            .unwrap();

        assert!(out.contains("92"), "实际输出：{out}");
    }

    #[tokio::test]
    async fn calculate_tool_rejects_bad_expression() {
        let tool = CalculateTool;

        let err = tool
            .call(&json!({ "expression": "1 +" }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    }

    #[tokio::test]
    async fn registry_reports_unknown_tool_as_business() {
        let registry = demo_registry(10, None);

        let err = registry
            .call("nope", &json!({}), &agent_18::tool::ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
        assert!(err.to_string().contains(GET_WEATHER_TOOL_NAME));
    }

    #[tokio::test]
    async fn registry_validates_arguments_before_calling() {
        let registry = demo_registry(10, None);

        // ⚠️ `resource` 不是 `slow_report` 的参数 → 校验失败。
        let err = registry
            .call(
                SLOW_REPORT_TOOL_NAME,
                &json!({ "resource": 42 }),
                &agent_18::tool::ToolContext::empty(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    }

    #[tokio::test]
    async fn blocking_tool_is_marked_as_blocking() {
        let tool = BlockingTool::new(10);
        assert!(tool.is_blocking());

        let tool = CalculateTool;
        assert!(!tool.is_blocking());
    }

    #[tokio::test]
    async fn blocking_tool_does_not_stall_the_event_loop() {
        // ⚠️ **本项目最重要的一条回归测试**。
        //
        // 如果 `BlockingTool` 直接在 async 里 `std::thread::sleep`，
        // 那 300ms 内整个事件循环都卡住 ——
        // 下面的「心跳任务」会被推迟到 300ms 之后才跑第 2 次。
        //
        // 正确实现（`spawn_blocking`）下，心跳应该能**按时**继续跳。
        let tool = BlockingTool::new(300);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        // 心跳任务：每 50ms 发一次。
        let heartbeat = tokio::spawn(async move {
            for i in 0..4 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let _ = tx.send(i);
            }
        });

        // 同时跑阻塞工具。
        let blocking = tokio::spawn(async move {
            tool.call(&json!({ "topic": "sales" }), &ToolContext::empty())
                .await
                .unwrap()
        });

        // 心跳应该在 250ms 内发完 4 次 —— 前提是事件循环没被卡住。
        let start = std::time::Instant::now();
        let mut beats = 0usize;
        while let Some(_i) = rx.recv().await {
            beats += 1;
            if beats == 4 {
                break;
            }
        }
        let beat_elapsed = start.elapsed();

        heartbeat.await.unwrap();
        blocking.await.unwrap();

        assert_eq!(beats, 4);
        assert!(
            beat_elapsed.as_millis() < 280,
            "心跳被阻塞工具拖慢了（耗时 {}ms）—— 说明它没走 spawn_blocking",
            beat_elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn registry_executes_calculate_through_the_adapter() {
        let registry = demo_registry(10, None);

        let outcome = registry
            .call(
                CALCULATE_TOOL_NAME,
                &json!({ "expression": "1+1" }),
                &agent_18::tool::ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.observation.contains("2"));
    }
}
