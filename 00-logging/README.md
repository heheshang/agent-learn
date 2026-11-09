# 00-logging —— workspace 共享日志设施

全仓库 14 个学习项目，每一个都是独立的可执行服务。
如果每个项目各自实现一套日志初始化，日志格式、级别语义、
请求追踪方式就会各写各的，跨项目对照实验时也没法把日志放一起看。

所以日志设施抽成**独立的 workspace 成员 crate**，所有项目共用。

## 三个能力

| 能力 | 说明 |
|---|---|
| `init(service)` | 初始化全局 subscriber，一次性设置格式、级别、来源 |
| `trace_layer()` | HTTP 请求追踪中间件：请求 ID、方法、路径、耗时、状态码 |
| `stage_guard(name)` | 阶段耗时守卫 —— 进入时打日志，退出时打耗时 |

外加宏再导出：`info!` / `debug!` / `warn!` / `error!` / `trace!`。

## 快速开始

```rust
use agent_logging::{info, init, trace_layer};

#[tokio::main]
async fn main() {
    init("rag-06");

    let app: axum::Router = axum::Router::new()
        .route("/", axum::routing::get(|| async { "ok" }))
        .layer(trace_layer());

    info!(addr = "127.0.0.1:3000", "服务已启动");
    // ...
}
```

## 级别约定

| 级别 | 用途 | 典型场景 |
|---|---|---|
| `ERROR` | 请求失败、服务不可用、数据不一致 | LLM 调用失败、索引加载失败 |
| `WARN` | 能兜住但不该发生 | 参数非法、工具执行失败、降级 |
| `INFO` | 关键业务节点 | 服务启动、索引就绪、一次请求的最终结果 |
| `DEBUG` | 过程细节 | 召回了哪些块、Prompt 全文、重排分数 |
| `TRACE` | 逐行细节 | 每条候选块的原始分值 |

## 过滤

用环境变量 `RUST_LOG` 控制，遵循 `tracing-subscriber` 的 `EnvFilter` 语法：

```bash
RUST_LOG=info                          # 全局 info
RUST_LOG=debug,rag_06=trace            # rag_06 单独开到 trace
RUST_LOG=warn,rag_12::retriever=debug  # 只对某个模块开 debug
```

不设 `RUST_LOG` 时用各项目传入的默认级别（一般是 `info`）。

## 为什么不用 `println!`

1. **可过滤**：`println!` 没法按模块级别关掉，debug 输出一旦写进去
   就永远在生产路径上（每次 LLM 调用都刷一遍 Prompt，是很大的噪音）。
2. **有结构**：`info!(chunk_count = 12, ...)` 出来的字段可以被
   日志采集器解析，而 `println!("{}", ...)` 只有一坨文本。
3. **带 span 上下文**：HTTP 请求里的日志自动带上 `request_id`，
   一次请求触发的所有日志能串起来。

## 请求追踪

`trace_layer()` 会在每个请求上开一个 `http_request` span：

| 字段 | 含义 |
|---|---|
| `request_id` | 进程内唯一请求 ID（`req-7` 形式） |
| `method` | HTTP 方法 |
| `path` | 请求路径 |
| `status` | 响应状态码（响应后写入） |
| `elapsed_ms` | 耗时毫秒（响应后写入） |

## 全仓库的三条硬约定

1. **4xx 用 `warn!`，5xx 用 `error!`** —— 告警不能被 4xx 淹没。
2. **降级必须打 `warn!`** —— 降级是「能兜住但不该发生」，静默降级等于埋雷。
3. **不打 `println!` 混进业务逻辑** —— `println!` 没法按模块关掉，
   debug 输出一旦写进去就永远在生产路径上。

---

## 代码调用详细流程图（函数级）

> 节点写的是**真实函数名**，边上标的是**调用点行号**。全部对着源码。

### 初始化：一次性的全局设置

```mermaid
flowchart TD
    MAIN["main.rs:1<br/>main()"] --> INIT["lib.rs:109<br/>init(service)"]
    INIT --> INITL["lib.rs:114<br/>init_with_level(service, default_level)"]
    INITL --> ENV["★ EnvFilter::try_from_default_env()<br/>RUST_LOG 优先"]
    ENV -->|"RUST_LOG 没设"| MKFILTER["EnvFilter::new（{level},{module}={level}）"]
    ENV -->|"RUST_LOG 设了"| USEENV["直接用它"]
    MKFILTER --> SUB["tracing_subscriber::fmt()<br/>.with_env_filter().with_target()<br/>.with_thread_ids().with_file().with_line_number()"]
    USEENV --> SUB
    SUB --> SET["★ set_global_default()<br/>（幂等：重复 init 不会 panic）"]
    SET --> MOD["lib.rs:153<br/>module_target(service)<br/>把 '-' 换成 '_'"]

    style ENV fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style SET fill:#ffe6e6,stroke:#d94a4a,stroke-width:2px
```

### 请求追踪：一个 HTTP 请求的日志生命周期

**核心是 `trace_handler`** —— 它给每个请求开一个 span，并把 `request_id` 传给这次请求触发的所有日志。

```mermaid
flowchart TD
    REQ["HTTP 请求"] --> LAYER["routes.rs::.layer(trace_layer())"]
    LAYER --> TL["lib.rs:193<br/>trace_layer()"]
    TL --> TH["★ lib.rs:206<br/>trace_handler(request, next)"]
    TH --> RID["lib.rs:94<br/>next_request_id()<br/>进程内唯一（req-7 形式）"]
    TH --> META["取 method / path / query"]
    META --> SPAN["tracing::info_span!（http_request,<br/>request_id, method, path)"]
    SPAN --> INSTR["★ 用 instrument 而不是 span.enter()"]
    INSTR --> NEXT["next.run(request).await"]
    NEXT --> RESP["拿到 Response"]
    RESP --> STATUS["record（status, status）<br/>record（elapsed_ms, elapsed）"]
    STATUS --> BR{"status?"}
    BR -->|">= 500"| E["★ tracing::error!<br/>「请求失败（服务端错误）」"]
    BR -->|">= 400"| W["★ tracing::warn!<br/>「请求被拒绝（客户端错误）」"]
    BR -->|"其它"| I["tracing::info!"]

    NOTE["⚠️ 为什么用 instrument 而不是 span.enter()<br/>Entered 守卫跨 .await 不 Send<br/>会让 future 变成 !Send<br/>而 axum 要求 handler future 是 Send"]

    style TH fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style INSTR fill:#ffe6e6,stroke:#d94a4a,stroke-width:2px
    style E fill:#ffe6e6,stroke:#d94a4a
    style W fill:#fff4e6,stroke:#e8a33d
    style NOTE fill:#fff4e6,stroke:#e8a33d
```

### 级别语义：为什么 4xx 和 5xx 要分开

**这是全仓库的硬约定** —— 告警不能被 4xx 淹没。

```mermaid
flowchart LR
    RESP["HTTP 响应"] --> CODE{"status code"}
    CODE -->|"2xx / 3xx"| OK["info!<br/>正常业务"]
    CODE -->|"4xx"| WARN["★ warn!<br/>客户端错误"]
    CODE -->|"5xx"| ERR["★ error!<br/>服务端错误"]
    CODE -->|"降级发生"| DEG["★ warn!<br/>降级必须打出来"]

    NOTE["静默降级 = 埋雷<br/>能兜住但不该发生的事<br/>必须留下痕迹"]

    style WARN fill:#fff4e6,stroke:#e8a33d
    style ERR fill:#ffe6e6,stroke:#d94a4a
    style DEG fill:#ffe6e6,stroke:#d94a4a
    style NOTE fill:#ffe6e6,stroke:#d94a4a
```

### 阶段耗时守卫

```mermaid
flowchart TD
    CALL["业务代码"] --> SG["lib.rs:261<br/>stage_guard（阶段名）"]
    SG --> GUARD["StageGuard"]
    GUARD --> WORK["执行耗时工作"]
    WORK --> DROP["★ impl Drop for StageGuard<br/>（lib.rs:282）"]
    DROP --> ELAPSED["lib.rs:277<br/>elapsed_ms()"]
    ELAPSED --> LOG["info! 记录耗时"]

    NOTE["★ 用 Drop 实现守卫<br/>即使中途 return / 提前退出<br/>耗时也会被记录"]

    style DROP fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style NOTE fill:#fff4e6,stroke:#e8a33d
```

## 踩坑记录

### `clippy::type_complexity`：`FromFnLayer` 签名太长

Axum 0.8 的 `from_fn` 返回 `FromFnLayer<...>`，直接写出来会触发
clippy 的 `type_complexity` 告警（三个泛型参数嵌套）。抽成别名：

```rust
pub type TraceLayer = axum::middleware::FromFnLayer<
    fn(Request, Next) -> Pin<Box<dyn Future<Output = Response> + Send>>,
    (),
    (Request,),
>;
```

### 请求 ID 为什么不用 uuid

日志里的请求 ID 只需要**在一次进程运行内唯一**，不需要跨进程/跨机器唯一。
跨机器唯一是分布式追踪（OpenTelemetry）该解决的问题 ——
那是本仓库不打算引入的复杂度。用原子计数器就够了。

## 技术参考速查

| 主题 | 关键点 | 本项目位置 |
|---|---|---|
| 发射端 | `tracing` 的 `info!` / `debug!` / `warn!` / `error!` / `trace!` | `lib.rs` 再导出 |
| 接收端 | `tracing-subscriber` 的 `fmt` + `EnvFilter` + `chrono` | `init_with_level` |
| 过滤 | `RUST_LOG`，`EnvFilter` 语法（可按 crate / 模块） | `init_with_level` |
| 请求追踪 | Axum 0.8 `from_fn` + `tracing::Instrument` | `trace_handler` |
| 阶段计时 | `tracing::Span` + `Drop` 守卫 | `stage_guard` |
| 目标过滤 | crate 名 `rag-06` → 模块 target `rag_06` | `module_target` |
