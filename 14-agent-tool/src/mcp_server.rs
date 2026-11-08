//! 内置 MCP 服务器 —— 一个真的能被 MCP 客户端连上的服务端。
//!
//! ## 为什么要有它
//!
//! 本项目的目标是「手撕 MCP」。但 MCP 协议的两端必须能对上，
//! 否则只能对着文档空想。
//!
//! 所以这里实现一个**最小但完整**的 MCP 服务端（Streamable HTTP 传输）：
//!
//! | 路径 | 说明 |
//! |---|---|
//! | `POST /mcp` | JSON-RPC 2.0 端点（`initialize` / `tools/list` / `tools/call`） |
//! | `GET /mcp` | SSE 事件流（本实现返回 405，说明不支持流式推送） |
//! | `GET /health` | 健康检查 |
//!
//! 它暴露两个工具：
//!
//! - `echo` —— 回显文本（最小的工具，用来验证协议通路）
//! - `get_weather` —— **和本地天气工具同名、同参数、同语义**
//!
//! ## 同名同义的意义
//!
//! 这是本项目最想演示的一件事：
//!
//! ```text
//! 本地工具：get_weather(location)          → 直接调 Open-Meteo
//! MCP 工具：get_weather(location)          → 经 JSON-RPC 转发到 MCP 服务端
//! ```
//!
//! 在 Agent 眼里**完全一样** —— 它只看到一个 `ToolSpec`。
//! 「工具来自哪里」被彻底解耦了。
//!
//! 这正是 MCP 的核心价值，也是 LangChain 里
//! `MCPAdapter(...).list_tools()` 之后能直接喂给 `create_agent` 的原因。

use agent_logging::{debug, info, warn};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::mcp::PROTOCOL_VERSION;
use crate::tool::Tool;
use crate::weather::{GET_WEATHER_TOOL_NAME, Units, WeatherTool, weather_schema};

/// MCP 工具处理器的 future 类型。
type HandlerFuture<'a> = Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>>;

/// MCP 工具处理器。
type Handler = Arc<dyn Fn(Value) -> HandlerFuture<'static> + Send + Sync>;

/// 内置 MCP 服务器的状态。
#[derive(Clone)]
pub struct McpServerState {
    /// 工具名 → 处理函数。
    tools: Arc<HashMap<String, Handler>>,

    /// 已建立的会话（会话 ID → 工具清单快照）。
    sessions: Arc<std::sync::Mutex<HashMap<String, Vec<Value>>>>,
}

impl std::fmt::Debug for McpServerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerState")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl McpServerState {
    /// 创建状态（带默认工具）。
    pub fn new() -> Self {
        let mut tools: HashMap<String, Handler> = HashMap::new();

        tools.insert(
            "echo".to_string(),
            Arc::new(|args: Value| {
                Box::pin(async move {
                    let text = args
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if text.is_empty() {
                        return Err("缺少参数 `text`".to_string());
                    }
                    Ok(format!("回显: {text}"))
                })
            }),
        );

        // `get_weather` 走**同一个**天气后端 —— 保证语义一致。
        let weather = Arc::new(WeatherTool::new(1, 7));
        tools.insert(
            GET_WEATHER_TOOL_NAME.to_string(),
            Arc::new(move |args: Value| {
                let weather = weather.clone();
                Box::pin(async move {
                    let location = args
                        .get("location")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if location.is_empty() {
                        return Err("缺少参数 `location`".to_string());
                    }

                    let units = args
                        .get("units")
                        .and_then(Value::as_str)
                        .and_then(Units::parse)
                        .unwrap_or(Units::Celsius);
                    let days = args
                        .get("days")
                        .and_then(Value::as_i64)
                        .unwrap_or(1)
                        .clamp(1, 7) as usize;

                    // 走本地的 `Tool::call` —— 于是 MCP 服务器和本地工具
                    // **共用同一份实现**，语义不可能漂移。
                    let outcome = weather
                        .call(
                            &json!({
                                "location": location,
                                "units": match units {
                                    Units::Celsius => "celsius",
                                    Units::Fahrenheit => "fahrenheit",
                                },
                                "days": days,
                            }),
                            &crate::tool::ToolContext::empty(),
                        )
                        .await;

                    match outcome {
                        Ok(result) => Ok(result.observation),
                        Err(e) => Err(e.to_string()),
                    }
                })
            }),
        );

        Self {
            tools: Arc::new(tools),
            sessions: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// 工具清单（`tools/list` 的返回）。
    pub fn tool_list(&self) -> Vec<Value> {
        vec![
            json!({
                "name": "echo",
                "description": "回显一段文本。用于验证 MCP 链路是否通畅。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "要回显的文本" }
                    },
                    "required": ["text"],
                    "additionalProperties": false
                }
            }),
            json!({
                "name": GET_WEATHER_TOOL_NAME,
                "description": "查询某个城市的实时天气和未来几天的预报。",
                "inputSchema": weather_schema()
            }),
        ]
    }

    /// 调用工具。
    async fn call_tool(&self, name: &str, args: &Value) -> Result<String, String> {
        let Some(handler) = self.tools.get(name) else {
            return Err(format!(
                "未知工具 `{name}`。可用工具: {}",
                self.tools.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        };
        handler(args.clone()).await
    }
}

impl Default for McpServerState {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP JSON-RPC 请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    /// 请求 ID（通知没有）。
    #[serde(default)]
    pub id: Option<u64>,
    /// 方法名。
    pub method: String,
    /// 参数。
    #[serde(default)]
    pub params: Value,
}

/// POST /mcp —— JSON-RPC 2.0 端点。
#[axum::debug_handler]
pub async fn handle_rpc(
    State(state): State<McpServerState>,
    headers: axum::http::HeaderMap,
    Json(request): Json<JsonRpcRequest>,
) -> axum::response::Response {
    // ── 通知（无 ID）──
    //
    // `notifications/initialized` —— 客户端确认握手完成。
    // JSON-RPC 规定通知不返回响应体。
    if request.id.is_none() {
        let mut response_headers = axum::http::HeaderMap::new();
        response_headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        return (
            axum::http::StatusCode::ACCEPTED,
            response_headers,
            Json(json!({})),
        )
            .into_response();
    }

    let id = request.id.unwrap();

    debug!(method = %request.method, id, "收到 MCP JSON-RPC 请求");

    let result: Result<Value, (i32, String)> = match request.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": {
                "name": "agent-learn-14-mcp",
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
        "tools/list" => Ok(json!({ "tools": state.tool_list() })),
        "tools/call" => {
            let name = request
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let arguments = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

            match state.call_tool(&name, &arguments).await {
                Ok(text) => {
                    info!(tool = %name, "MCP 工具调用成功");
                    Ok(json!({
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    }))
                }
                Err(message) => {
                    warn!(tool = %name, error = %message, "MCP 工具调用失败");
                    Ok(json!({
                        "content": [{ "type": "text", "text": message }],
                        "isError": true
                    }))
                }
            }
        }
        "ping" => Ok(json!({})),
        other => Err((-32601, format!("方法未找到: {other}"))),
    };

    // ── 响应 ──────────────────────────────────────────────────
    //
    // ⚠️ 会话 ID：**只有 `initialize` 会新建会话**，
    //    其它方法复用现有会话。
    //
    // 会话 ID 通过响应头 `Mcp-Session-Id` 传出去 ——
    // 这是 Streamable HTTP 传输的规定。
    let mut response_headers = axum::http::HeaderMap::new();
    response_headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );

    let body = match result {
        Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
        Err((code, message)) => {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": code, "message": message }
            })
        }
    };

    // 客户端带了会话 ID —— 说明它已经握过手了。
    // 没带且是 initialize —— 发一个新的。
    let session_header = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let session_id = match (&session_header, request.method.as_str()) {
        (Some(id), _) => id.clone(),
        (None, "initialize") => {
            let id = format!("sess-{}", uuid_like());
            if let Ok(mut sessions) = state.sessions.lock() {
                sessions.insert(id.clone(), state.tool_list());
            }
            info!(session_id = %id, "已创建 MCP 会话");
            id
        }
        (None, _) => String::new(),
    };

    if !session_id.is_empty()
        && let Ok(value) = axum::http::HeaderValue::from_str(&session_id)
    {
        response_headers.insert("mcp-session-id", value);
    }

    (response_headers, Json(body)).into_response()
}

/// GET /mcp —— SSE 事件流。
///
/// 本实现**不支持**服务端主动推送，所以返回 405。
/// 但保留这个端点是为了让协议语义完整 ——
/// 客户端会先探测它。
#[axum::debug_handler]
pub async fn handle_stream() -> impl IntoResponse {
    (
        axum::http::StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "error": "本服务器不提供服务端推送流（SSE）",
            "hint": "所有交互都通过 POST /mcp 的 JSON-RPC 请求完成"
        })),
    )
}

/// GET /health —— 健康检查。
#[axum::debug_handler]
pub async fn health(State(state): State<McpServerState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "service": "14-agent-tool 内置 MCP 服务器",
        "protocol_version": PROTOCOL_VERSION,
        "tools": state.tool_list(),
    }))
}

/// 构造 MCP 服务器的路由。
pub fn router(state: McpServerState) -> axum::Router {
    axum::Router::new()
        .route("/mcp", post(handle_rpc).get(handle_stream))
        .route("/health", get(health))
        .with_state(state)
}

/// 生成一个伪随机的会话 ID。
///
/// ## 为什么不用 `uuid` crate
///
/// 为一个会话 ID 引入一个新依赖不值得。
/// 这里用「时间戳 + 计数器」拼一个够用的字符串 ——
/// 它只需要在本进程内唯一，不需要密码学强度。
fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(1);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

    format!("{nanos:x}{counter:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tool_list_includes_weather_tool() {
        let state = McpServerState::new();
        let list = state.tool_list();

        let names: Vec<&str> = list
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();

        assert!(names.contains(&"echo"));
        assert!(names.contains(&"get_weather"));
    }

    #[tokio::test]
    async fn weather_tool_schema_matches_local_one() {
        // ⚠️ 这是本项目的核心断言：
        // MCP 暴露的 `get_weather` 和本地天气工具**参数契约完全一致**。
        let state = McpServerState::new();
        let list = state.tool_list();

        let mcp_schema = list
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some("get_weather"))
            .and_then(|t| t.get("inputSchema"))
            .cloned()
            .unwrap();

        assert_eq!(mcp_schema, weather_schema());
    }

    #[tokio::test]
    async fn echo_tool_runs() {
        let state = McpServerState::new();
        let result = state.call_tool("echo", &json!({ "text": "你好" })).await;
        assert_eq!(result.unwrap(), "回显: 你好");
    }

    #[tokio::test]
    async fn unknown_tool_lists_available_tools() {
        let state = McpServerState::new();
        let err = state.call_tool("nope", &json!({})).await.unwrap_err();

        assert!(err.contains("未知工具"));
        assert!(err.contains("echo"));
    }

    #[tokio::test]
    async fn missing_required_arg_is_reported() {
        let state = McpServerState::new();
        let err = state.call_tool("echo", &json!({})).await.unwrap_err();
        assert!(err.contains("text"));
    }

    #[test]
    fn session_ids_are_unique() {
        let a = uuid_like();
        let b = uuid_like();
        assert_ne!(a, b);
    }
}
