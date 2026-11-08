//! MCP（Model Context Protocol）客户端 —— 对应 LangChain 的 `MCPAdapter`。
//!
//! ## MCP 是什么
//!
//! MCP 是一个开放协议，规定了「应用如何把工具暴露给语言模型」。
//! 参考 [modelcontextprotocol.io](https://modelcontextprotocol.io) 与
//! [Smithery](https://smithery.ai/)（MCP 服务器的公共注册中心）。
//!
//! LangChain 教程里的对应物是：
//!
//! ```python
//! async with MCPAdapter("https://example.com/mcp") as adapter:
//!     tools = await adapter.list_tools()
//!     agent = create_agent("claude-sonnet-4-6", tools)
//! ```
//!
//! ## 本项目实现什么
//!
//! **Streamable HTTP 传输**上的 JSON-RPC 2.0，覆盖三个方法：
//!
//! | 方法 | 方向 | 作用 |
//! |---|---|---|
//! | `initialize` | 客户端 → 服务端 | 协商协议版本，建立会话 |
//! | `tools/list` | 客户端 → 服务端 | 拿到工具清单（名字 / 描述 / 输入 Schema） |
//! | `tools/call` | 客户端 → 服务端 | 调用某个工具 |
//!
//! ## 关键设计：MCP 工具 → 本地 `Tool`
//!
//! MCP 服务器返回的工具定义**格式和本地工具一样**
//! （名字 + 描述 + JSON Schema）。所以本项目写了一个适配器
//! [`McpTool`]，把远端工具包装成本地 [`Tool`]。
//!
//! 这样做的价值是**可互换性**：
//!
//! - 本地天气工具（[`crate::weather::WeatherTool`]）
//! - MCP 服务器上的天气工具
//!
//! 在 Agent 眼里**没有任何区别** —— 都是 `Box<dyn Tool>`。
//! 这正是 MCP 的核心价值：**工具的来源与工具的使用方式解耦**。
//!
//! ## 会话管理
//!
//! Streamable HTTP 传输里，服务端在 `initialize` 的响应头里返回
//! `Mcp-Session-Id`，之后每个请求都要带上它。
//! 服务端重启会导致会话失效 —— 此时客户端会**自动重新 initialize**。

use agent_logging::{debug, info, warn};
use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolFuture, ToolOutcome, ToolSpec};

/// 本客户端实现的 MCP 协议版本。
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC 2.0 响应。
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    /// 请求 ID（服务端必须回填）。
    #[serde(default)]
    pub id: Option<u64>,

    /// 结果（成功时）。
    #[serde(default)]
    pub result: Option<Value>,

    /// 错误（失败时）。
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 错误。
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    /// 错误码。
    pub code: i32,
    /// 错误描述。
    pub message: String,
    /// 附加数据。
    #[serde(default)]
    pub data: Option<Value>,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code={})", self.message, self.code)
    }
}

/// MCP 远程工具的定义。
#[derive(Debug, Clone, Deserialize)]
pub struct McpToolDefinition {
    /// 工具名。
    pub name: String,
    /// 工具描述。
    #[serde(default)]
    pub description: String,
    /// 输入 Schema。
    ///
    /// ⚠️ JSON 字段名是 `inputSchema`（驼峰）—— MCP 协议规定的。
    ///    Rust 侧用 `rename` 映射，否则反序列化拿不到。
    #[serde(default = "empty_object_schema", rename = "inputSchema")]
    pub input_schema: Value,
}

/// 空对象 Schema（MCP 服务器可以省略 `inputSchema`）。
fn empty_object_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// MCP 工具调用结果。
#[derive(Debug, Clone, Deserialize)]
pub struct McpCallResult {
    /// 内容块列表。
    #[serde(default)]
    pub content: Vec<McpContent>,

    /// 是否是错误结果。
    ///
    /// ⚠️ JSON 字段名是 `isError`（驼峰）—— MCP 协议规定的。
    ///    Rust 侧用 `rename` 映射，否则反序列化拿不到（永远是 `false`）。
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

/// MCP 内容块。
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpContent {
    /// 文本块。
    Text {
        /// 文本内容。
        text: String,
    },
    /// 其它类型的块（本项目只保留原始 JSON）。
    #[serde(untagged)]
    Other(Value),
}

impl McpContent {
    /// 提取文本内容。
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text { text } => Some(text),
            McpContent::Other(_) => None,
        }
    }
}

/// MCP 客户端。
///
/// ## 生命周期
///
/// ```text
/// connect()  → initialize + notifications/initialized（拿到 session id）
/// list_tools() → tools/list
/// call_tool() → tools/call
/// ```
///
/// `connect()` 是**幂等**的 —— 已连接就直接返回；
/// 遇到「会话失效」会重新握手。
pub struct McpClient {
    endpoint: String,
    client: reqwest::Client,
    session_id: std::sync::Mutex<Option<String>>,
    next_id: std::sync::Mutex<u64>,
    client_name: String,
    client_version: String,
    /// 已发现的工具（`name` → 定义）。
    tools: std::sync::Mutex<HashMap<String, McpToolDefinition>>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("endpoint", &self.endpoint)
            .field(
                "session_id",
                &self.session_id.lock().map(|s| s.clone()).unwrap_or(None),
            )
            .finish()
    }
}

impl McpClient {
    /// 创建客户端。
    pub fn new(endpoint: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("agent-learn/14-agent-tool")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            endpoint: endpoint.into(),
            client,
            session_id: std::sync::Mutex::new(None),
            next_id: std::sync::Mutex::new(1),
            client_name: "agent-learn-14".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            tools: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// 服务端地址。
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// 当前会话 ID（`None` = 还没握手）。
    pub fn session_id(&self) -> Option<String> {
        self.session_id.lock().ok().and_then(|s| s.clone())
    }

    /// 已发现的工具数量。
    pub fn tool_count(&self) -> usize {
        self.tools.lock().map(|t| t.len()).unwrap_or(0)
    }

    /// 握手（幂等）。
    pub async fn connect(&self) -> Result<(), ToolError> {
        if let Some(session_id) = self.session_id() {
            debug!(session_id = %session_id, "MCP 会话已存在，跳过握手");
            return Ok(());
        }

        // 握手：协商协议版本。
        //
        // ⚠️ 会话 ID 在**响应头**里 —— `send()` 负责把它存下来，
        //    所以这里不需要（也不应该）再发一次 initialize。
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "clientInfo": {
                        "name": self.client_name,
                        "version": self.client_version
                    }
                }),
            )
            .await?;

        let _ = result;

        // 发送 initialized 通知（无 ID，服务器不应回响应）。
        self.notify("notifications/initialized", json!({})).await?;

        info!(
            endpoint = %self.endpoint,
            session_id = ?self.session_id(),
            "MCP 握手完成"
        );

        Ok(())
    }

    /// 列出服务器上的工具。
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, ToolError> {
        self.connect().await?;

        let result = self.request("tools/list", json!({})).await?;

        let definitions = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| ToolError::upstream("tools/list 响应缺少 tools 字段".to_string()))?;

        let mut tools = Vec::with_capacity(definitions.len());
        for definition in definitions {
            let definition: McpToolDefinition = serde_json::from_value(definition.clone())
                .map_err(|e| ToolError::upstream(format!("解析工具定义失败: {e}")))?;
            tools.push(definition);
        }

        // 缓存下来 —— `as_tools()` 要用。
        if let Ok(mut cache) = self.tools.lock() {
            cache.clear();
            for tool in &tools {
                cache.insert(tool.name.clone(), tool.clone());
            }
        }

        info!(tool_count = tools.len(), "MCP 工具列表已拉取");

        Ok(tools)
    }

    /// 调用服务器上的工具。
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<McpCallResult, ToolError> {
        self.connect().await?;

        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;

        info!(tool = %name, "MCP tools/call 已发出");

        serde_json::from_value(result)
            .map_err(|e| ToolError::upstream(format!("解析 tools/call 响应失败: {e}")))
    }

    /// 把远端工具包装成本地 [`Tool`]。
    ///
    /// ## 这是 MCP 最核心的价值
    ///
    /// 返回的工具和本地工具**完全同构** ——
    /// 于是 Agent 循环不需要知道这个工具是本地的还是远端的。
    pub fn as_tools(self: &std::sync::Arc<Self>) -> Vec<std::sync::Arc<dyn Tool>> {
        let Ok(cache) = self.tools.lock() else {
            return Vec::new();
        };

        let mut definitions: Vec<McpToolDefinition> = cache.values().cloned().collect();
        // 排序 —— 保证工具清单顺序稳定（否则提示词每次都不一样）。
        definitions.sort_by(|a, b| a.name.cmp(&b.name));

        definitions
            .into_iter()
            .map(|definition| {
                std::sync::Arc::new(McpTool {
                    client: self.clone(),
                    definition,
                }) as std::sync::Arc<dyn Tool>
            })
            .collect()
    }

    /// 发送一个 JSON-RPC 请求。
    async fn request(&self, method: &str, params: Value) -> Result<Value, ToolError> {
        let id = {
            let mut counter = self
                .next_id
                .lock()
                .map_err(|_| ToolError::upstream("MCP 客户端内部状态锁中毒".to_string()))?;
            let id = *counter;
            *counter += 1;
            id
        };

        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        let response = self.send(body).await?;

        if let Some(err) = response.error {
            warn!(method, error = %err, "MCP JSON-RPC 调用失败");
            return Err(ToolError::upstream(format!(
                "MCP 调用 `{method}` 失败: {err}"
            )));
        }

        Ok(response.result.unwrap_or(Value::Null))
    }

    /// 发送一个 JSON-RPC 通知（无返回值）。
    async fn notify(&self, method: &str, params: Value) -> Result<(), ToolError> {
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        self.send(body).await?;

        Ok(())
    }

    /// 发送原始请求并解析响应。
    async fn send(&self, body: Value) -> Result<JsonRpcResponse, ToolError> {
        let mut request = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        // 带会话 ID（如果有）。
        if let Some(session_id) = self.session_id() {
            request = request.header("Mcp-Session-Id", session_id);
        }

        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| ToolError::upstream(format!("连接 MCP 服务端失败: {e}")))?;

        let status = response.status();

        // ⚠️ 会话可能已经失效（服务端重启过）——
        //    409/404 都可能表示「会话没了」。这时清掉会话 ID，
        //    让下一次调用重新握手。
        if (status == reqwest::StatusCode::CONFLICT || status == reqwest::StatusCode::NOT_FOUND)
            && let Ok(mut guard) = self.session_id.lock()
        {
            *guard = None;
            warn!(
                status = status.as_u16(),
                "MCP 会话失效 —— 已清空会话 ID，下次调用会重新握手"
            );
        }

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            warn!(status = status.as_u16(), "MCP 服务端返回错误状态");
            return Err(ToolError::upstream(format!(
                "MCP 服务端返回 HTTP {status}: {}",
                text.chars().take(200).collect::<String>()
            )));
        }

        // 抓会话 ID（`initialize` 的响应头里有）。
        if let Some(session_id) = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            && let Ok(mut guard) = self.session_id.lock()
        {
            *guard = Some(session_id);
        }

        let text = response
            .text()
            .await
            .map_err(|e| ToolError::upstream(format!("读取 MCP 响应失败: {e}")))?;

        // Streamable HTTP 允许返回 SSE 格式 —— 这里只支持纯 JSON，
        // SSE 的解析留给需要流式输出的场景。
        parse_json_rpc_response(&text)
    }
}

/// 解析 JSON-RPC 响应。
///
/// ## 为什么还要兼容 SSE
///
/// Streamable HTTP 传输规定：服务器可以返回
/// `Content-Type: application/json`（单个响应）
/// 或 `text/event-stream`（SSE 流）。
///
/// 大多数简单实现用前者，但有些服务器用后者返回单条响应 ——
/// 格式是 `data: {json}\n\n`。
///
/// 所以这里两种都支持。
pub fn parse_json_rpc_response(text: &str) -> Result<JsonRpcResponse, ToolError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ToolError::upstream("MCP 服务端返回了空响应".to_string()));
    }

    // 直接是 JSON。
    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed)
            .map_err(|e| ToolError::upstream(format!("解析 JSON-RPC 响应失败: {e}")));
    }

    // SSE：从 `data:` 行里取 JSON。
    for line in trimmed.lines() {
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(payload) {
            return Ok(response);
        }
    }

    Err(ToolError::upstream(format!(
        "无法解析 MCP 响应: {}",
        trimmed.chars().take(200).collect::<String>()
    )))
}

/// MCP 远程工具的本地适配器。
///
/// 把「远端调用」包装成「本地 [`Tool::call`]」。
pub struct McpTool {
    /// 共享的客户端（用来复用会话）。
    client: std::sync::Arc<McpClient>,
    /// 远端工具定义。
    definition: McpToolDefinition,
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTool")
            .field("name", &self.definition.name)
            .field("endpoint", &self.client.endpoint)
            .finish()
    }
}

impl McpTool {
    /// 工具定义。
    pub fn definition(&self) -> &McpToolDefinition {
        &self.definition
    }
}

impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            self.definition.name.clone(),
            self.definition.description.clone(),
            self.definition.input_schema.clone(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let result = self.client.call_tool(&self.definition.name, args).await?;

            // 拼接所有文本内容块。
            let text = result
                .content
                .iter()
                .filter_map(McpContent::as_text)
                .collect::<Vec<_>>()
                .join("\n");

            if result.is_error {
                // ⚠️ MCP 的「错误结果」**不是**传输错误 —— 它是工具正常执行后
                // 返回的业务错误。仍然要变成观察喂回模型。
                return Err(ToolError::execution(
                    self.definition.name.clone(),
                    if text.is_empty() {
                        "MCP 工具返回了错误（无详情）".to_string()
                    } else {
                        text
                    },
                ));
            }

            Ok(ToolOutcome::ok_with_data(
                self.definition.name.clone(),
                text,
                json!({ "mcp_endpoint": self.client.endpoint }),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json_response() {
        let response =
            parse_json_rpc_response(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#).unwrap();
        assert_eq!(response.id, Some(1));
        assert_eq!(response.result.unwrap()["ok"], json!(true));
    }

    #[test]
    fn parses_sse_response() {
        // Streamable HTTP 允许 SSE 格式返回单条响应。
        let text = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"v\":1}}\n\n";
        let response = parse_json_rpc_response(text).unwrap();
        assert_eq!(response.id, Some(2));
        assert_eq!(response.result.unwrap()["v"], json!(1));
    }

    #[test]
    fn rejects_garbage_response() {
        assert!(parse_json_rpc_response("not json at all").is_err());
        assert!(parse_json_rpc_response("").is_err());
    }

    #[test]
    fn parses_error_response() {
        let response = parse_json_rpc_response(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#,
        )
        .unwrap();
        assert!(response.error.is_some());
        assert!(response.error.unwrap().message.contains("Method not found"));
    }

    #[test]
    fn parses_tool_definition() {
        let definition: McpToolDefinition = serde_json::from_value(json!({
            "name": "get_weather",
            "description": "查天气",
            "inputSchema": {
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }
        }))
        .unwrap();

        assert_eq!(definition.name, "get_weather");
        assert_eq!(definition.description, "查天气");
        assert_eq!(definition.input_schema["required"], json!(["location"]));
    }

    #[test]
    fn tool_definition_without_schema_gets_empty_object() {
        let definition: McpToolDefinition =
            serde_json::from_value(json!({ "name": "noop" })).unwrap();
        assert_eq!(definition.input_schema["type"], json!("object"));
    }

    #[test]
    fn extracts_text_content() {
        let result: McpCallResult = serde_json::from_value(json!({
            "content": [
                { "type": "text", "text": "北京 20°C" },
                { "type": "text", "text": "湿度 45%" }
            ],
            "isError": false
        }))
        .unwrap();

        let text = result
            .content
            .iter()
            .filter_map(McpContent::as_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(text, "北京 20°C\n湿度 45%");
        assert!(!result.is_error);
    }

    #[test]
    fn client_starts_without_session() {
        let client = McpClient::new("http://127.0.0.1:9999/mcp");
        assert!(client.session_id().is_none());
        assert_eq!(client.tool_count(), 0);
        assert_eq!(client.endpoint(), "http://127.0.0.1:9999/mcp");
    }
}
