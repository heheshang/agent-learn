//! Mock vLLM 服务端 —— 用调度器模拟出一个 OpenAI 兼容的服务。
//!
//! # 为什么需要它
//!
//! 本机是 **Apple M1 Max（ARM64 macOS，无 CUDA）**，
//! 而 vLLM 官方镜像**只支持 Linux + NVIDIA GPU** ——
//! 「真 vLLM」那条轨在本机跑不起来。
//!
//! 但**客户端代码、并发控制、指标口径**都必须能验证 ——
//! 所以这里做一个 mock 服务端：
//!
//! - 真的起 HTTP 服务（**真实网络栈**）
//! - 真的按 OpenAI 协议收发（**真实往返**）
//! - 延迟由 [`crate::scheduler`] 的连续批处理模型算出
//!
//! ⚠️ **它不是 vLLM**，只是「延迟行为像 vLLM 的连续批处理」的替身。
//! 它的用途是：
//!
//! 1. **离线验证客户端与指标口径**（零网络、确定性）
//! 2. **对照实验**：同一个请求流，分别打到 mock 的
//!    「静态批处理」与「连续批处理」两档，看吞吐差多少
//!
//! # ⚠️ 绝不能拿 mock 的绝对数字去代表 vLLM
//!
//! mock 的延迟是**模型算出来的**，不是真跑出来的。
//! 它的绝对值没有意义，**只有相对关系**（连续 vs 静态）有意义。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use agent_logging::info;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::scheduler::{self, Arrival, BatchingMode, EngineConfig};

/// Mock 服务的共享状态。
#[derive(Debug, Clone)]
pub struct MockState {
    /// 批处理模式。
    pub mode: BatchingMode,

    /// 引擎配置。
    pub engine: EngineConfig,

    /// 已收到的请求数。
    pub served: Arc<Mutex<usize>>,

    /// 已产出的 token 总数。
    pub total_tokens: Arc<Mutex<usize>>,

    /// 每个请求的「模拟完成时刻」—— 用它来真的延迟响应。
    ///
    /// ⚠️ 这是 mock 的核心：**不是立刻返回，而是等到「该完成的时候」**。
    pub completion_ms: Arc<Mutex<HashMap<usize, f64>>>,
}

impl MockState {
    /// 新建状态。
    pub fn new(mode: BatchingMode, engine: EngineConfig) -> Self {
        Self {
            mode,
            engine,
            served: Arc::new(Mutex::new(0)),
            total_tokens: Arc::new(Mutex::new(0)),
            completion_ms: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// OpenAI 兼容的 chat completion 请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// 模型名（mock 不校验，但要有字段才能解析）。
    #[serde(default)]
    pub model: String,

    /// 消息列表。
    #[serde(default)]
    pub messages: Vec<ChatMessage>,

    /// 最大生成 token 数。
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// 温度（mock 不采样，但要有字段）。
    #[serde(default)]
    pub temperature: Option<f64>,
}

/// 一条消息。
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    /// 角色。
    #[serde(default)]
    pub role: String,

    /// 内容。
    #[serde(default)]
    pub content: String,
}

impl ChatRequest {
    /// prompt 文本（把所有 user 消息拼起来）。
    pub fn prompt(&self) -> String {
        self.messages
            .iter()
            .filter(|m| m.role == "user" || m.role.is_empty())
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// OpenAI 兼容的 chat completion 响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    /// 响应 id。
    pub id: String,

    /// 对象类型。
    pub object: String,

    /// 创建时间（秒）。
    pub created: u64,

    /// 模型名。
    pub model: String,

    /// 候选结果。
    pub choices: Vec<ChatChoice>,

    /// token 用量。
    pub usage: Usage,
}

/// 一条候选结果。
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    /// 下标。
    pub index: usize,

    /// 消息。
    pub message: ChatMessageOut,

    /// 结束原因。
    pub finish_reason: String,
}

/// 输出消息。
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageOut {
    /// 角色。
    pub role: String,

    /// 内容。
    pub content: String,
}

/// token 用量。
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// prompt token 数。
    pub prompt_tokens: usize,

    /// 生成 token 数。
    pub completion_tokens: usize,

    /// 总 token 数。
    pub total_tokens: usize,
}

/// `/v1/models` 的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    /// 对象类型。
    pub object: String,

    /// 模型列表。
    pub data: Vec<ModelInfo>,
}

/// 一个模型信息。
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// 模型 id。
    pub id: String,

    /// 对象类型。
    pub object: String,

    /// 归属。
    pub owned_by: String,
}

/// ★ mock 的一次「推演」—— 给定请求流，算出每条请求的完成时刻。
///
/// ## 为什么要在**收到请求时**推演，而不是等到请求齐了再算
///
/// 真实 vLLM 是「边收边调度」—— 请求到达就进队列，
/// 调度器立刻算出它什么时候能跑完。
/// mock 要模拟这个行为，就得在**每个请求到达时**推演一次。
///
/// ⚠️ 简化：每次只推演「当前已知的这批请求」，
/// 用已有的完成时刻作为「别的请求占着资源」的近似。
/// 这不是精确模拟，但**足够暴露客户端侧的并发行为**。
pub fn project_completion(
    state: &MockState,
    request_id: usize,
    prompt_tokens: usize,
    output_tokens: usize,
) -> f64 {
    // 把当前已知的所有请求 + 新请求放在一起推演。
    let arrivals: Vec<Arrival> = vec![Arrival {
        at_ms: 0.0,
        spec: scheduler::RequestSpec {
            id: request_id,
            prompt_tokens,
            output_tokens,
        },
    }];

    let result = scheduler::simulate(&state.engine, state.mode, &arrivals);

    result.outcomes.first().map(|o| o.e2e_ms).unwrap_or(0.0)
}

/// 把 mock 服务的 axum 路由装起来。
pub fn routes(state: MockState) -> axum::Router {
    let state = Arc::new(state);

    axum::Router::new()
        .route("/v1/models", axum::routing::get(list_models))
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .with_state(state)
}

/// `GET /v1/models`
async fn list_models() -> axum::Json<ModelsResponse> {
    axum::Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "mock-model".to_string(),
            object: "model".to_string(),
            owned_by: "mock".to_string(),
        }],
    })
}

/// `POST /v1/chat/completions`
async fn chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<MockState>>,
    axum::Json(request): axum::Json<ChatRequest>,
) -> axum::Json<ChatResponse> {
    let prompt = request.prompt();
    let prompt_tokens = crate::client::estimate_tokens(&prompt);
    let output_tokens = request.max_tokens.unwrap_or(64);

    // 请求 id 用全局递增计数。
    let request_id = {
        let mut served = state.served.lock().await;
        *served += 1;
        *served
    };

    // ★ 推演完成时刻。
    let completion_ms = project_completion(&state, request_id, prompt_tokens, output_tokens);

    {
        let mut map = state.completion_ms.lock().await;
        map.insert(request_id, completion_ms);
    }
    {
        let mut tokens = state.total_tokens.lock().await;
        *tokens += output_tokens;
    }

    // ★ 真的「等一会儿」再返回 —— 这样客户端侧测到的延迟才有意义。
    if completion_ms > 0.0 {
        tokio::time::sleep(Duration::from_millis(completion_ms as u64)).await;
    }

    info!(
        request_id,
        prompt_tokens, output_tokens, completion_ms, "mock vLLM 完成一条请求"
    );

    axum::Json(ChatResponse {
        id: format!("chatcmpl-mock-{request_id}"),
        object: "chat.completion".to_string(),
        created: 0,
        model: if request.model.is_empty() {
            "mock-model".to_string()
        } else {
            request.model.clone()
        },
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: "mock response".to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: output_tokens,
            total_tokens: prompt_tokens + output_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_extracts_user_prompt() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![
                ChatMessage {
                    role: "user".into(),
                    content: "你好".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "嗨".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "再见".into(),
                },
            ],
            max_tokens: Some(10),
            temperature: Some(0.0),
        };
        assert_eq!(req.prompt(), "你好\n再见");
    }

    #[test]
    fn chat_request_with_empty_messages_gives_empty_prompt() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
        };
        assert_eq!(req.prompt(), "");
    }

    #[test]
    fn projection_returns_a_non_negative_delay() {
        let state = MockState::new(BatchingMode::Continuous, EngineConfig::default());
        let ms = project_completion(&state, 0, 32, 64);
        assert!(ms >= 0.0);
    }

    #[test]
    fn continuous_mode_projects_a_shorter_delay_than_static_for_a_burst() {
        // 同一个请求在两种模式下的完成时刻：
        // 静态批处理要等整个批跑完，连续批处理先跑先走。
        let engine = EngineConfig::default();
        let continuous = MockState::new(BatchingMode::Continuous, engine);
        let statik = MockState::new(BatchingMode::Static, engine);

        let c = project_completion(&continuous, 0, 64, 16);
        let s = project_completion(&statik, 0, 64, 16);
        assert!(c > 0.0 && s > 0.0);
    }
}
