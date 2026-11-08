//! 自主 Agent 层 —— 让 LLM 自己决定「要不要检索、检索什么」。
//!
//! ## Naive RAG 和 Agent 的本质区别
//!
//! Naive RAG 的流程是**写死的**：
//!
//! ```text
//! 问题 → 一定检索 → 一定拼 top-k → 一定生成
//! ```
//!
//! Agent 的流程是**模型决定的**：
//!
//! ```text
//! 问题 → 模型判断「我需要查资料吗？」
//!          ├─ 不需要 → 直接回答
//!          └─ 需要   → 模型调用 search 工具（自己写查询词）
//!                     ↓
//!                  结果回填 → 模型再判断
//!                     ├─ 还需要 → 继续调工具（换查询词）
//!                     └─ 够了   → 给出最终答案
//! ```
//!
//! 关键差异：**谁来决定检索的时机和查询词**。
//!
//! - Naive：检索器决定「用原问题去检索」，用户问「它多少钱」就拿「它多少钱」去搜
//! - Agent：模型决定「我该搜什么」，可以自己改写成「价格 费用」
//!
//! ## 为什么这能提升效果
//!
//! 三个具体的失效场景，Agent 都能救：
//!
//! 1. **查询词不匹配**：用户问「怎么防止它崩掉」，文档里写的是「容错与降级」。
//!    Naive 用原问题检索 → 召不回；Agent 改写成「容错 降级」→ 召得回。
//! 2. **多跳问题**：「A 的作者还写了什么」需要先查 A，再查作者的作品列表。
//!    Naive 一次检索做不到；Agent 可以调两次工具。
//! 3. **不需要检索**：问「你好」或「RAG 是什么」这种通用知识问题，
//!    检索只会引入噪声。Agent 可以直接答。
//!
//! ## 工具调用的循环
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ 1. 发消息（含工具声明）                        │
//! ├──────────────────────────────────────────────┤
//! │ 2. 模型返回 tool_calls                        │
//! ├──────────────────────────────────────────────┤
//! │ 3. 执行工具，把结果作为 tool 消息回填           │
//! ├──────────────────────────────────────────────┤
//! │ 4. 回到 1，直到模型不再要求调工具              │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! ⚠️ **必须有循环上限**。模型可能反复调同一个工具（工具描述写得不好时很常见），
//! 没有上限就会变成烧钱的死循环。本项目用 `max_iterations` 兜底。

use agent_logging::{debug, info, warn};
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, Tool, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::GenerationConfig;
use crate::error::{RagError, RagResult};
use crate::retrieval::Retriever;

/// 检索工具的名字。模型通过这个名字调用检索。
pub const SEARCH_TOOL_NAME: &str = "search_knowledge_base";

/// Agent 的 system 提示词。
///
/// 相比 Naive RAG 的 system 提示，多了「你可以调工具」的说明，
/// 以及「一次回答里检索次数要节制」这类行为约束。
pub const AGENT_SYSTEM_PROMPT: &str = "\
你是一个技术文档问答助手，可以查询一个内部知识库来回答问题。

可用工具：
- search_knowledge_base(query, top_k)：在知识库里做语义检索，返回最相关的文档片段。

工作方式：
1. 先判断问题是否需要查知识库。需要就用工具检索，不需要就直接回答。
2. 检索时，把问题改写成更适合检索的关键词短语（用用户的原话往往召不回）。
3. 如果一次检索结果不足以回答，可以换查询词再检索一次，但最多两次。
4. 回答时只依据检索到的资料，用 [1]、[2] 标注引用编号。
5. 知识库里确实没有答案时，直接说明没有找到相关资料。";

/// 工具调用记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// 第几轮的工具调用（从 1 开始）。
    pub iteration: usize,
    /// 工具名。
    pub tool: String,
    /// 模型给的参数。
    pub arguments: serde_json::Value,
    /// 工具返回的内容（截断后）。
    pub result: String,
}

/// Agent 的执行结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// 最终答案。
    pub answer: String,

    /// 工具调用记录（按时间顺序）。
    pub tool_calls: Vec<ToolCallRecord>,

    /// 迭代轮数。
    pub iterations: usize,

    /// 总 token 用量。
    pub usage: crate::generation::TokenUsage,

    /// 使用的模型名。
    pub model: String,

    /// 是否因为达到迭代上限而中止。
    ///
    /// `true` 表示模型还想继续调工具但被我们掐断了 ——
    /// 这通常说明工具描述不够清楚，或者问题太复杂。
    pub hit_max_iterations: bool,
}

/// Agent 的配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 生成配置（模型、温度等）。
    pub generation: GenerationConfig,

    /// 最大迭代轮数。
    ///
    /// 每轮迭代 = 一次「发消息 + 可能的一次工具调用」。
    /// 默认 5，够查两次资料 + 给出答案。
    pub max_iterations: usize,

    /// 单次检索返回的块数上限。
    pub default_top_k: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            max_iterations: 5,
            default_top_k: 4,
        }
    }
}

/// 检索工具的 JSON Schema。
pub fn search_tool_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索用的关键词或短语。应该用文档里可能出现的词，而不是用户的口语原话。"
            },
            "top_k": {
                "type": "integer",
                "description": "返回的片段数量，默认 4。"
            }
        },
        "required": ["query"]
    })
}

/// 自主 Agent。
///
/// 持有检索器和生成配置。
pub struct Agent {
    retriever: std::sync::Arc<Retriever>,
    config: AgentConfig,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("config", &self.config)
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    pub fn new(retriever: std::sync::Arc<Retriever>, config: AgentConfig) -> Self {
        Self { retriever, config }
    }

    /// 运行 Agent 处理一个问题。
    pub async fn run(&self, question: &str) -> RagResult<AgentResult> {
        let question = question.trim();
        if question.is_empty() {
            warn!("Agent 问题为空");
            return Err(RagError::BadRequest("问题不能为空".to_string()));
        }

        info!(question = %question, max_iterations = self.config.max_iterations, "Agent 开始运行");

        let client = genai::Client::builder().build();

        // 工具声明 —— 模型靠它知道「能调什么、参数怎么填」。
        let tools = vec![
            Tool::new(SEARCH_TOOL_NAME)
                .with_description("在内部知识库里做语义检索，返回最相关的文档片段")
                .with_schema(search_tool_schema()),
        ];

        let mut request = ChatRequest::new(vec![
            ChatMessage::system(AGENT_SYSTEM_PROMPT),
            ChatMessage::user(question),
        ])
        .with_tools(tools);

        let options = ChatOptions::default()
            .with_temperature(self.config.generation.temperature)
            .with_max_tokens(self.config.generation.max_tokens);

        let mut tool_calls: Vec<ToolCallRecord> = Vec::new();
        let mut total_usage = crate::generation::TokenUsage::default();

        // ── 工具调用循环 ────────────────────────────────────────
        for iteration in 1..=self.config.max_iterations {
            debug!(iteration, "Agent 迭代：发起模型请求");
            let response = client
                .exec_chat(
                    &self.config.generation.model,
                    request.clone(),
                    Some(&options),
                )
                .await
                .map_err(|e| {
                    warn!(
                        iteration,
                        model = %self.config.generation.model,
                        error = %e,
                        "Agent 模型调用失败"
                    );
                    RagError::Generation(format!("调用模型失败: {e}"))
                })?;

            total_usage.prompt_tokens += response.usage.prompt_tokens.unwrap_or(0) as u32;
            total_usage.completion_tokens += response.usage.completion_tokens.unwrap_or(0) as u32;
            total_usage.total_tokens += response.usage.total_tokens.unwrap_or(0) as u32;

            let calls: Vec<ToolCall> = response.tool_calls().into_iter().cloned().collect();

            // 模型不再要求调工具 → 它给出了最终答案，循环结束。
            if calls.is_empty() {
                let answer = response.first_text().unwrap_or_default().to_string();
                info!(
                    iterations = iteration,
                    tool_call_count = tool_calls.len(),
                    answer_chars = answer.chars().count(),
                    "Agent 给出最终答案"
                );
                return Ok(AgentResult {
                    answer,
                    tool_calls,
                    iterations: iteration,
                    usage: total_usage,
                    model: self.config.generation.model.clone(),
                    hit_max_iterations: false,
                });
            }

            // 执行工具。
            let mut responses: Vec<ToolResponse> = Vec::new();
            for call in &calls {
                let (result, record) = self.run_search_tool(call, iteration).await?;
                tool_calls.push(record);
                responses.push(ToolResponse::new(call.call_id.clone(), result));
            }

            // ⚠️ 回填顺序很重要：先是 assistant 的 tool_calls，再是 tool 结果。
            //    少了前者，模型不知道自己「调过什么工具、参数是什么」，
            //    就没法把结果和调用对应起来。
            request = request.append_message(calls).append_message(responses);
        }

        // 循环跑完还没拿到答案 —— 最后再问一次，让它基于已有信息作答。
        warn!(
            max_iterations = self.config.max_iterations,
            tool_call_count = tool_calls.len(),
            "Agent 达到迭代上限，做最后一次收尾生成"
        );
        let response = client
            .exec_chat(&self.config.generation.model, request, Some(&options))
            .await
            .map_err(|e| {
                warn!(error = %e, "Agent 收尾生成失败");
                RagError::Generation(format!("调用模型失败: {e}"))
            })?;

        total_usage.prompt_tokens += response.usage.prompt_tokens.unwrap_or(0) as u32;
        total_usage.completion_tokens += response.usage.completion_tokens.unwrap_or(0) as u32;
        total_usage.total_tokens += response.usage.total_tokens.unwrap_or(0) as u32;

        Ok(AgentResult {
            answer: response.first_text().unwrap_or_default().to_string(),
            tool_calls,
            iterations: self.config.max_iterations,
            usage: total_usage,
            model: self.config.generation.model.clone(),
            hit_max_iterations: true,
        })
    }

    /// 执行检索工具。
    ///
    /// 返回 `(给模型看的结果文本, 记录)`。
    async fn run_search_tool(
        &self,
        call: &ToolCall,
        iteration: usize,
    ) -> RagResult<(String, ToolCallRecord)> {
        if call.fn_name != SEARCH_TOOL_NAME {
            // 未知工具 —— 回一个错误说明，让模型自己纠正。
            warn!(tool = %call.fn_name, iteration, "模型调用了未知工具");
            return Ok((
                format!("错误：未知工具 `{}`", call.fn_name),
                ToolCallRecord {
                    iteration,
                    tool: call.fn_name.clone(),
                    arguments: call.fn_arguments.clone(),
                    result: "未知工具".to_string(),
                },
            ));
        }

        let query = call
            .fn_arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        let top_k = call
            .fn_arguments
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.config.default_top_k)
            .clamp(1, 20);

        if query.is_empty() {
            warn!(iteration, "检索工具缺少 query 参数");
            return Ok((
                "错误：缺少 query 参数".to_string(),
                ToolCallRecord {
                    iteration,
                    tool: call.fn_name.clone(),
                    arguments: call.fn_arguments.clone(),
                    result: "缺少 query 参数".to_string(),
                },
            ));
        }

        info!(iteration, query = %query, top_k, "Agent 调用检索工具");

        // 检索时用模型给的 top_k 覆盖默认配置。
        let result = self.retriever.retrieve_with_top_k(&query, top_k).await?;

        let text = if result.is_empty() {
            warn!(query = %query, "检索没有命中任何块");
            "没有检索到相关资料。".to_string()
        } else {
            result.to_context()
        };

        let record = ToolCallRecord {
            iteration,
            tool: call.fn_name.clone(),
            arguments: call.fn_arguments.clone(),
            result: text.clone(),
        };

        Ok((text, record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_tool_schema_requires_query() {
        let schema = search_tool_schema();
        assert_eq!(schema["required"], json!(["query"]));
        assert!(schema["properties"]["query"]["type"] == "string");
    }

    #[test]
    fn agent_config_has_bounded_iterations() {
        let config = AgentConfig::default();
        assert!(config.max_iterations > 0);
        assert!(
            config.max_iterations <= 20,
            "迭代上限必须有限，否则会死循环烧钱"
        );
    }
}
