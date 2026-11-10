//! **结构化输出层** —— 让 Agent 的最终结论成为**可校验的 JSON**。
//!
//! # 为什么要有这一层
//!
//! 25 的硬约束里写着「数字必须来自工具结果，**不要心算**」——
//! 但那只是**提示词里的软约束**，模型可以不听。
//!
//! 15-agent-sql 实测过这个坑：模型**不查库就作答**，
//! 数字是编的，而且「答案看起来很专业但数字是编的」——
//! **极难从答案本身发现**。
//!
//! 这一层把它从软约束变成**机制化兜底**：
//!
//! 1. 让模型按 **JSON Schema** 输出「结论 + 数字断言清单」；
//! 2. **本地校验 Schema**（不合规 → 明确报错，不是「差不多就行」）；
//! 3. **接地校验** —— 每个数字断言都必须能在工具观察里找到出处。
//!
//! # ★ 为什么拆成两阶段（研究 → 成文）
//!
//! 与 19-agent-summary 同思路：
//! 一个提示词里同时要「查资料」和「写报告」时，
//! 模型会**先写**（写作的 token 收益更高）然后**编引用**。
//!
//! 所以：**先跑工具收集证据，再单独一次调用成文**。
//! 成文那次调用**不带工具** —— 带工具模型又想调，回到阶段 A。
//!
//! # 三条硬约定
//!
//! ## ① 本地 Schema 校验是**必需品**，不是锦上添花
//!
//! `json_mode`（DeepSeek 等）只保证「输出是 JSON」，
//! **不保证符合你的 Schema** ——
//! 字段名错了、类型错了、多了字段，它照样输出。
//! 所以解析后必须过一遍 [`crate::tooling::validate`]。
//!
//! ## ② 接地校验的**默认是「不通过」**
//!
//! 拿不到证据时 `grounded` 必须是 `false`，
//! 「默认接地」是最危险的默认值 —— 校验形同虚设。
//!
//! ## ③ 数字要按**数值**比，不是按字符串比
//!
//! 工具返回 `10`，模型写 `10.0` 或 `1,000` ——
//! 字符串比会误判成「没有出处」。

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::MemoryFact;
use crate::error::{AppError, AppResult};
use crate::llm::{LlmUsage, StructuredLlm};
use crate::model::Window;
use crate::tool::ToolRegistry;

/// 一条**数字断言** —— 模型声称「答案里的这个数字来自证据」。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumberClaim {
    /// 数字（字符串形式 —— 保留模型给的原样，便于回显给它）。
    pub value: String,

    /// 这个数字的上下文（它代表什么）。
    pub context: String,
}

/// 结构化结论。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredSummary {
    /// 自然语言结论。
    pub answer: String,

    /// 数字断言清单。
    ///
    /// ⚠️ **不是**「答案里的所有数字」——
    /// 是模型**声称**来自证据的那些数字。
    /// 接地校验就是逐条核对它们。
    pub numbers: Vec<NumberClaim>,
}

/// 接地校验的报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundReport {
    /// 是否全部数字断言都有出处。
    ///
    /// ⚠️ **默认 `false`** —— 见模块文档的约定 ②。
    pub grounded: bool,

    /// 校验了几条断言。
    pub checked: usize,

    /// **没有**出处的数字（模型编的那些）。
    pub unsupported: Vec<String>,
}

impl GroundReport {
    /// 全部有出处。
    pub fn all_supported(checked: usize) -> Self {
        Self {
            grounded: true,
            checked,
            unsupported: Vec::new(),
        }
    }

    /// 一句话摘要（给日志与 API 响应）。
    pub fn summary(&self) -> String {
        if self.grounded {
            format!("接地校验通过：{} 条数字断言全部有出处", self.checked)
        } else {
            format!(
                "接地校验失败：{} 条断言中 {} 条没有出处（{}）",
                self.checked,
                self.unsupported.len(),
                self.unsupported.join("、")
            )
        }
    }
}

/// 周报结论的 JSON Schema。
///
/// ## ⚠️ 为什么**不用** `strict` 的那种「全部字段必填」写法
///
/// OpenAI 的 strict 方言要求**每个** `properties` 都进 `required`
/// 且每层都要 `additionalProperties: false`。
/// 这里用**本地校验**兜底（[`crate::tooling::validate`]），
/// 所以 Schema 可以写得**语义清晰**（`numbers` 可空），
/// 而不必为了迁就某个 provider 把 `required` 撑满。
///
/// ⚠️ 需要 OpenAI 严格模式时，用
/// [`genai::chat::sanitize_json_schema`] 配
/// [`genai::chat::JsonSchemaDialect::OpenAiStrict`] 改写即可。
pub fn summary_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "结论正文。只使用工具结果里出现过的信息。"
            },
            "numbers": {
                "type": "array",
                "description": "答案里所有统计数字的清单，每条注明它代表什么。",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": "数字本身，例如 12 或 3.5。"
                        },
                        "context": {
                            "type": "string",
                            "description": "这个数字代表什么，例如 cache 模块的活动条数。"
                        }
                    },
                    "required": ["value", "context"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["answer", "numbers"],
        "additionalProperties": false
    })
}

/// 解析模型输出成 [`StructuredSummary`]。
///
/// ## ⚠️ 三步都要做，缺一不可
///
/// 1. **抽取 JSON** —— 模型常在 JSON 外面包 ` ```json ` 围栏，
///    前言里还可能有 `{周报}` 这类**括号配对但不是合法 JSON**的占位。
///    所以用 [`crate::agent::extract_json_object`]（逐字符扫配对括号后试解析）。
/// 2. **Schema 校验** —— `json_mode` 不保证符合 Schema。
/// 3. **反序列化** —— 类型对不上就明确报错。
///
/// 失败一律是 **502**（`BadGateway`）——
/// 模型输出不合规是**上游问题**，不是调用方的问题。
pub fn parse_summary(text: &str, schema: &Value) -> AppResult<StructuredSummary> {
    let value = crate::agent::extract_json_object(text).ok_or_else(|| {
        AppError::BadGateway(format!(
            "模型没有输出合法的 JSON 对象。收到的内容前 200 字：{}",
            text.chars().take(200).collect::<String>()
        ))
    })?;

    let validation = crate::tooling::validate(schema, &value);
    if !validation.is_ok() {
        return Err(AppError::BadGateway(format!(
            "模型输出不符合约定的 Schema：{}",
            validation.message()
        )));
    }

    serde_json::from_value::<StructuredSummary>(value)
        .map_err(|e| AppError::BadGateway(format!("模型输出的字段类型不对：{e}")))
}

/// ★ 两阶段结构化运行的报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredRun {
    /// 原始问题。
    pub question: String,

    /// 结构化结论。
    pub summary: StructuredSummary,

    /// 接地校验报告。
    pub grounding: GroundReport,

    /// ★ 收集到的证据（工具观察原文）。
    pub evidence: Vec<String>,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,

    /// token 用量（两阶段之和）。
    pub usage: LlmUsage,

    /// ★ 输出是否被截断。
    pub truncated: bool,
}

/// ★★ 「研究 → 成文」两阶段流程。
///
/// ## ⚠️ 为什么必须拆成两阶段
///
/// 19-agent-summary 实测的坑：
/// 一个提示词里同时要「查资料」和「写报告」时，
/// 模型会**先写**（写作的 token 收益更高）然后**编引用编号**。
///
/// 所以：
///
/// - **阶段 A（研究）**：挂工具，只收集证据
/// - **阶段 B（成文）**：**不带工具**，输入**只有证据**
///
/// ⚠️ 成文阶段带工具 → 模型又想调 → 回到阶段 A。
///
/// ## ⚠️ 成文阶段的输出**必须本地校验**
///
/// `json_mode`（DeepSeek）只保证「是 JSON」，
/// 不保证「符合 Schema」—— 所以要过 [`crate::tooling::validate`]。
///
/// ## ⚠️ 接地校验的证据池要**含结构化尾巴**
///
/// 工具返回的 `@@DATA:{...}` 里也有数字 ——
/// 模型报的 `total: 12` 要能在「共 12 条」**或** `{"total":12}` 里找到。
pub async fn run_structured(
    structured_llm: &dyn StructuredLlm,
    research_llm: std::sync::Arc<dyn crate::llm::FunctionLlm>,
    question: &str,
    facts: &[MemoryFact],
    registry: &ToolRegistry,
    window: &Window,
) -> AppResult<StructuredRun> {
    let start = std::time::Instant::now();

    // ── 阶段 A：研究（挂工具）───────────────────────────────
    let config = crate::agent_loop::AgentConfig::default();
    let researcher = crate::agent_loop::FunctionAgent::new(research_llm, registry.clone(), config);

    let research = researcher.run(question, facts, window).await?;

    // ★ 证据 = 所有工具观察的**原文**（含 `@@DATA:` 尾巴）。
    //
    // ⚠️ **不能剥掉** —— 剥掉就只剩给人看的文字，
    //    结构化数据里的数字会找不到出处，
    //    接地校验就会误判成「模型编的」。
    let evidence: Vec<String> = research
        .steps
        .iter()
        .flat_map(|s| s.tool_calls.iter())
        .map(|c| c.observation.clone())
        .collect();

    // ── 阶段 B：成文（不带工具）───────────────────────────────
    let schema = summary_schema();
    let prompt = compose_prompt(question, &evidence);

    let system = "你是一个严谨的活动记录分析助手。\n\
         你必须只输出**一个 JSON 对象**，不要输出任何解释文字、不要用代码围栏。\n\
         JSON 的结构固定为：\n\
         {\"answer\": \"结论正文\", \"numbers\": [{\"value\": \"数字\", \"context\": \"这个数字代表什么\"}]}\n\
         \n\
         硬约束：\n\
         - 数字必须**逐字**来自证据，**不要心算、不要四舍五入**\n\
         - 证据里没有的数字，一律不要写进答案\n\
         - 如果证据里没有相关数据，就在 answer 里如实说明「查不到数据」\n";

    let reply = structured_llm.structured(system, &prompt, &schema).await?;

    let summary = parse_summary(&reply.text, &schema)?;

    // ★ 接地校验 —— 拿**全部**证据（含结构化尾巴）当证据池。
    let grounding = ground_check(&summary, &evidence);

    if !grounding.grounded {
        // ⚠️ 降级必须 `warn!` —— 静默降级等于埋雷。
        agent_logging::warn!(
            unsupported = ?grounding.unsupported,
            "接地校验失败 —— 结论里有找不到出处的数字"
        );
    }

    Ok(StructuredRun {
        question: question.to_string(),
        summary,
        grounding,
        evidence,
        elapsed_ms: start.elapsed().as_millis() as u64,
        model: structured_llm.model_name().to_string(),
        usage: research.usage,
        truncated: research.truncated || reply.truncated,
    })
}

/// ★ 从一段文本里抽出**所有数字**。
///
/// ## ⚠️ 必须处理千分位与小数
///
/// 工具返回 `total: 1,000` 或 `12.5`，
/// 模型可能写成 `1000` 或 `12.50` —— 字符串比会误判。
///
/// 抽到的每个数字都**去掉千分位逗号**与尾随的点，
/// 保留原始形态供 [`numbers_equal`] 做数值比较。
pub fn extract_numbers(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_digit()
                    || chars[i] == ','
                    || chars[i] == '.'
                    || chars[i] == '_')
            {
                i += 1;
            }
            let raw: String = chars[start..i].iter().collect();
            let cleaned = raw.replace([',', '_'], "");
            // ⚠️ 尾随的点要去掉（"12." → "12"），
            //    否则 "12." 与 "12" 字符串不等、数值也解析失败。
            let cleaned = cleaned.trim_end_matches('.').to_string();
            if !cleaned.is_empty() {
                out.push(cleaned);
            }
        } else {
            i += 1;
        }
    }

    out
}

/// 两个数字是否「是同一个数」。
///
/// 先比字符串（快路径），再比**数值**（`10` vs `10.0`）。
pub fn numbers_equal(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => (x - y).abs() < 1e-9,
        _ => false,
    }
}

/// ★ 接地校验 —— 核对每条数字断言是否在证据里出现过。
///
/// ## ⚠️ 证据池来自**工具观察**（含 `@@DATA:` 结构化尾巴）
///
/// 因为结构化数据里的数字也是数字 ——
/// 模型报的 `total: 12` 在观察文本里是「共 12 条」，两者都要能对上。
///
/// ## ⚠️ 「没找到出处」**不等于**「模型编了」
///
/// 也可能是模型把 `12` 写成了「十二」。
/// 所以报告里给的是**待核对清单**，让调用方决定是拒绝还是追问。
pub fn ground_check(summary: &StructuredSummary, evidence: &[String]) -> GroundReport {
    let pool: Vec<String> = evidence.iter().flat_map(|e| extract_numbers(e)).collect();

    let mut unsupported = Vec::new();
    for claim in &summary.numbers {
        let found = pool.iter().any(|p| numbers_equal(p, &claim.value));
        if !found {
            unsupported.push(claim.value.clone());
        }
    }

    GroundReport {
        grounded: unsupported.is_empty() && !summary.numbers.is_empty(),
        checked: summary.numbers.len(),
        unsupported,
    }
}

/// 构造「研究 → 成文」两阶段的**成文提示**。
///
/// ## ⚠️ 证据是**唯一**的输入
///
/// 成文阶段**不带工具** —— 带工具模型又想调工具，回到阶段 A。
/// 同时提示词里**不给**工具描述，避免它「凭记忆」补充证据里没有的东西。
pub fn compose_prompt(question: &str, evidence: &[String]) -> String {
    let mut out = String::from(
        "请根据下面的证据回答问题。只输出一个 JSON 对象，不要输出任何解释文字。\n\
         JSON 结构：{\"answer\": \"结论正文\", \"numbers\": [{\"value\": \"数字\", \"context\": \"这个数字代表什么\"}]}\n\
         规则：\n\
         - answer 里出现的每一个统计数字，都必须在 numbers 里列出来\n\
         - numbers 里的每个数字都必须**逐字**来自证据，**不要心算、不要四舍五入**\n\
         - 证据里没有的数字，一律不要写进答案\n\n",
    );

    out.push_str("## 问题\n");
    out.push_str(question);
    out.push_str("\n\n## 证据\n");

    if evidence.is_empty() {
        out.push_str("（没有查到任何证据 —— 请在 answer 里如实说明「查不到数据」，不要编造。）\n");
    } else {
        for (i, e) in evidence.iter().enumerate() {
            out.push_str(&format!("[证据 {}]\n{}\n\n", i + 1, e));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_numbers_with_thousands_separators() {
        assert_eq!(extract_numbers("共 1,234 条"), vec!["1234"]);
        assert_eq!(extract_numbers("权重 12.5"), vec!["12.5"]);
        // ⚠️ 尾随的点要去掉。
        assert_eq!(extract_numbers("序号 12."), vec!["12"]);
        assert_eq!(extract_numbers("2023-11-14"), vec!["2023", "11", "14"]);
    }

    #[test]
    fn numbers_compare_by_value_not_by_string() {
        assert!(numbers_equal("10", "10"));
        assert!(numbers_equal("10", "10.0"));
        assert!(!numbers_equal("10", "11"));
    }

    #[test]
    fn parses_a_well_formed_summary() {
        let text = r#"前言 {"answer": "cache 模块共 12 条", "numbers": [{"value": "12", "context": "活动条数"}]}"#;
        let summary = parse_summary(text, &summary_schema()).unwrap();
        assert_eq!(summary.answer, "cache 模块共 12 条");
        assert_eq!(summary.numbers.len(), 1);
    }

    #[test]
    fn skips_placeholder_braces_before_the_real_json() {
        // ⚠️ `{周报}` 括号配对但不是合法 JSON —— 不能当成结果。
        let text = "这是 {周报} ：{\"answer\": \"ok\", \"numbers\": []}";
        let summary = parse_summary(text, &summary_schema()).unwrap();
        assert_eq!(summary.answer, "ok");
    }

    #[test]
    fn rejects_output_that_violates_the_schema() {
        // numbers 里的项缺 `context`。
        let text = r#"{"answer": "ok", "numbers": [{"value": "12"}]}"#;
        let err = parse_summary(text, &summary_schema()).unwrap_err();
        // ★ 模型输出不合规是**上游问题** → 502，不是 400。
        assert_eq!(err.status_code(), axum::http::StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn rejects_non_json_output() {
        let err = parse_summary("今天天气不错", &summary_schema()).unwrap_err();
        assert_eq!(err.status_code(), axum::http::StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn grounding_passes_when_the_number_appears_in_evidence() {
        let summary = StructuredSummary {
            answer: "cache 模块共 12 条".into(),
            numbers: vec![NumberClaim {
                value: "12".into(),
                context: "活动条数".into(),
            }],
        };
        let report = ground_check(&summary, &["模块 cache：共 12 条活动".to_string()]);
        assert!(report.grounded);
        assert_eq!(report.checked, 1);
        assert!(report.unsupported.is_empty());
    }

    #[test]
    fn grounding_fails_when_the_number_is_made_up() {
        let summary = StructuredSummary {
            answer: "cache 模块共 99 条".into(),
            numbers: vec![NumberClaim {
                value: "99".into(),
                context: "活动条数".into(),
            }],
        };
        let report = ground_check(&summary, &["模块 cache：共 12 条活动".to_string()]);
        assert!(!report.grounded);
        assert_eq!(report.unsupported, vec!["99".to_string()]);
    }

    #[test]
    fn grounding_is_false_when_there_are_no_claims_at_all() {
        // ⚠️ 「没有断言」不等于「全部有出处」——
        //    默认接地是最危险的默认值。
        let summary = StructuredSummary {
            answer: "没问题".into(),
            numbers: Vec::new(),
        };
        let report = ground_check(&summary, &["模块 cache：共 12 条活动".to_string()]);
        assert!(!report.grounded);
        assert_eq!(report.checked, 0);
    }

    #[test]
    fn grounding_matches_numbers_inside_the_structured_tail() {
        let summary = StructuredSummary {
            answer: "共 12 条".into(),
            numbers: vec![NumberClaim {
                value: "12".into(),
                context: "总数".into(),
            }],
        };
        let evidence = vec!["模块 cache：共 12 条活动\n@@DATA:{\"total\":12}".to_string()];
        assert!(ground_check(&summary, &evidence).grounded);
    }

    #[test]
    fn prompt_marks_the_empty_evidence_case() {
        let prompt = compose_prompt("cache 模块做了什么？", &[]);
        assert!(prompt.contains("查不到数据"));
        assert!(prompt.contains("cache 模块做了什么？"));
    }

    #[test]
    fn prompt_lists_the_evidence() {
        let prompt = compose_prompt("问题", &["证据一".to_string(), "证据二".to_string()]);
        assert!(prompt.contains("[证据 1]"));
        assert!(prompt.contains("[证据 2]"));
    }

    // ── 两阶段流程 ────────────────────────────────────────────

    fn window() -> crate::model::Window {
        crate::model::Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400)
    }

    fn registry() -> crate::tool::ToolRegistry {
        use crate::tool::{CalculateTool, ModuleStatsTool};
        use std::sync::Arc;

        let activities = Arc::new(
            (0..20)
                .map(|i| crate::model::Activity {
                    id: format!("a-{i:04}"),
                    ts: 1_700_000_000 + (i as i64 % 7) * 86_400,
                    kind: ["feature", "fix"][i % 2].to_string(),
                    author: ["alice", "bob"][i % 2].to_string(),
                    scope: ["cache", "rag"][i % 2].to_string(),
                    title: format!("任务 {i}"),
                    weight: (i % 10) as i32,
                    risk: false,
                })
                .collect(),
        );

        crate::tool::ToolRegistry::new(vec![
            Arc::new(ModuleStatsTool::new(activities)),
            Arc::new(CalculateTool),
        ])
    }

    #[tokio::test]
    async fn two_phase_run_grounds_the_numbers_against_the_evidence() {
        use crate::llm::{ScriptedFunctionLlm, ScriptedStep, ScriptedStructuredLlm};
        use std::sync::Arc;

        // 阶段 A：研究 —— 调一次工具。
        let research = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::call("c1", "module_stats", serde_json::json!({"scope": "cache"})),
            ScriptedStep::text("查完了。"),
        ]));

        // 阶段 B：成文 —— 输出一个 JSON。
        let structured = ScriptedStructuredLlm::always(
            r#"{"answer": "cache 模块共 10 条活动", "numbers": [{"value": "10", "context": "活动条数"}]}"#,
        );

        let report = run_structured(
            &structured,
            research,
            "cache 模块做了什么？",
            &[],
            &registry(),
            &window(),
        )
        .await
        .unwrap();

        assert_eq!(report.summary.answer, "cache 模块共 10 条活动");
        // ★ 接地校验通过 —— 「10」能在工具观察里找到。
        assert!(report.grounding.grounded, "{}", report.grounding.summary());
        assert_eq!(report.grounding.checked, 1);
        assert!(!report.evidence.is_empty());
    }

    #[tokio::test]
    async fn two_phase_run_flags_made_up_numbers() {
        use crate::llm::{ScriptedFunctionLlm, ScriptedStep, ScriptedStructuredLlm};
        use std::sync::Arc;

        let research = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::call("c1", "module_stats", serde_json::json!({"scope": "cache"})),
            ScriptedStep::text("查完了。"),
        ]));

        // ⚠️ 模型编了一个「999」。
        let structured = ScriptedStructuredLlm::always(
            r#"{"answer": "cache 模块共 999 条活动", "numbers": [{"value": "999", "context": "活动条数"}]}"#,
        );

        let report = run_structured(
            &structured,
            research,
            "cache 模块做了什么？",
            &[],
            &registry(),
            &window(),
        )
        .await
        .unwrap();

        // ★ 接地校验**必须**失败 —— 这正是这一层的价值。
        assert!(!report.grounding.grounded);
        assert_eq!(report.grounding.unsupported, vec!["999".to_string()]);
    }

    #[tokio::test]
    async fn two_phase_run_rejects_output_that_violates_the_schema() {
        use crate::llm::{ScriptedFunctionLlm, ScriptedStep, ScriptedStructuredLlm};
        use std::sync::Arc;

        let research = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::call("c1", "module_stats", serde_json::json!({"scope": "cache"})),
            ScriptedStep::text("查完了。"),
        ]));

        // ⚠️ `numbers` 里的项缺 `context` —— Schema 不合规。
        let structured =
            ScriptedStructuredLlm::always(r#"{"answer": "ok", "numbers": [{"value": "10"}]}"#);

        let err = run_structured(
            &structured,
            research,
            "cache 模块做了什么？",
            &[],
            &registry(),
            &window(),
        )
        .await
        .unwrap_err();

        // ★ 模型输出不合规是**上游问题** → 502。
        assert_eq!(err.status_code(), axum::http::StatusCode::BAD_GATEWAY);
    }
}
