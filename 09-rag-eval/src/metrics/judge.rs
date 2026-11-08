//! LLM-as-a-judge 指标 —— RAGAs 的核心，也是本项目「答案侧」的评估。
//!
//! ## 四个指标
//!
//! | 指标 | 问的问题 | 对应 RAGAs |
//! |---|---|---|
//! | [`Faithfulness`] | 答案里的每句话都有据可依吗？ | `faithfulness` |
//! | [`AnswerRelevancy`] | 答案和**问题**相关吗？ | `answer_relevancy` |
//! | [`ContextRelevancy`] | 检索到的上下文和问题相关吗？ | `context_relevancy` |
//! | [`AnswerCorrectness`] | 答案和标准答案说的是一回事吗？ | `answer_correctness` |
//!
//! ## 为什么这四个指标问的是四个不同的问题
//!
//! 一个 RAG 系统的失败有四种模式，**每种对应一个指标**：
//!
//! ```text
//! 问题 ──> 检索 ──> 上下文 ──> 生成 ──> 答案
//!           │                   │
//!    context_relevancy         answer_relevancy
//!    （检错了）                （答偏了）
//!
//!         faithfulness（编造了）    answer_correctness（答错了）
//! ```
//!
//! ## judge 的三个可靠性前提
//!
//! LLM-as-a-judge 最大的问题是**不可靠**。本项目用三种手段把它压住：
//!
//! 1. **结构化输出** —— judge 必须返回 JSON（分数 + 理由），
//!    而不是自由文本。解析失败 = 这次评估失败（可降级），不是静默 0 分。
//! 2. **低温度** —— judge 的温度必须低（0.0），否则同一份输入会得到不同分数，
//!    「优化后涨了 0.05」就无从谈起。
//! 3. **要求给出理由** —— 没有理由的分数不可解释，无法定位问题。

use agent_logging::warn;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{EvalError, EvalResult};
use crate::metrics::types::{EvalContext, Metric, MetricFuture, MetricScore};

/// judge 的输出格式。
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Judgement {
    /// 分数（`[0, 1]`）。
    score: f32,

    /// 理由。
    #[serde(default)]
    reason: String,
}

/// judge 的重试次数。
///
/// 3 次足够 —— 瞬时故障一次就恢复，格式问题重试也没用。
const JUDGE_RETRY_ATTEMPTS: usize = 3;

/// judge 提示词的公共部分。
const JUDGE_INSTRUCTION: &str = "\
你是一个严格公正的评估员。请根据给定的标准评估被评估对象。

规则：
1. 只输出一个 JSON 对象，不要有任何其它文字、不要用 markdown 代码块包裹。
2. JSON 格式为：{\"score\": 0.0~1.0 的数字, \"reason\": \"评估理由\"}
3. score 必须在 0 到 1 之间，保留两位小数。
4. reason 必须指出具体的问题所在（引用被评估对象中的原句），不要泛泛而谈。
5. 遇到任何不确定的情况，给较低的分数 —— 宁可保守，不要宽容。";

/// 解析 judge 的 JSON 输出。
///
/// ## 为什么必须容错
///
/// LLM 的 JSON 输出有各种「小毛病」：
///
/// - 裹在 ```` ```json ```` 代码块里
/// - 前后有解释文字
/// - 用了单引号
/// - 分数写成了 `"0.85"`（字符串）
///
/// 这些**不是错误** —— 是模型的自然输出习惯。
/// 直接 `serde_json::from_str` 会失败，然后整次评估挂掉 —— 太脆了。
///
/// 所以这里做三层兜底：
///
/// 1. 剥掉 markdown 代码块
/// 2. 提取最外层的 `{...}`
/// 3. 数值字段容错（字符串 → 数字）
fn parse_judgement(raw: &str) -> EvalResult<Judgement> {
    let cleaned = strip_code_block(raw);
    let json_part = extract_json_object(cleaned).ok_or_else(|| missing_json_error(raw))?;

    serde_json::from_str::<Judgement>(&json_part).map_err(|e| {
        EvalError::Judging(format!(
            "解析 judge 的 JSON 失败（{e}）: {}",
            truncate(&json_part)
        ))
    })
}

/// 剥掉 markdown 代码块包裹。
fn strip_code_block(text: &str) -> &str {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }

    // 跳过开头的 ``` 和可能的语言标注。
    let after_open = match trimmed.find('\n') {
        Some(i) => &trimmed[i + 1..],
        None => return trimmed,
    };

    match after_open.rfind("```") {
        Some(i) => after_open[..i].trim(),
        None => after_open.trim(),
    }
}

/// 提取最外层的 `{...}`。
///
/// 从第一个 `{` 开始，找到**配对**的 `}` —— 括号计数，跳过字符串字面量里的括号。
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for i in start..bytes.len() {
        let c = bytes[i];

        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }

        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

/// 截断长文本（用于错误消息）。
fn truncate(text: &str) -> String {
    text.chars().take(200).collect()
}

/// 单句判定结果。
#[derive(Debug, Clone, serde::Deserialize)]
struct SentenceVerdict {
    /// 该句的分数。
    #[serde(default)]
    score: f32,

    /// 该句的理由（没有问题时为空）。
    #[serde(default)]
    reason: String,
}

/// 批量判定的响应体。
#[derive(Debug, Clone, serde::Deserialize)]
struct SentenceVerdicts {
    /// 每句一个分数。
    #[serde(default)]
    scores: Vec<f32>,

    /// 每句一个理由。
    #[serde(default)]
    reasons: Vec<String>,
}

/// 构造「judge 输出里找不到 JSON」的错误信息。
///
/// ## 为什么不能只说「找不到 JSON」
///
/// 最常见的失败原因其实**不是**模型格式乱，而是**输出被截断**：
///
/// ```text
/// {"scores": [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0], "reasons": ["……"   ← 截断在这里
/// ```
///
/// 截断的 JSON **有 `{` 没有 `}`** —— 括号配对失败，看起来就像「没有 JSON」，
/// 但真正的原因是 `max_tokens` 不够（或者推理 token 吃光了预算）。
///
/// 所以这里要区分两种情况：
///
/// - **完全没有 `{`** —— 模型压根没输出 JSON（提示词被无视了）
/// - **有 `{` 但括号没配平** —— 输出被截断
///
/// 后者要明确提示「调大 max_tokens」，否则排查的人会一直去改提示词。
fn missing_json_error(raw: &str) -> EvalError {
    let has_open = raw.contains('{');
    let hint = if has_open {
        "（JSON 被截断 —— 尝试调大 EVAL_JUDGE_MAX_TOKENS 或关闭思考模式）"
    } else {
        "（模型没有输出 JSON —— 检查 judge 提示词）"
    };

    EvalError::Judging(format!(
        "judge 输出里找不到完整 JSON{hint}: {}",
        truncate(raw)
    ))
}

/// 解析批量判定结果。
///
/// ## 为什么必须容错「数量对不上」
///
/// LLM 返回的数组长度可能和输入的句子数不一致：
///
/// - 少给一个（漏判了）
/// - 多给一个（多判了）
/// - 空数组（完全没干活）
///
/// 这三种情况**不能**直接崩掉 —— 那会让整个评估挂掉。
///
/// 处理策略：**按能拿到的分数算平均值**。
///
/// - 给少了 → 缺的按「无依据」（0 分）补齐 —— 保守
/// - 给多了 → 只用前 N 个 —— 忽略多余的
/// - 一个都没给 → 全部按 0 分，并明确说明「judge 没给出判定」
///
/// ## 为什么缺的按 0 分而不是跳过
///
/// 跳过会让分数虚高 —— 「judge 没判的句子」不等于「这句话有依据」。
/// 保守处理更符合「宁可低估，不要高估」的原则。
fn parse_sentence_verdicts(raw: &str, expected: usize) -> EvalResult<Vec<SentenceVerdict>> {
    let cleaned = strip_code_block(raw);
    let json_part = extract_json_object(cleaned).ok_or_else(|| missing_json_error(raw))?;

    let parsed: SentenceVerdicts = serde_json::from_str(&json_part).map_err(|e| {
        EvalError::Judging(format!(
            "解析 judge 的 JSON 失败（{e}）: {}",
            truncate(&json_part)
        ))
    })?;

    let mut verdicts: Vec<SentenceVerdict> = Vec::with_capacity(expected);

    for i in 0..expected {
        let score = parsed.scores.get(i).copied().unwrap_or(0.0);
        let reason = parsed.reasons.get(i).cloned().unwrap_or_default();
        verdicts.push(SentenceVerdict {
            score: score.clamp(0.0, 1.0),
            reason,
        });
    }

    Ok(verdicts)
}

/// 批量反推问题的响应体。
#[derive(Debug, Clone, serde::Deserialize)]
struct InferredQuestions {
    /// 与输入片段一一对应的问题列表。
    #[serde(default)]
    questions: Vec<String>,
}

/// 解析批量反推出���问题。
///
/// 和 [`parse_sentence_verdicts`] 同样的容错策略：
/// 少了的补空串（会被当成「这句提不出问题」→ 不算相关）。
fn parse_inferred_questions(raw: &str, expected: usize) -> EvalResult<Vec<String>> {
    let cleaned = strip_code_block(raw);
    let json_part = extract_json_object(cleaned).ok_or_else(|| missing_json_error(raw))?;

    let parsed: InferredQuestions = serde_json::from_str(&json_part).map_err(|e| {
        EvalError::Judging(format!(
            "解析 judge 的 JSON 失败（{e}）: {}",
            truncate(&json_part)
        ))
    })?;

    let mut questions: Vec<String> = Vec::with_capacity(expected);
    for i in 0..expected {
        let q = parsed
            .questions
            .get(i)
            .map(|s| clean_inferred_question(s))
            .unwrap_or_default();
        questions.push(q);
    }

    Ok(questions)
}

/// 调用 judge，失败时自动重试。
///
/// ## 为什么需要重试
///
/// 实测发现 judge 偶尔会返回**空响应** —— 网络抖动或上游限流时的瞬时故障。
///
/// 这种情况下直接记「计算失败」是对的（失败 ≠ 0 分），
/// 但**不该因为一次抖动就丢掉整个指标的评估**。
///
/// 所以这里做有限次重试：
///
/// - 空响应 → 重试（最常见）
/// - 调用出错 → 重试（网络抖动）
/// - **解析失败 → 不重试** —— 说明模型确实返回了看不懂的内容，重试也是同样结果
///
/// ## 为什么解析失败不重试
///
/// 重试只会浪费时间和 API 费用 —— 如果模型的输出格式持续不对，
/// 多试几次还是不对。这种情况应该**如实记为失败**，
/// 让报告暴露「judge 提示词需要改」这个事实。
async fn chat_with_retry(
    llm: &dyn rag_07::llm::Llm,
    system: &str,
    user: &str,
    attempts: usize,
) -> EvalResult<String> {
    let mut last_error = String::new();

    for attempt in 1..=attempts.max(1) {
        match llm.chat(system, user).await {
            Ok(raw) if !raw.trim().is_empty() => return Ok(raw),
            Ok(_) => {
                // ⚠️ 空响应是最难诊断的失败 —— 报告里只写「judge 返回空响应」，
                // 根本看不出是「思考吃光了 token」还是「网络抖动」。
                //
                // 推理模型 + 小 `max_tokens` 是头号嫌疑：
                // 思考 token 和正文共用预算，思考没结束正文就是空的。
                // 所以这里把「最可能的原因」直接写进错误信息。
                last_error = "judge 返回空响应（可能是推理 token 吃光了 max_tokens 预算，\
                    尝试关闭思考模式或调大 EVAL_JUDGE_MAX_TOKENS）"
                    .to_string();
                warn!(attempt, attempts, "judge 返回空响应，准备重试");
            }
            Err(e) => {
                last_error = e.to_string();
                warn!(attempt, attempts, error = %e, "judge 调用失败，准备重试");
            }
        }

        // 还有重试机会 → 等一下再试（指数退避）。
        if attempt < attempts.max(1) {
            tokio::time::sleep(std::time::Duration::from_millis(300 * attempt as u64)).await;
        }
    }

    Err(EvalError::Judging(format!(
        "调用 judge 失败（重试 {} 次）: {last_error}",
        attempts.max(1)
    )))
}

/// Faithfulness —— 答案里的陈述是否都能从上下文里找到依据。
///
/// ## 评估逻辑
///
/// 1. 把答案**拆成陈述句**
/// 2. 对每句问 judge：「这句话能不能从上下文推出来？」
/// 3. 忠实度 = 有依据的陈述数 / 总陈述数
///
/// ## 这是 RAG 系统最重要的指标
///
/// 因为**幻觉**是 RAG 最大的风险：模型在上下文里找不到答案时，
/// 会用自己的先验知识编一个看起来合理的答案。
///
/// 用户无法分辨「这是文档里写的」和「这是模型自己想的」——
/// 除非答案带引用标记（这是 06-rag 的 Prompt 在做的事）。
///
/// ## 为什么拆句而不是整段判
///
/// 整段判的问题是：**一段里可能 3 句对、1 句错，judge 会给 0.75 这种模糊分数**，
/// 而你不知道错在哪句。
///
/// 拆句判能得到「4 句里第 3 句无据」这样的定位信息。
pub struct Faithfulness {
    llm: Arc<dyn rag_07::llm::Llm>,
}

impl Faithfulness {
    /// 创建。
    pub fn new(llm: Arc<dyn rag_07::llm::Llm>) -> Self {
        Self { llm }
    }
}

impl std::fmt::Debug for Faithfulness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Faithfulness")
            .field("model", &self.llm.model_name())
            .finish()
    }
}

/// 拆分答案成陈述句。
///
/// 按中文/英文的句末标点切分：`。！？.!?` 以及换行。
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '。' | '！' | '？' | '!' | '?' | '；' | ';' | '\n') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

impl Metric for Faithfulness {
    fn name(&self) -> &str {
        "faithfulness"
    }

    fn description(&self) -> &str {
        "答案中的陈述有多少能从检索到的上下文里得到支持（RAGAs faithfulness）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let context = ctx.context_text();
            if context.trim().is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "没有检索到任何上下文，无法评估忠实度",
                ));
            }

            let sentences = split_sentences(&ctx.answer);
            if sentences.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "答案为空"));
            }

            // ⚠️ **批量判定** —— 一次 LLM 调用处理所有句子。
            //
            // 逐句调用听起来更精确，但代价是 N 次网络往返：
            // 22 个样本 × 平均 8 句 = 176 次调用，光这一项就要几分钟。
            //
            // 批量版让 judge 一次返回所有句子的判定：
            //
            // ```json
            // { "scores": [1.0, 0.0, 1.0], "reasons": ["", "这句上下文没有提到", ""] }
            // ```
            //
            // 既省了 8 倍的往返，又**保留了逐句定位能力** —— 哪个句子是幻觉照样能指出来。
            let system = format!(
                "{JUDGE_INSTRUCTION}\n\n\
             任务：逐条判断【待评估的陈述列表】里的每一句，是否能从【上下文】中找到直接依据。\n\n\
             - 如果陈述的内容能在上下文里找到支持（哪怕是换一种说法），该条 score 给 1.0。\n\
             - 如果陈述的内容上下文里完全没有提到，该条 score 给 0.0。\n\
             - 如果陈述只有部分能得到支持，该条 score 给 0.5。\n\
             - 数字、专有名词、结论必须完全一致才算支持。\n\n\
             输出格式：{{\"scores\": [每句一个分数], \"reasons\": [每句一个理由，无问题则为空字符串]}}"
            );

            let numbered: Vec<String> = sentences
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {s}", i + 1))
                .collect();
            let user = format!(
                "【上下文】\n{context}\n\n【待评估的陈述列表】\n{}",
                numbered.join("\n")
            );

            let raw =
                chat_with_retry(self.llm.as_ref(), &system, &user, JUDGE_RETRY_ATTEMPTS).await?;

            let verdicts = parse_sentence_verdicts(&raw, sentences.len())?;

            let mut supported = 0usize;
            let mut reasons: Vec<String> = Vec::new();

            for (i, sentence) in sentences.iter().enumerate() {
                let verdict = &verdicts[i];
                if verdict.score >= 0.5 {
                    supported += 1;
                } else {
                    let why = if verdict.reason.is_empty() {
                        String::new()
                    } else {
                        format!("（{}）", verdict.reason)
                    };
                    reasons.push(format!("「{}」无依据{}", truncate(sentence), why));
                }
            }

            let value = supported as f32 / sentences.len() as f32;

            let reason = if reasons.is_empty() {
                format!("{supported}/{} 句都有依据", sentences.len())
            } else {
                format!(
                    "{supported}/{} 句有依据；问题：{}",
                    sentences.len(),
                    reasons.join("；")
                )
            };

            Ok(MetricScore::with_reason(value, reason))
        })
    }

    fn threshold(&self) -> f32 {
        0.9
    }
}

/// Answer Relevancy —— 答案和**问题**相关吗？
///
/// ## 评估逻辑
///
/// RAGAs 的做法很巧：
///
/// 1. 让模型**从答案反推出问题**（"根据这个答案，你能问出什么问题？"）
/// 2. 把反推出来的问题和原问题算**语义相似度**
/// 3. 相似度就是 relevancy
///
/// ## 为什么不用「答案和标准答案比」
///
/// 因为答案和标准答案的措辞可能完全不同（这是正常的），
/// 而「答案和问题的相关性」是**更本质**的性质：
///
/// - 答偏了（说了和问题无关的东西）→ 反推出的问题对不上原问题 → 低分
/// - 答得啰嗦但对 → 反推出的问题和原问题一致 → 高分
///
/// 这个方法还顺带检测了**答非所问**和**答了但没答到点上**两种失败。
///
/// ## 语义相似度怎么算
///
/// 用**嵌入模型**算余弦相似度 —— 这就是为什么本指标需要
/// 一个 [`rag_05::embed::Embedder`]。
pub struct AnswerRelevancy {
    llm: Arc<dyn rag_07::llm::Llm>,
    embedder: Arc<dyn rag_05::embed::Embedder>,
}

impl AnswerRelevancy {
    /// 创建。
    pub fn new(llm: Arc<dyn rag_07::llm::Llm>, embedder: Arc<dyn rag_05::embed::Embedder>) -> Self {
        Self { llm, embedder }
    }
}

impl std::fmt::Debug for AnswerRelevancy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnswerRelevancy")
            .field("model", &self.llm.model_name())
            .field("embedder", &self.embedder.model_name())
            .finish()
    }
}

/// 从答案反推出问题。
pub async fn infer_question(llm: &dyn rag_07::llm::Llm, answer: &str) -> EvalResult<String> {
    let system = "\
你是一个问题生成器。根据给定的答案文本，推断出用户可能提出的那个问题。
只输出问题本身，不要输出任何解释、前缀或引号。";

    let user = format!("【答案】\n{answer}\n\n请推断出对应的提问：");

    let raw = chat_with_retry(llm, system, &user, JUDGE_RETRY_ATTEMPTS).await?;

    Ok(clean_inferred_question(&raw))
}

/// 清洗模型反推出的问题。
///
/// 模型经常返回 `"根据文档内容，问题是：XXX"` 或者 `问题：XXX`。
/// 这些前缀对嵌入是纯噪声 —— 去掉。
fn clean_inferred_question(raw: &str) -> String {
    let mut text = raw.trim();

    // 去掉 markdown 包裹。
    if text.starts_with("```")
        && let Some(i) = text.find('\n')
    {
        text = text[i + 1..].trim();
    }
    if let Some(i) = text.rfind("```") {
        text = text[..i].trim();
    }

    // 去掉常见前缀。
    for prefix in [
        "问题：",
        "问题:",
        "提问：",
        "提问:",
        "Question:",
        "question:",
        "根据答案，问题是：",
        "根据答案，问题是:",
        "推断的问题：",
        "推断的问题:",
        "可能的问题是：",
        "可能的问题是:",
    ] {
        if let Some(stripped) = text.strip_prefix(prefix) {
            text = stripped.trim();
            break;
        }
    }

    // 去掉首尾引号。
    for (open, close) in [('"', '"'), ('\'', '\''), ('「', '」'), ('“', '”')] {
        if text.len() >= 2 && text.starts_with(open) && text.ends_with(close) {
            text = text[open.len_utf8()..text.len() - close.len_utf8()].trim();
            break;
        }
    }

    // 只取第一行 —— 模型有时会补充说明。
    text.lines().next().unwrap_or(text).trim().to_string()
}

impl Metric for AnswerRelevancy {
    fn name(&self) -> &str {
        "answer_relevancy"
    }

    fn description(&self) -> &str {
        "答案与问题的语义相关度（反推问题法，RAGAs answer_relevancy）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            if ctx.answer.trim().is_empty() {
                return Ok(MetricScore::with_reason(0.0, "答案为空"));
            }

            // 1. 从答案反推出问题。
            let inferred = infer_question(self.llm.as_ref(), &ctx.answer).await?;

            if inferred.trim().is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "无法从答案反推出问题（答案可能完全跑题）",
                ));
            }

            // 2. 算「原问题」和「反推问题」的语义相似度。
            //
            // ⚠️ 两边都按 **query** 方式嵌入 —— 它们都是「问题」，
            // 用 passage 前缀会落到模型分布的另一个区域。
            let original = ctx.case.question.clone();
            let similarity = self.semantic_similarity(&original, &inferred).await?;

            Ok(MetricScore::with_reason(
                similarity,
                format!(
                    "原问题「{}」↔ 反推问题「{}」，相似度 {similarity:.3}",
                    truncate(&original),
                    truncate(&inferred)
                ),
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.7
    }
}

impl AnswerRelevancy {
    /// 两段文本的语义相似度（余弦）。
    async fn semantic_similarity(&self, a: &str, b: &str) -> EvalResult<f32> {
        let embedder = Arc::clone(&self.embedder);
        let a = a.to_string();
        let b = b.to_string();

        // 嵌入是 CPU 密集的同步调用 —— 放到阻塞线程池。
        let (va, vb) = tokio::task::spawn_blocking(move || {
            let va = embedder
                .embed_query(&a)
                .map_err(|e| EvalError::Judging(format!("嵌入失败: {e}")))?;
            let vb = embedder
                .embed_query(&b)
                .map_err(|e| EvalError::Judging(format!("嵌入失败: {e}")))?;
            Ok::<_, EvalError>((va, vb))
        })
        .await
        .map_err(|e| EvalError::Internal(format!("嵌入任务失败: {e}")))??;

        Ok(rag_05::embed::cosine_similarity(&va, &vb).clamp(0.0, 1.0))
    }
}

/// Context Relevancy —— 检索到的上下文和问题相关吗？
///
/// ## 和其它指标的区别
///
/// | 指标 | 评估对象 |
/// |---|---|
/// | `context_relevancy` | **上下文** ↔ 问题 |
/// | `answer_relevancy` | **答案** ↔ 问题 |
/// | `faithfulness` | **答案** ↔ 上下文 |
///
/// 三个指标构成一个三角形：
///
/// ```text
///        问题
///        ╱  ╲
///    相关╱    ╲相关
///      ╱      ╲
///   上下文 ──相关── 答案
///             （faithfulness）
/// ```
///
/// ## 评估逻辑
///
/// 1. 把上下文**拆成句**
/// 2. 让模型从每个句子里**抽取问题**
/// 3. 每个抽取的问题和原问题算语义相似度
/// 4. 相关度 = 相似度超过阈值的句子比例
///
/// ⚠️ 注意第 2 步：不是直接问「这句和问题相关吗」，
/// 而是**让模型反过来从这句里提出问题**，再和问题比。
///
/// 这样做的原因：直接问「相关吗」，模型会给很宽松的分数；
/// 而「这句话能回答什么问题」是个更**可操作**的问题 ——
/// 如果这句话根本提不出任何合理的问题，它就是无关内容。
pub struct ContextRelevancy {
    llm: Arc<dyn rag_07::llm::Llm>,
    embedder: Arc<dyn rag_05::embed::Embedder>,
    /// 单句相关度阈值（超过才算相关）。
    threshold: f32,
}

impl ContextRelevancy {
    /// 创建。
    pub fn new(llm: Arc<dyn rag_07::llm::Llm>, embedder: Arc<dyn rag_05::embed::Embedder>) -> Self {
        Self {
            llm,
            embedder,
            threshold: 0.5,
        }
    }

    /// 设置单句相关度阈值。
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

impl std::fmt::Debug for ContextRelevancy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextRelevancy")
            .field("model", &self.llm.model_name())
            .field("embedder", &self.embedder.model_name())
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl Metric for ContextRelevancy {
    fn name(&self) -> &str {
        "context_relevancy"
    }

    fn description(&self) -> &str {
        "检索到的上下文中有多少内容与问题相关（RAGAs context_relevancy）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            if ctx.contexts.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "没有检索到任何上下文"));
            }

            // 把所有上下文拆句。
            let mut sentences: Vec<String> = Vec::new();
            for context in &ctx.contexts {
                sentences.extend(split_sentences(context));
            }

            if sentences.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "上下文没有可评估的句子"));
            }

            let original = ctx.case.question.clone();

            // ⚠️ **批量反推** —— 一次 LLM 调用处理所有句子。
            //
            // 和 faithfulness 同样的理由：逐句调用是 N 次网络往返，
            // 而上下文通常有 20~40 句 —— 22 个样本就是近千次调用，根本跑不动。
            let system = "\
你是一个问题生成器。根据给定的文本片段列表，为**每一句**分别推断出它能够回答的那个问题。
只输出 JSON，不要输出任何解释、前缀或引号。

输出格式：{\"questions\": [与输入片段一一对应的问题列表]}";

            let numbered: Vec<String> = sentences
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {s}", i + 1))
                .collect();
            let user = format!("【文本片段列表】\n{}", numbered.join("\n"));

            let raw =
                chat_with_retry(self.llm.as_ref(), system, &user, JUDGE_RETRY_ATTEMPTS).await?;

            let inferred = parse_inferred_questions(&raw, sentences.len())?;

            let mut relevant = 0usize;
            let mut reasons: Vec<String> = Vec::new();

            for (i, sentence) in sentences.iter().enumerate() {
                let question = &inferred[i];
                if question.trim().is_empty() {
                    reasons.push(format!("「{}」提不出问题", truncate(sentence)));
                    continue;
                }

                let similarity = self.semantic_similarity(&original, question).await?;

                if similarity >= self.threshold {
                    relevant += 1;
                } else {
                    reasons.push(format!(
                        "「{}」与问题相似度仅 {similarity:.2}",
                        truncate(sentence)
                    ));
                }
            }

            let value = relevant as f32 / sentences.len() as f32;

            let reason = if reasons.is_empty() {
                format!("{}/{} 句与问题相关", relevant, sentences.len())
            } else {
                format!(
                    "{}/{} 句与问题相关；问题：{}",
                    relevant,
                    sentences.len(),
                    reasons.join("；")
                )
            };

            Ok(MetricScore::with_reason(value, reason))
        })
    }

    fn threshold(&self) -> f32 {
        0.5
    }
}

impl ContextRelevancy {
    /// 两段文本的语义相似度（余弦）。
    async fn semantic_similarity(&self, a: &str, b: &str) -> EvalResult<f32> {
        let embedder = Arc::clone(&self.embedder);
        let a = a.to_string();
        let b = b.to_string();

        let (va, vb) = tokio::task::spawn_blocking(move || {
            let va = embedder
                .embed_query(&a)
                .map_err(|e| EvalError::Judging(format!("嵌入失败: {e}")))?;
            let vb = embedder
                .embed_query(&b)
                .map_err(|e| EvalError::Judging(format!("嵌入失败: {e}")))?;
            Ok::<_, EvalError>((va, vb))
        })
        .await
        .map_err(|e| EvalError::Internal(format!("嵌入任务失败: {e}")))??;

        Ok(rag_05::embed::cosine_similarity(&va, &vb).clamp(0.0, 1.0))
    }
}

/// Answer Correctness —— 答案和标准答案说的是一回事吗？
///
/// ## 评估逻辑
///
/// 让 judge 同时看到「标准答案」和「实际答案」，判断两者的**内容**是否一致。
///
/// ## 为什么不给「正确性」更高的权重
///
/// 因为它对**措辞**太敏感，而且要求评估集有**标准答案**。
/// 真实场景里很多问题没有唯一标准答案（比如「解释一下 TCP 三次握手」）。
///
/// 所以本项目把它作为**参考指标**而不是核心指标 ——
/// 核心还是 faithfulness（答案有没有据可依）。
///
/// ## ⚠️ 和 `token_f1` 的关系
///
/// 两者都是「答案 vs 标准答案」，但口径不同：
///
/// - `token_f1` —— 词面重合度，**确定性**，可复现
/// - `answer_correctness` —— 语义一致度，**judge**，有抖动
///
/// 一起看：F1 高 + correctness 高 = 稳；F1 低 + correctness 高 = 对但换了说法；
/// F1 高 + correctness 低 = 关键词对但意思错。
pub struct AnswerCorrectness {
    llm: Arc<dyn rag_07::llm::Llm>,
}

impl AnswerCorrectness {
    /// 创建。
    pub fn new(llm: Arc<dyn rag_07::llm::Llm>) -> Self {
        Self { llm }
    }
}

impl std::fmt::Debug for AnswerCorrectness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnswerCorrectness")
            .field("model", &self.llm.model_name())
            .finish()
    }
}

impl Metric for AnswerCorrectness {
    fn name(&self) -> &str {
        "answer_correctness"
    }

    fn description(&self) -> &str {
        "答案与标准答案的语义一致度（RAGAs answer_correctness）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let reference = ctx.case.reference_answer.trim();
            if reference.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标准答案，无法评估正确性",
                ));
            }

            if ctx.answer.trim().is_empty() {
                return Ok(MetricScore::with_reason(0.0, "答案为空"));
            }

            let system = format!(
                "{JUDGE_INSTRUCTION}\n\n\
             任务：判断【实际答案】在内容上是否与【标准答案】一致。\n\n\
             - 只比较**内容含义**，忽略措辞差异、用词习惯和详略程度。\n\
             - 实际答案包含了标准答案的全部要点且没有矛盾 → score 给 1.0。\n\
             - 实际答案覆盖了大部分要点但有遗漏或错误 → score 给 0.5 左右。\n\
             - 实际答案与标准答案矛盾或完全跑题 → score 给 0.0。"
            );

            let user = format!("【标准答案】\n{reference}\n\n【实际答案】\n{}", ctx.answer);

            let raw =
                chat_with_retry(self.llm.as_ref(), &system, &user, JUDGE_RETRY_ATTEMPTS).await?;

            let judgement = parse_judgement(&raw)?;

            Ok(MetricScore::with_reason(
                judgement.score.clamp(0.0, 1.0),
                if judgement.reason.is_empty() {
                    format!("judge 判定分数 {}", judgement.score)
                } else {
                    judgement.reason
                },
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json() {
        let j = parse_judgement(r#"{"score": 0.8, "reason": "还行"}"#).unwrap();
        assert_eq!(j.score, 0.8);
        assert_eq!(j.reason, "还行");
    }

    #[test]
    fn parses_json_in_code_block() {
        let raw = "```json\n{\"score\": 0.6, \"reason\": \"有依据\"}\n```";
        let j = parse_judgement(raw).unwrap();
        assert_eq!(j.score, 0.6);
    }

    #[test]
    fn parses_json_with_surrounding_prose() {
        let raw =
            "好的，我的评估结果如下：\n{\"score\": 0.9, \"reason\": \"很准确\"}\n希望有帮助！";
        let j = parse_judgement(raw).unwrap();
        assert_eq!(j.score, 0.9);
    }

    #[test]
    fn parses_json_with_nested_braces() {
        let raw = r#"{"score": 0.5, "reason": "包含 {示例} 花括号"}"#;
        let j = parse_judgement(raw).unwrap();
        assert!(j.reason.contains("示例"));
    }

    #[test]
    fn parses_json_with_string_containing_brace() {
        let raw = r#"{"score": 0.5, "reason": "提到 } 的情况"}"#;
        let j = parse_judgement(raw).unwrap();
        assert_eq!(j.score, 0.5);
    }

    #[test]
    fn rejects_output_without_json() {
        let err = parse_judgement("我认为这个答案还不错").unwrap_err();
        assert_eq!(err.status_code(), 502);
    }

    #[test]
    fn missing_json_error_hints_at_prompt_when_no_brace() {
        // 模型压根没输出 JSON —— 提示词的问题。
        let err = missing_json_error("我认为这个答案还不错");
        let msg = err.to_string();
        assert!(msg.contains("没有输出 JSON"), "{msg}");
        assert!(msg.contains("提示词"), "{msg}");
    }

    #[test]
    fn missing_json_error_hints_at_truncation_when_brace_present() {
        // 有 `{` 没有 `}` —— 截断。
        let err = missing_json_error(r#"{"scores": [1.0, 0.5], "reasons": ["某个很长的理由"#);
        let msg = err.to_string();
        assert!(msg.contains("截断"), "{msg}");
        assert!(msg.contains("MAX_TOKENS"), "{msg}");
    }

    #[test]
    fn rejects_malformed_json() {
        let err = parse_judgement("{score: 0.5}").unwrap_err();
        assert_eq!(err.status_code(), 502);
    }

    #[test]
    fn splits_sentences_on_punctuation() {
        let sentences = split_sentences("第一句。第二句！第三句？");
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("第一句"));
    }

    #[test]
    fn splits_sentences_keeps_trailing_fragment() {
        let sentences = split_sentences("完整句。没有句号");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn splits_sentences_on_newline() {
        let sentences = split_sentences("第一行\n第二行");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn cleans_inferred_question_prefix() {
        assert_eq!(
            clean_inferred_question("问题：什么是 RAG？"),
            "什么是 RAG？"
        );
        assert_eq!(
            clean_inferred_question("Question: what is RAG?"),
            "what is RAG?"
        );
    }

    #[test]
    fn cleans_inferred_question_quotes() {
        assert_eq!(clean_inferred_question("\"什么是 RAG？\""), "什么是 RAG？");
    }

    #[test]
    fn cleans_inferred_question_code_block() {
        assert_eq!(
            clean_inferred_question("```\n什么是 RAG？\n```"),
            "什么是 RAG？"
        );
    }

    #[test]
    fn cleans_inferred_question_takes_first_line() {
        assert_eq!(
            clean_inferred_question("什么是 RAG？\n补充说明：……"),
            "什么是 RAG？"
        );
    }

    #[tokio::test]
    async fn chat_with_retry_retries_on_empty_response() {
        // 第一次返回空，第二次返回正常内容。
        let llm = rag_07::llm::ScriptedLlm::new(vec!["   ".to_string(), "正常答案".to_string()]);

        let raw = chat_with_retry(&llm, "s", "u", 3).await.unwrap();

        assert_eq!(raw, "正常答案");
        assert_eq!(llm.call_count(), 2, "空响应应该触发一次重试");
    }

    #[tokio::test]
    async fn chat_with_retry_gives_up_after_attempts() {
        let llm = rag_07::llm::ScriptedLlm::always("");

        let err = chat_with_retry(&llm, "s", "u", 2).await.unwrap_err();

        assert_eq!(err.status_code(), 502);
        assert_eq!(llm.call_count(), 2, "应该尝试了 2 次");
    }
}
