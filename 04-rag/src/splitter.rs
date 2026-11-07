//! 文本分块层（Splitters）—— 把长文本变成适合检索的块。
//!
//! 分块是 RAG 里**最影响效果、也最被低估**的一步。块太大，检索精度下降；
//! 块太小，语义不完整，模型拿到碎片答不出来。
//!
//! ## 本模块的三种策略
//!
//! | 策略 | 切分依据 | 适用 |
//! |---|---|---|
//! | [`RecursiveSplitter`] | 递归降级：段落 → 句子 → 词 | 通用文本，无结构 |
//! | [`MarkdownSplitter`] | CommonMark 块级元素 → 标题 | Markdown |
//! | [`StructureAwareSplitter`] | 标题层级（section 边界） | 有层级的文档 |
//!
//! 后面两者都是「结构感知」的：切分点落在文档作者明确标出的边界上，
//! 而不是硬按字符数切。后者还会把**标题路径**（`# 文档 > ## 章节`）
//! 作为前缀附到块文本上 —— 这是提升召回率最有效的单个技巧之一。

use serde::{Deserialize, Serialize};

use crate::loader::LoadedDoc;

/// 一个文本块。
///
/// 块不只是文本，还带着元数据 —— 这些元数据是后续「过滤 + 引用」的基础。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// 来源文件路径。
    pub source: String,

    /// 块在文档中的序号（从 0 开始）。
    pub index: usize,

    /// 标题路径，如 `["RAG 检索增强生成", "文本分块策略", "语义分块"]`。
    /// 空表示该块不在任何标题下。
    pub heading_path: Vec<String>,

    /// 块正文（不含标题路径前缀）。
    pub text: String,

    /// 实际送去做嵌入的文本（含标题路径前缀）。
    ///
    /// ⚠️ 检索用的是这个，**不是** `text`。带上标题路径能让向量带上层级语境。
    pub embedded_text: String,

    /// token 数（用真实 tokenizer 数的，不是字符数估算）。
    pub token_count: usize,

    /// 字符数。
    pub char_count: usize,
}

impl Chunk {
    /// 标题路径渲染成一行，如 `# 文档 > ## 章节`。
    #[allow(dead_code)]
    pub fn heading_context(&self) -> String {
        if self.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}", self.heading_path.join(" > "))
        }
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        let head = if self.heading_path.is_empty() {
            "(无标题)".to_string()
        } else {
            self.heading_path.join(" > ")
        };
        let preview: String = self.text.chars().take(50).collect();
        format!(
            "#{} [{}] {} tok / {} 字符 | {}",
            self.index,
            head,
            self.token_count,
            self.char_count,
            preview.replace('\n', " ")
        )
    }
}

/// 分块配置。
#[derive(Debug, Clone)]
pub struct SplitterConfig {
    /// 目标块大小（token 数）。
    pub chunk_size: usize,

    /// 相邻块的重叠大小（token 数）。
    ///
    /// 存在的意义：关键句正好落在切分线上时，靠重叠保证它完整出现在某一块里。
    pub chunk_overlap: usize,

    /// 是否给每个块附加标题路径前缀。
    pub add_heading_context: bool,
}

impl Default for SplitterConfig {
    /// 默认 300 token、10% 重叠。
    ///
    /// 300 是经验值：块再大语义会被稀释，再小上下文不完整。
    fn default() -> Self {
        Self {
            chunk_size: 300,
            chunk_overlap: 30,
            add_heading_context: true,
        }
    }
}

/// token 计数器。
///
/// ⚠️ 为什么不用「字符数 ÷ 4」这种估算？因为误差在中文和代码里极大。
/// 英文大约 4 字符/token，中文大约 1.5 字符/token，代码更碎。
/// 用真实 tokenizer 才能保证块不会超出模型的上下文预算。
///
/// `CoreBPE` 本身实现了 `text_splitter` 的 `ChunkSizer` trait，
/// 所以它既能当独立计数器，也能直接喂给 splitter 做**精确的 token 级切分**。
pub struct TokenCounter {
    bpe: tiktoken_rs::CoreBPE,
}

impl std::fmt::Debug for TokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCounter").finish()
    }
}

impl TokenCounter {
    /// 创建计数器（cl100k_base 编码，OpenAI 系列的通用编码）。
    pub fn new() -> Result<Self, anyhow::Error> {
        let bpe = tiktoken_rs::cl100k_base()?;
        Ok(Self { bpe })
    }

    /// 数一段文本的 token 数。
    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_ordinary(text).len()
    }

    /// 拿到底层 BPE，用于交给 `text-splitter` 做 token 级精确切分。
    pub fn bpe(&self) -> &tiktoken_rs::CoreBPE {
        &self.bpe
    }
}

/// 分块器 trait。
pub trait Splitter: Send + Sync {
    /// 分块器的名字。
    fn name(&self) -> &str;

    /// 把一份加载好的文档切成块。
    fn split(&self, doc: &LoadedDoc, counter: &TokenCounter) -> Vec<Chunk>;
}

/// 从标题路径构造块的公共逻辑。
fn build_chunk(
    doc: &LoadedDoc,
    index: usize,
    heading_path: Vec<String>,
    body: &str,
    counter: &TokenCounter,
    add_heading_context: bool,
) -> Chunk {
    let body = body.trim();
    let heading_context = if heading_path.is_empty() {
        String::new()
    } else {
        format!("# {}", heading_path.join(" > "))
    };

    // 嵌入用的文本 = 标题路径 + 空行 + 正文。
    // 这是提升召回最有效的单个技巧：语义相同的句子在不同章节含义完全不同，
    // 带上路径等于把层级语境注入了向量。
    let embedded_text = if add_heading_context && !heading_context.is_empty() {
        format!("{heading_context}\n\n{body}")
    } else {
        body.to_string()
    };

    Chunk {
        source: doc.file_name.clone(),
        index,
        heading_path,
        text: body.to_string(),
        token_count: counter.count(&embedded_text),
        char_count: embedded_text.chars().count(),
        embedded_text,
    }
}

/// 递归分块器 —— 不看结构，只按语义边界降级切分。
///
/// 对应 LlamaIndex 的 `SentenceSplitter` / LangChain 的
/// `RecursiveCharacterTextSplitter`。切分点优先级：
///
/// ```text
/// 段落(\n\n) → 换行(\n) → 句子 → 词 → 字符
/// ```
///
/// 每一级都尽量多塞，填满目标大小才停。
pub struct RecursiveSplitter {
    config: SplitterConfig,
}

impl RecursiveSplitter {
    /// 创建。
    pub fn new(config: SplitterConfig) -> Self {
        Self { config }
    }

    /// 构造 text-splitter 的配置。
    ///
    /// ⚠️ `with_overlap` 在 overlap ≥ capacity 时会失败（返回 `Err`）。
    /// 这里失败就退回「无重叠」而不是 panic —— 重叠只是优化，
    /// 不该让整条流水线挂掉。
    fn splitter_config<'c>(
        &self,
        counter: &'c TokenCounter,
    ) -> text_splitter::ChunkConfig<&'c tiktoken_rs::CoreBPE> {
        let base =
            text_splitter::ChunkConfig::new(self.config.chunk_size).with_sizer(counter.bpe());
        match base.with_overlap(self.config.chunk_overlap) {
            Ok(cfg) => cfg,
            Err(_) => {
                text_splitter::ChunkConfig::new(self.config.chunk_size).with_sizer(counter.bpe())
            }
        }
    }
}

impl Splitter for RecursiveSplitter {
    fn name(&self) -> &str {
        "recursive"
    }

    fn split(&self, doc: &LoadedDoc, counter: &TokenCounter) -> Vec<Chunk> {
        // 关键：把真实的 BPE tokenizer 作为容量单位喂给 splitter，
        // 这样切分点是在 **token 预算**上选的，而不是「字符数近似 token」。
        //
        // text-splitter 的切分点优先级（从粗到细）：
        //   段落(\n\n) → 换行(\n) → 句子 → 词 → 字形簇 → 字符
        // 每一级都尽量多塞，填满 chunk_size 个 token 才停。
        let config = self.splitter_config(counter);
        let splitter = text_splitter::TextSplitter::new(config);

        let mut chunks = Vec::new();
        for (i, (byte_offset, text)) in splitter.chunk_indices(&doc.text).enumerate() {
            let heading_path = heading_path_at(&doc.text, byte_offset);
            chunks.push(build_chunk(
                doc,
                i,
                heading_path,
                text,
                counter,
                self.config.add_heading_context,
            ));
        }

        chunks
    }
}

/// Markdown 分块器 —— 按 CommonMark 的块级元素切。
///
/// `text-splitter` 的 `MarkdownSplitter` 会解析 CommonMark，
/// 按块级元素（段落、列表、表格、代码块）和标题层级切分。
pub struct MarkdownSplitter {
    config: SplitterConfig,
}

impl MarkdownSplitter {
    /// 创建。
    pub fn new(config: SplitterConfig) -> Self {
        Self { config }
    }

    /// 构造 text-splitter 的配置（overlap 非法时退回无重叠）。
    fn splitter_config<'c>(
        &self,
        counter: &'c TokenCounter,
    ) -> text_splitter::ChunkConfig<&'c tiktoken_rs::CoreBPE> {
        let base =
            text_splitter::ChunkConfig::new(self.config.chunk_size).with_sizer(counter.bpe());
        match base.with_overlap(self.config.chunk_overlap) {
            Ok(cfg) => cfg,
            Err(_) => {
                text_splitter::ChunkConfig::new(self.config.chunk_size).with_sizer(counter.bpe())
            }
        }
    }
}

impl Splitter for MarkdownSplitter {
    fn name(&self) -> &str {
        "markdown"
    }

    fn split(&self, doc: &LoadedDoc, counter: &TokenCounter) -> Vec<Chunk> {
        let config = self.splitter_config(counter);
        let splitter = text_splitter::MarkdownSplitter::new(config);

        let mut chunks = Vec::new();
        for (i, (byte_offset, text)) in splitter.chunk_indices(&doc.text).enumerate() {
            let heading_path = heading_path_at(&doc.text, byte_offset);
            chunks.push(build_chunk(
                doc,
                i,
                heading_path,
                text,
                counter,
                self.config.add_heading_context,
            ));
        }

        chunks
    }
}

/// 结构感知分块器 —— 沿标题层级切，并附加标题路径。
///
/// 这是本项目**推荐的**策略，用 docling 的 section 解析器实现：
///
/// 1. 把 Markdown 解析成「标题路径 + 正文词」的 section 序列
/// 2. 沿 section 边界切，**块永不跨标题**
/// 3. section 内部超长时，再按词窗口滑动切分
///
/// 对比 [`RecursiveSplitter`] 的关键差异：**块不会跨越标题边界**。
/// 递归分块可能把「语义分块」这一节的最后一段和「文档结构感知分块」
/// 这一节的开头拼在一起 —— 语义完全混乱。
pub struct StructureAwareSplitter {
    config: SplitterConfig,
}

impl StructureAwareSplitter {
    /// 创建。
    pub fn new(config: SplitterConfig) -> Self {
        Self { config }
    }
}

impl Splitter for StructureAwareSplitter {
    fn name(&self) -> &str {
        "structure-aware"
    }

    fn split(&self, doc: &LoadedDoc, counter: &TokenCounter) -> Vec<Chunk> {
        let sections = docling::chunker::parse_sections_with_stack(&doc.text, Vec::new()).0;

        let mut chunks = Vec::new();
        let mut index = 0usize;

        for section in &sections {
            if section.words.is_empty() {
                continue;
            }

            let body = section.words.join(" ");
            let body_tokens = counter.count(&body);

            // section 本身就能塞进一个块 → 不再细分。
            if body_tokens <= self.config.chunk_size {
                chunks.push(build_chunk(
                    doc,
                    index,
                    section.heading_path.clone(),
                    &body,
                    counter,
                    self.config.add_heading_context,
                ));
                index += 1;
                continue;
            }

            // section 超长 → 在 section 内部按 token 预算滑动窗口切，
            // 并保留重叠词，保证跨界的句子完整。
            for window in self.sliding_windows(&body, counter) {
                chunks.push(build_chunk(
                    doc,
                    index,
                    section.heading_path.clone(),
                    &window,
                    counter,
                    self.config.add_heading_context,
                ));
                index += 1;
            }
        }

        chunks
    }
}

impl StructureAwareSplitter {
    /// 在一段文本上滑动 token 预算窗口，词级别带重叠。
    ///
    /// 做法：以词为单位累积，每凑够 `chunk_size` 个 token 就切一刀，
    /// 下一刀从上一刀末尾回退 `chunk_overlap` 个 token 开始。
    fn sliding_windows(&self, text: &str, counter: &TokenCounter) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let mut windows = Vec::new();
        let mut start = 0usize;

        while start < words.len() {
            // 从 start 开始，往后累加到超出 chunk_size 为止。
            let mut end = start;
            let mut tokens = 0usize;
            while end < words.len() {
                let t = counter.count(words[end]);
                if tokens + t > self.config.chunk_size && end > start {
                    break;
                }
                tokens += t;
                end += 1;
            }

            windows.push(words[start..end].join(" "));

            if end >= words.len() {
                break;
            }

            // 回退 overlap 个 token（换算成词数）作为下一窗口的起点。
            let overlap_words = self.overlap_words(counter, &words[start..end]);
            // 保证窗口一定前进，否则会死循环。
            let step = (end - start).saturating_sub(overlap_words).max(1);
            start += step;
        }

        windows
    }

    /// 把 token 级的重叠预算换算成词数。
    fn overlap_words(&self, counter: &TokenCounter, words: &[&str]) -> usize {
        if self.config.chunk_overlap == 0 {
            return 0;
        }
        let mut used = 0usize;
        let mut count = 0usize;
        // 从后往前累加，凑够 overlap 的 token 就停。
        for word in words.iter().rev() {
            let t = counter.count(word);
            if used + t > self.config.chunk_overlap {
                break;
            }
            used += t;
            count += 1;
        }
        count.min(words.len().saturating_sub(1))
    }
}

/// 找出某个字节偏移处所处的标题路径。
///
/// 做法：扫描全文，维护标题栈（按层级），返回该偏移位置生效的路径。
fn heading_path_at(text: &str, byte_offset: usize) -> Vec<String> {
    use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

    let mut stack: Vec<String> = Vec::new();
    let mut current_level = 0usize;
    let mut heading_buf = String::new();
    let mut in_heading = false;

    for (event, range) in Parser::new(text).into_offset_iter() {
        // 只有块开始位置在目标偏移之前的标题才生效。
        if range.start > byte_offset {
            break;
        }

        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                in_heading = true;
                heading_buf.clear();
                current_level = match level {
                    HeadingLevel::H1 => 1,
                    HeadingLevel::H2 => 2,
                    HeadingLevel::H3 => 3,
                    HeadingLevel::H4 => 4,
                    HeadingLevel::H5 => 5,
                    HeadingLevel::H6 => 6,
                };
            }
            Event::End(TagEnd::Heading(_)) => {
                in_heading = false;
                let idx = current_level.saturating_sub(1);
                if stack.len() <= idx {
                    stack.resize(idx + 1, String::new());
                } else {
                    stack.truncate(idx + 1);
                }
                stack[idx] = heading_buf.trim().to_string();
            }
            Event::Text(t) | Event::Code(t) if in_heading => {
                if !heading_buf.is_empty() {
                    heading_buf.push(' ');
                }
                heading_buf.push_str(&t);
            }
            _ => {}
        }
    }

    stack.into_iter().filter(|h| !h.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(text: &str) -> LoadedDoc {
        LoadedDoc {
            source: "test.md".into(),
            file_name: "test.md".into(),
            format: "markdown".into(),
            text: text.into(),
            page_count: None,
            loader: "test".into(),
        }
    }

    const SAMPLE: &str = "\
# 顶层标题

这是顶层的第一段。

## 二级标题

这是二级下的第一段，内容比较长，用来测试分块行为。

### 三级标题

这是三级下的内容。
";

    #[test]
    fn structure_aware_never_crosses_heading() {
        let counter = TokenCounter::new().unwrap();
        let splitter = StructureAwareSplitter::new(SplitterConfig {
            chunk_size: 200,
            chunk_overlap: 0,
            add_heading_context: true,
        });

        let chunks = splitter.split(&doc(SAMPLE), &counter);

        assert!(!chunks.is_empty());
        // 每个块的标题路径都应该非空（示例全是标题下的内容）
        for c in &chunks {
            assert!(!c.heading_path.is_empty(), "chunk lost its heading: {c:?}");
        }
        // 一级标题块和二级标题块不应混在一起
        assert_eq!(chunks[0].heading_path, vec!["顶层标题"]);
    }

    #[test]
    fn heading_context_is_prefixed() {
        let counter = TokenCounter::new().unwrap();
        let splitter = StructureAwareSplitter::new(SplitterConfig::default());
        let chunks = splitter.split(&doc(SAMPLE), &counter);

        let with_ctx = chunks.iter().find(|c| !c.heading_path.is_empty()).unwrap();
        assert!(with_ctx.embedded_text.starts_with("# "));
        assert!(with_ctx.embedded_text.len() > with_ctx.text.len());
    }

    #[test]
    fn overlapping_windows_share_content() {
        let counter = TokenCounter::new().unwrap();
        // 用可区分的词，这样重叠才看得出来。
        // （用重复的 "word" 会让每个窗口长得一模一样，测不出重叠。）
        let words: Vec<String> = (0..200).map(|i| format!("w{i}")).collect();
        let long = words.join(" ");

        let splitter = StructureAwareSplitter::new(SplitterConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            add_heading_context: false,
        });

        let chunks = splitter.split(&doc(&long), &counter);
        assert!(chunks.len() > 1, "long text should be split");

        // 相邻窗口应该共享尾部/头部的一些词 —— 这就是重叠。
        let first: Vec<&str> = chunks[0].text.split_whitespace().collect();
        let second: Vec<&str> = chunks[1].text.split_whitespace().collect();

        let shared: Vec<&&str> = first.iter().filter(|w| second.contains(w)).collect();
        assert!(
            !shared.is_empty(),
            "adjacent windows should overlap, but first={:?} second={:?}",
            first.last(),
            second.first()
        );
    }

    #[test]
    fn no_overlap_yields_disjoint_windows() {
        let counter = TokenCounter::new().unwrap();
        let words: Vec<String> = (0..200).map(|i| format!("w{i}")).collect();
        let long = words.join(" ");

        let splitter = StructureAwareSplitter::new(SplitterConfig {
            chunk_size: 50,
            chunk_overlap: 0,
            add_heading_context: false,
        });

        let chunks = splitter.split(&doc(&long), &counter);
        assert!(chunks.len() > 1);

        let first: Vec<&str> = chunks[0].text.split_whitespace().collect();
        let second: Vec<&str> = chunks[1].text.split_whitespace().collect();
        assert!(
            !first.iter().any(|w| second.contains(w)),
            "windows should not share words when overlap is 0"
        );
    }

    #[test]
    fn token_count_is_measured_not_estimated() {
        let counter = TokenCounter::new().unwrap();
        // 中文大约 1.5 字符/token，用真实 tokenizer 应该明显大于 字符数/4
        let text = "检索增强生成是一种把外部知识库接进大语言模型的方法";
        let chars = text.chars().count();
        let tokens = counter.count(text);
        assert!(tokens > 0);
        assert!(
            tokens > chars / 4,
            "real token count ({tokens}) should exceed naive chars/4 estimate ({})",
            chars / 4
        );
    }
}
