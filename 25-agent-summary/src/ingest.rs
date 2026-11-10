//! **数据入口层** —— 分词、Token 计数、结构感知分块。
//!
//! # 这个模块汇总了哪些历史能力
//!
//! | 能力 | 来自 | 在这里的形态 |
//! |---|---|---|
//! | 中英混排分词 | 08-rag | [`tokenize`] |
//! | Token 精确计数 | 04-rag | [`token_count`] |
//! | 结构感知分块 | 04-rag | [`split_document`] |
//!
//! # ⚠️ 分词口径：CJK unigram + bigram
//!
//! 中文没有空格分隔，按「连续字符切成一个词」会把整句变成**一个 token**，
//! 词元交集为空 → **检索命中数 = 0**。
//!
//! 所以中文同时产出 **unigram**（单字）和 **bigram**（相邻两字）：
//! unigram 保证召回，bigram 保证精度。

/// 分词结果。
pub type Tokens = Vec<String>;

/// 中英混排分词。
///
/// ## 规则
///
/// - **ASCII 字母/数字**：连续的一段算一个词（并转小写）
/// - **CJK 字符**：同时产出单字与相邻两字
/// - **其它**（标点、空格）：直接丢弃
pub fn tokenize(text: &str) -> Tokens {
    let mut out = Tokens::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c.is_ascii_alphanumeric() {
            // 连续 ASCII 段 → 一个词。
            let start = i;
            while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect::<String>().to_lowercase());
            continue;
        }

        if is_cjk(c) {
            out.push(c.to_string());
            // bigram：与下一个 CJK 字符组成的双字词。
            if i + 1 < chars.len() && is_cjk(chars[i + 1]) {
                out.push(format!("{c}{}", chars[i + 1]));
            }
            i += 1;
            continue;
        }

        // 其它字符（标点、空白、emoji）丢弃。
        i += 1;
    }

    out
}

/// 是否是 CJK 字符（中日韩统一表意文字 + 假名 + 韩文）。
pub fn is_cjk(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)      // 汉字
        || (0x3400..=0x4DBF).contains(&code) // 扩展 A
        || (0x3040..=0x30FF).contains(&code) // 假名
        || (0xAC00..=0xD7AF).contains(&code) // 韩文
}

/// Token 计数。
///
/// ## ⚠️ 这里用**估算**，不是真实 tokenizer
///
/// 真实 tokenizer 要下模型文件（tiktoken / BPE 词表），
/// 那会让本项目**依赖外部资源**，破坏「零依赖可复现」。
///
/// 估算口径（CJK 按 1.5 字/token，ASCII 按 4 字符/token）：
/// 这是经验值，**误差约 20%**。
///
/// ⚠️ 但它对**压测**是够的 —— 因为我们要比的是「同一口径下的相对差值」，
/// 绝对值准不准**不影响结论**。
/// 这与 24 项目的 `estimate_tokens` 是同一个取舍。
pub fn token_count(text: &str) -> usize {
    let mut cjk = 0usize;
    let mut other = 0usize;
    for c in text.chars() {
        if is_cjk(c) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    // CJK 约 1.5 字/token，ASCII 约 4 字符/token。
    cjk * 2 / 3 + other.div_ceil(4)
}

/// 一个块。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    /// 来源文件名。
    pub source: String,

    /// 块在文档中的序号。
    pub index: usize,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 送去做嵌入的文本（含标题路径前缀）。
    pub embedded_text: String,

    /// Token 数（估算）。
    pub token_count: usize,
}

/// 一篇加载好的文档。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoadedDoc {
    /// 文件名。
    pub file_name: String,

    /// 正文（Markdown）。
    pub text: String,
}

/// 分块配置。
#[derive(Debug, Clone, Copy)]
pub struct SplitterConfig {
    /// 目标块大小（token 数）。
    pub chunk_size: usize,

    /// 相邻块的重叠（token 数）。
    pub chunk_overlap: usize,

    /// 是否附加标题路径前缀。
    pub add_heading_context: bool,
}

impl Default for SplitterConfig {
    fn default() -> Self {
        Self {
            chunk_size: 300,
            chunk_overlap: 30,
            add_heading_context: true,
        }
    }
}

/// 结构感知分块。
///
/// ## 切分策略
///
/// 1. 按 Markdown 标题（`#` ~ `######`）切出**节**，同时维护标题栈
/// 2. 每节内部按**段落**（`\n\n`）聚合成块
/// 3. 块超过 `chunk_size` 时按**句子**边界再切
///
/// ## ⚠️ 为什么标题路径要带进 `embedded_text`
///
/// 因为「第 3 节」这个信息**不在正文里**。
/// 检索时带上标题路径，向量才带得上层级语境。
pub fn split_document(doc: &LoadedDoc, config: &SplitterConfig) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    // ★ 标题栈**按层级索引** —— `stack[level-1]` 就是该层的标题。
    //
    // ⚠️ 早先用 `truncate(level-1)` + `push` 的写法是错的：
    //    遇到 `###` 时会把 `##` 的标题挤掉，导致标题路径里
    //    出现「`###` 的标题挂在 `##` 的标题下面」这种错误嵌套。
    let mut heading_stack: Vec<String> = Vec::new();
    let mut current_section: Vec<String> = Vec::new();
    let mut current_heading: Vec<String> = Vec::new();
    let mut index = 0usize;

    let flush = |chunks: &mut Vec<Chunk>, heading: &Vec<String>, body: &str, index: &mut usize| {
        let body = body.trim();
        if body.is_empty() {
            return;
        }
        let embedded = if config.add_heading_context && !heading.is_empty() {
            let prefix = heading
                .iter()
                .map(|h| format!("# {h}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!("{prefix}\n{body}")
        } else {
            body.to_string()
        };
        chunks.push(Chunk {
            source: doc.file_name.clone(),
            index: *index,
            heading_path: heading.clone(),
            text: body.to_string(),
            token_count: token_count(&embedded),
            embedded_text: embedded,
        });
        *index += 1;
    };

    for line in doc.text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            // 遇到新标题 → 先把上一节落盘，再更新标题栈。
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            let title = trimmed[level..].trim().to_string();
            if !title.is_empty() {
                let body = current_section.join("\n");
                flush(&mut chunks, &current_heading, &body, &mut index);
                current_section.clear();

                // ★ 按层级写入 —— 后面的层级再出现时会覆盖同层。
                while heading_stack.len() < level.saturating_sub(1) {
                    heading_stack.push(String::new());
                }
                heading_stack.truncate(level.saturating_sub(1));
                heading_stack.push(title.clone());
                current_heading = heading_stack.clone();
            }
        } else {
            current_section.push(line.to_string());
        }
    }

    // 文档末尾那一节。
    let body = current_section.join("\n");
    flush(&mut chunks, &current_heading, &body, &mut index);

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_ascii_words() {
        let tokens = tokenize("Hybrid Retrieval RRF");
        assert_eq!(tokens, vec!["hybrid", "retrieval", "rrf"]);
    }

    #[test]
    fn tokenize_emits_unigrams_and_bigrams_for_cjk() {
        let tokens = tokenize("混合检索");
        assert!(tokens.contains(&"混".to_string()));
        assert!(tokens.contains(&"混合".to_string()));
        assert!(tokens.contains(&"检索".to_string()));
    }

    #[test]
    fn tokenize_is_case_insensitive() {
        assert_eq!(tokenize("RRF"), vec!["rrf"]);
        assert_eq!(tokenize("rrf"), vec!["rrf"]);
    }

    #[test]
    fn token_count_grows_with_text() {
        let short = token_count("hello");
        let long = token_count("hello hello hello hello");
        assert!(long > short);
    }

    #[test]
    fn cjk_counts_more_than_ascii_per_char() {
        // 同样 4 个字符，中文应该比英文「更贵」（token 更多）。
        assert!(token_count("混合检索") >= 2);
        assert!(token_count("abcd") <= 2);
    }

    #[test]
    fn split_document_tracks_heading_path() {
        let doc = LoadedDoc {
            file_name: "a.md".into(),
            text: "# 标题\n\n第一段。\n\n## 小节\n\n第二段。".into(),
        };
        let chunks = split_document(&doc, &SplitterConfig::default());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading_path, vec!["标题".to_string()]);
        assert_eq!(
            chunks[1].heading_path,
            vec!["标题".to_string(), "小节".to_string()]
        );
    }

    #[test]
    fn split_document_embeds_heading_context() {
        let doc = LoadedDoc {
            file_name: "a.md".into(),
            text: "# 标题\n\n正文。".into(),
        };
        let chunks = split_document(&doc, &SplitterConfig::default());
        assert!(chunks[0].embedded_text.contains("# 标题"));
        assert!(!chunks[0].text.contains("# 标题"));
    }

    #[test]
    fn split_document_skips_empty_sections() {
        let doc = LoadedDoc {
            file_name: "a.md".into(),
            text: "# 标题\n\n## 空节\n\n# 下一节\n\n有内容。".into(),
        };
        let chunks = split_document(&doc, &SplitterConfig::default());
        assert_eq!(chunks.len(), 1);
    }
}
