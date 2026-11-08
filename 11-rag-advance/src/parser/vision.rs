//! 图片描述 —— 用多模态模型把图片变成**可检索的文本**。
//!
//! ## 为什么必须有这一步
//!
//! 图片不能直接检索。向量模型吃文本。
//!
//! 所以图片进入知识库的唯一路径是：
//!
//! ```text
//! 图片 ──多模态模型──► 「一张柱状图，横轴是策略，纵轴是耗时」
//!                                    │
//!                                    ▼
//!                            文本 ──嵌入──► 向量 ──► 可检索
//! ```
//!
//! ## 不做这一步会怎样
//!
//! 图片被完全忽略 —— 或者更糟，只能靠文件名和周边文字，
//! 检索到「见第 3 页的图」这种没信息量的东西。
//!
//! ## 为什么默认关闭
//!
//! - 要花钱（多模态 API 比文本模型贵）
//! - 要 API key（`OPENAI_API_KEY` 之类）
//! - 不是每个场景都需要
//!
//! 所以默认是**关**的，用 `ADVANCE_IMAGE_DESCRIBE=true` 打开。
//!
//! ## ⚠️ 抽出来的图和正文的关系
//!
//! 这里有个容易忽略的点：**图片的描述应该带上周边上下文**。
//!
//! 光看图片，模型只能说「一张柱状图」。但如果告诉它
//! 「这段文字描述的是三种提取策略在表格还原上的耗时对比」，
//! 它就能说「一张柱状图，展示 Flat / Column aware / Layout model
//! 三种策略的耗时，Layout model 最慢」。
//!
//! 后者才是**能检索到**的描述。

use agent_logging::{info, warn};
use std::path::Path;

use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ContentPart, MessageContent};

use crate::error::{AdvanceError, AdvanceResult};
use crate::parser::image::ExtractedImage;

/// 图片描述器。
pub struct VisionDescriber {
    client: genai::Client,
    model: String,
    max_tokens: u32,
}

impl std::fmt::Debug for VisionDescriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionDescriber")
            .field("model", &self.model)
            .finish()
    }
}

impl VisionDescriber {
    /// 创建描述器。
    pub fn new(model: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            client: genai::Client::builder().build(),
            model: model.into(),
            max_tokens,
        }
    }

    /// 描述一张图片。
    ///
    /// `context` 是图片周边的文字（通常是它前面那个段落的末尾）。
    /// 它能显著提升描述的信息量。
    pub async fn describe(&self, image_path: &Path, context: &str) -> AdvanceResult<String> {
        let data_url = crate::parser::image::to_data_url(image_path)?;

        let prompt = build_prompt(context);

        let request = ChatRequest::new(vec![ChatMessage::user(MessageContent::from_parts(vec![
            ContentPart::from_text(prompt),
            ContentPart::from_binary_base64(
                media_type(&data_url),
                data_url,
                image_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string()),
            ),
        ]))]);

        let options = ChatOptions::default()
            .with_temperature(0.1)
            .with_max_tokens(self.max_tokens);

        let response = self
            .client
            .exec_chat(&self.model, request, Some(&options))
            .await
            .map_err(|e| AdvanceError::Internal(format!("调用多模态模型失败: {e}")))?;

        Ok(response
            .into_first_text()
            .unwrap_or_default()
            .trim()
            .to_string())
    }
}

/// 从 data URL 里抠出 MIME 类型。
fn media_type(data_url: &str) -> String {
    data_url
        .strip_prefix("data:")
        .and_then(|rest| rest.split(';').next())
        .unwrap_or("image/png")
        .to_string()
}

/// 构造提示词。
///
/// ## 提示词为什么这么写
///
/// 目标是「产出的文本能被检索命中」，所以要求：
///
/// - **描述图里的具体内容**（不是「这是一张柱状图」这种废话）
/// - **用文档里的术语**（和正文用同一套词，检索才对得上）
/// - **带上周边上下文**里的概念
/// - **控制长度** —— 太长的描述会稀释向量
fn build_prompt(context: &str) -> String {
    let context_part = if context.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\n\n这张图出现在下面这段文字附近，请用同一套术语来描述它：\n---\n{}\n---",
            context.trim()
        )
    };

    format!(
        "请用一段话描述这张图片的内容，用于知识库检索。\n\
         要求：\n\
         1. 说清楚图里**具体画了什么**（类型、坐标轴、图例、关键数值）\n\
         2. 如果图里有文字或数字，把它们写出来\n\
         3. 不要写「这是一张图片」这类空话\n\
         4. 长度控制在 60 字以内\n\
         只输出描述本身，不要任何前缀。{context_part}"
    )
}

/// 给一批图片生成描述。
///
/// 返回 `(图片 ID, 描述)` 的列表。失败的图片被跳过 ——
/// 一张图描述失败不该让整批失败。
pub async fn describe_images(
    describer: &VisionDescriber,
    images: &[ExtractedImage],
    asset_dir: &Path,
    contexts: &std::collections::HashMap<String, String>,
) -> AdvanceResult<Vec<(String, String)>> {
    let mut out = Vec::new();

    for image in images {
        if image.status != crate::parser::image::ImageStatus::Extracted {
            continue;
        }

        let path = asset_dir.join(&image.path);
        let context = contexts.get(&image.id).map(|s| s.as_str()).unwrap_or("");

        match describer.describe(&path, context).await {
            Ok(description) if !description.is_empty() => {
                out.push((image.id.clone(), description));
            }
            Ok(_) => {}
            Err(e) => {
                warn!(image_id = %image.id, error = %e, "图片描述失败");
            }
        }
    }

    info!(image_count = out.len(), "图片描述完成");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_type_is_extracted_from_the_data_url() {
        assert_eq!(media_type("data:image/png;base64,AAAA"), "image/png");
        assert_eq!(media_type("data:image/jpeg;base64,AAAA"), "image/jpeg");
    }

    #[test]
    fn prompt_includes_the_context() {
        let prompt = build_prompt("这段文字讲的是混合检索。");
        assert!(prompt.contains("混合检索"));
    }

    #[test]
    fn prompt_without_context_still_works() {
        let prompt = build_prompt("   ");
        assert!(prompt.contains("用于知识库检索"));
        assert!(!prompt.contains("---"));
    }
}
