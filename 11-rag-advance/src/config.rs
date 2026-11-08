//! 全局配置。
//!
//! 环境变量前缀 `ADVANCE_` —— 和 04（无）、06（`RAG_`）、07（`QUERY_`）、
//! 08（`HYBRID_`）、09（`EVAL_`）、10（`MILVUS_`）都不撞。
//!
//! ## 为什么需要单独一套配置
//!
//! 本项目的参数和别处**完全不同类**：
//!
//! - `ADVANCE_DATA_DIR` —— 要处理的复杂文档目录
//! - `ADVANCE_ASSET_DIR` —— 抽出来的图片落到哪
//! - `ADVANCE_TABLE_MIN_CELLS` —— 一个区域至少多少个单元格才当成表格
//! - `ADVANCE_VISION_MODEL` —— 给图片生成描述用的多模态模型
//!
//! 最后一条是本项目独有的：**图片不能直接检索**，必须先变成文本。

use std::path::PathBuf;

use rag_04::splitter::SplitterConfig;

/// 表格识别参数。
#[derive(Debug, Clone)]
pub struct TableConfig {
    /// 一个区域至少要有多少个单元格才被当成表格。
    ///
    /// 为什么要这个阈值？因为几何聚类会把「两行并排的段落」也识别成表格。
    /// 2 个单元格起步是最保守的线 —— 一行两列的表照样能识别。
    pub min_cells: usize,

    /// 行聚类的容差系数（相对于行高）。
    ///
    /// 同一行的单元格，垂直中心差小于 `行高 × 容差` 才算同一行。
    pub row_tolerance: f32,

    /// 列聚类的容差系数（相对于行高）。
    ///
    /// 左边缘相距小于 `行高 × 容差` 的单元格算同一列。
    pub column_tolerance: f32,
}

impl Default for TableConfig {
    fn default() -> Self {
        Self {
            min_cells: 4,
            row_tolerance: 0.7,
            column_tolerance: 1.5,
        }
    }
}

/// 图片处理参数。
#[derive(Debug, Clone)]
pub struct ImageConfig {
    /// 小于这个字节数的图片直接忽略（多半是分隔点、图标、噪点）。
    pub min_bytes: usize,

    /// 图片的最小边长（像素）。
    ///
    /// PDF 里嵌的小图标尺寸往往只有几个像素 —— 它们没有检索价值，
    /// 抽出来只是浪费 token。
    pub min_dimension: u32,

    /// 是否给图片生成描述（需要多模态模型 + API key）。
    ///
    /// 关掉的话图片仍然会被抽出来落盘，但检索时只能靠文件名和周边文字。
    pub describe: bool,

    /// 生成描述用的模型。
    pub vision_model: String,

    /// 生成描述时的最大 token 数。
    pub vision_max_tokens: u32,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            min_bytes: 1024,
            min_dimension: 32,
            describe: false,
            vision_model: "gpt-4o-mini".to_string(),
            vision_max_tokens: 300,
        }
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 待处理的文档目录。
    pub data_dir: PathBuf,

    /// 抽出来的图片 / 中间产物的输出目录。
    pub asset_dir: PathBuf,

    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 表格识别参数。
    pub table: TableConfig,

    /// 图片处理参数。
    pub image: ImageConfig,

    /// 分块配置。
    pub splitter: SplitterConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            data_dir: manifest.join("data"),
            asset_dir: manifest.join("assets"),
            bind_addr: "127.0.0.1:3040".to_string(),
            table: TableConfig::default(),
            image: ImageConfig::default(),
            splitter: SplitterConfig::default(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `ADVANCE_DATA_DIR` | 文档目录 | `11-rag-advance/data` |
    /// | `ADVANCE_ASSET_DIR` | 图片输出目录 | `11-rag-advance/assets` |
    /// | `ADVANCE_BIND_ADDR` | 监听地址 | `127.0.0.1:3040` |
    /// | `ADVANCE_TABLE_MIN_CELLS` | 表格最少单元格数 | `4` |
    /// | `ADVANCE_IMAGE_DESCRIBE` | 是否生成图片描述 | `false` |
    /// | `ADVANCE_VISION_MODEL` | 多模态模型 | `gpt-4o-mini` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("ADVANCE_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("ADVANCE_ASSET_DIR") {
            config.asset_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("ADVANCE_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_usize("ADVANCE_TABLE_MIN_CELLS") {
            config.table.min_cells = v.max(2);
        }
        if let Some(v) = env_bool("ADVANCE_IMAGE_DESCRIBE") {
            config.image.describe = v;
        }
        if let Some(v) = env_str("ADVANCE_VISION_MODEL") {
            config.image.vision_model = v;
        }
        if let Some(v) = env_u32("ADVANCE_VISION_MAX_TOKENS") {
            config.image.vision_max_tokens = v;
        }
        if let Some(v) = env_usize("ADVANCE_CHUNK_SIZE") {
            config.splitter.chunk_size = v;
        }
        if let Some(v) = env_usize("ADVANCE_CHUNK_OVERLAP") {
            config.splitter.chunk_overlap = v;
        }

        config
    }

    /// 打印当前配置。
    pub fn summary(&self) -> String {
        format!(
            "文档目录: {}\n图片目录: {}\n监听地址: {}\n\
             表格: 最少 {} 个单元格, 行容差 {}, 列容差 {}\n\
             图片: 最小 {} 字节 / {} 像素, 生成描述={} 模型={}\n\
             分块: {} token / {} 重叠",
            self.data_dir.display(),
            self.asset_dir.display(),
            self.bind_addr,
            self.table.min_cells,
            self.table.row_tolerance,
            self.table.column_tolerance,
            self.image.min_bytes,
            self.image.min_dimension,
            self.image.describe,
            self.image.vision_model,
            self.splitter.chunk_size,
            self.splitter.chunk_overlap,
        )
    }
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `u32` 环境变量。
fn env_u32(key: &str) -> Option<u32> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个布尔环境变量（`1/true/yes/on` 为真）。
fn env_bool(key: &str) -> Option<bool> {
    env_str(key).map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bind_addr_matches_the_workspace_range() {
        // 04~10 已经占了 3000~3030，这里用 3040。
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3040");
    }

    #[test]
    fn table_min_cells_has_a_floor() {
        // 少于 2 个单元格就不可能是表格。
        let config = AppConfig::default();
        assert!(config.table.min_cells >= 2);
    }

    #[test]
    fn image_description_is_off_by_default() {
        // 默认不开 —— 它要花钱（多模态 API）。
        assert!(!AppConfig::default().image.describe);
    }
}
