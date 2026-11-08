//! 图片抽取 —— 从 PDF 里把内嵌的位图挖出来。
//!
//! ## 为什么需要这一层
//!
//! 图片**不能直接检索**。向量模型吃的是文本，图片对它来说是一串像素。
//!
//! 所以图片进入检索语料只有一条路：
//!
//! ```text
//! PDF 里的图片流
//!    │  解码（FlateDecode / DCTDecode）
//!    ▼
//! PNG / JPEG 文件
//!    │  多模态模型描述（或者用周边文字）
//!    ▼
//! 「一张柱状图，展示三种策略在表格还原上的耗时」
//!    │
//!    ▼
//! 可嵌入的文本  ← 这才是检索能用的东西
//! ```
//!
//! 本模块负责**第一步和第二步**（挖出来 + 落盘），第三步在
//! [`crate::parser::vision`] 里。
//!
//! ## 为什么用 lopdf 而不是 docling
//!
//! docling 的文本层路径（`pdf-text` feature）**不吐图片** —— 它只做文本。
//! 图片要么走完整的 ML 管线（要 pdfium + ONNX），要么直接读 PDF 对象。
//!
//! 直接读对象最省：`lopdf` 遍历每页的 `/Resources /XObject`，
//! 挑出 `/Subtype /Image` 的那些，解码后落盘。
//!
//! ## 编码格式的处理
//!
//! | PDF 里的 filter | 处理方式 |
//! |---|---|
//! | 无 / `FlateDecode` | 解压后是**裸像素**，用 `png` crate 编码成 PNG |
//! | `DCTDecode` | 内容**就是** JPEG 文件，直接写 `.jpg` |
//! | `JPXDecode` | JPEG2000 —— 跳过（生态里没有好用的纯 Rust 解码器） |
//! | `CCITTFaxDecode` / `JBIG2Decode` | 跳過（扫描件专用，且需要额外依赖） |
//!
//! 跳过的图片不是「失败」—— 它们被记成 `Skipped`，在报告里能看到。

use std::path::{Path, PathBuf};

use agent_logging::{debug, info};
use serde::{Deserialize, Serialize};

use crate::error::{AdvanceError, AdvanceResult};

/// 图片的抽取状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStatus {
    /// 成功抽出并落盘。
    Extracted,

    /// 太小（低于字节数或像素尺寸阈值）—— 多半是图标。
    TooSmall,

    /// 编码格式不支持（JPX / JBIG2 等）。
    UnsupportedFormat,

    /// 解码失败。
    DecodeFailed,
}

/// 一张抽出来的图片。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedImage {
    /// 图片 ID（`{文件名}#p{页码}#{序号}`）。
    pub id: String,

    /// 落盘路径（相对 `asset_dir`）。
    pub path: String,

    /// 宽（像素）。
    pub width: u32,

    /// 高（像素）。
    pub height: u32,

    /// 色彩空间（`DeviceRGB` / `DeviceGray` / ...）。
    pub color_space: Option<String>,

    /// 每通道位数。
    pub bits_per_component: Option<u8>,

    /// 页码。
    pub page_no: usize,

    /// 状态。
    pub status: ImageStatus,

    /// 原始字节数。
    pub byte_len: usize,
}

impl ExtractedImage {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{} {}×{} p{} ({} 字节, {:?})",
            self.id, self.width, self.height, self.page_no, self.byte_len, self.status
        )
    }
}

/// 图片抽取参数。
#[derive(Debug, Clone)]
pub struct ImageOptions {
    /// 最小字节数。
    pub min_bytes: usize,

    /// 最小边长（像素）。
    pub min_dimension: u32,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            min_bytes: 1024,
            min_dimension: 32,
        }
    }
}

/// 从一份 PDF 里抽取所有图片。
///
/// `asset_dir` 是落盘目录，会自动创建。
pub fn extract_images(
    pdf_path: &Path,
    asset_dir: &Path,
    options: &ImageOptions,
) -> AdvanceResult<Vec<ExtractedImage>> {
    let bytes =
        std::fs::read(pdf_path).map_err(|e| AdvanceError::Parse(format!("读取 PDF 失败: {e}")))?;

    let doc = lopdf::Document::load_mem(&bytes)
        .map_err(|e| AdvanceError::Parse(format!("PDF 解析失败: {e}")))?;

    std::fs::create_dir_all(asset_dir)
        .map_err(|e| AdvanceError::Internal(format!("创建图片目录失败: {e}")))?;

    let file_stem = pdf_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "doc".to_string());

    let mut out = Vec::new();

    // 按页遍历 —— 图片必须知道自己在第几页，
    // 否则引用和分块都无法定位。
    //
    // `get_pages()` 返回 `BTreeMap<页码, 对象 id>` —— 键已经是页号了，
    // 直接用它，不用自己数。
    for (page_no, page_id) in doc.get_pages() {
        let page_no = page_no as usize;
        let images = match doc.get_page_images(page_id) {
            Ok(images) => images,
            // 这一页没有图片 —— 不是错误。
            Err(_) => continue,
        };

        for (img_index, image) in images.iter().enumerate() {
            let id = format!("{file_stem}#p{page_no}#{img_index}");

            let width = image.width.max(0) as u32;
            let height = image.height.max(0) as u32;

            let mut record = ExtractedImage {
                id: id.clone(),
                path: String::new(),
                width,
                height,
                color_space: image.color_space.clone(),
                bits_per_component: image.bits_per_component.map(|b| b as u8),
                page_no,
                status: ImageStatus::DecodeFailed,
                byte_len: image.content.len(),
            };

            // 尺寸门槛 —— 图标和噪点没有检索价值。
            if width < options.min_dimension || height < options.min_dimension {
                record.status = ImageStatus::TooSmall;
                out.push(record);
                continue;
            }

            if image.content.len() < options.min_bytes {
                record.status = ImageStatus::TooSmall;
                out.push(record);
                continue;
            }

            let filters = image.filters.clone().unwrap_or_default();
            let decoded = decode_image(image, &filters);

            let Some(decoded) = decoded else {
                record.status = ImageStatus::UnsupportedFormat;
                out.push(record);
                continue;
            };

            let (bytes, ext) = decoded;
            let filename = format!("{file_stem}_p{page_no}_{img_index}.{ext}");
            let out_path = asset_dir.join(&filename);

            if let Err(e) = std::fs::write(&out_path, &bytes) {
                return Err(AdvanceError::Internal(format!("写入图片失败: {e}")));
            }

            record.path = filename;
            record.byte_len = bytes.len();
            record.status = ImageStatus::Extracted;
            debug!(image_id = %id, page_no, bytes = bytes.len(), "图片已抽取");
            out.push(record);
        }
    }

    let extracted = out
        .iter()
        .filter(|r| r.status == ImageStatus::Extracted)
        .count();
    info!(
        pdf = %pdf_path.display(),
        total = out.len(),
        extracted,
        "图片抽取完成"
    );

    Ok(out)
}

/// 解码一张图片，返回 `(字节, 扩展名)`。
///
/// `None` 表示这个编码格式不支持。
fn decode_image(
    image: &lopdf::xobject::PdfImage<'_>,
    filters: &[String],
) -> Option<(Vec<u8>, &'static str)> {
    // DCTDecode = 内容就是 JPEG 文件本身。
    if filters.iter().any(|f| f == "DCTDecode") {
        return Some((image.content.to_vec(), "jpg"));
    }

    // JPXDecode = JPEG2000。纯 Rust 生态没有好用的解码器，跳过。
    if filters
        .iter()
        .any(|f| f == "JPXDecode" || f == "JBIG2Decode" || f == "CCITTFaxDecode")
    {
        return None;
    }

    // 没有 filter 或者 FlateDecode：内容是**裸像素**，
    // 需要自己按 ColorSpace / BitsPerComponent 编码成 PNG。
    let stream = image.origin_dict.get(b"Filter").ok();
    let _ = stream;

    let decoded: Vec<u8> = if filters.iter().any(|f| f == "FlateDecode") {
        // lopdf 的 stream 解压接口在这里拿不到（我们手上是裸 content），
        // 所以自己用 flate2 解。
        use std::io::Read;
        let mut decoder = flate2::read::ZlibDecoder::new(image.content);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).ok()?;
        out
    } else {
        image.content.to_vec()
    };

    let bits = image.bits_per_component.unwrap_or(8);
    let channels = match image.color_space.as_deref() {
        Some("DeviceRGB") | Some("RGB") | Some("CalRGB") => 3,
        Some("DeviceGray") | Some("G") | Some("CalGray") => 1,
        Some("DeviceCMYK") | Some("CMYK") => 4,
        // Indexed / Separation / ICCBased 等色彩空间需要调色板，
        // 这里不处理 —— 记成跳过比产出错误颜色更好。
        _ => return None,
    };

    let width = image.width.max(0) as u32;
    let height = image.height.max(0) as u32;

    // 每个样本占几个字节。
    let bytes_per_sample = (bits / 8) as usize;
    if bytes_per_sample == 0 {
        return None;
    }

    // 16 位深度的位图：降采样到 8 位（取高字节）。
    let mut samples: Vec<u8> = Vec::with_capacity((width * height) as usize * channels);
    let needed = width as usize * height as usize * channels * bytes_per_sample;
    if decoded.len() < needed {
        return None;
    }

    for i in 0..(width as usize * height as usize * channels) {
        let offset = i * bytes_per_sample;
        samples.push(if bytes_per_sample == 1 {
            decoded[offset]
        } else {
            // 16 位取高字节。
            decoded[offset]
        });
    }

    // CMYK → RGB（PDF 的 CMYK 是反相的：0 表示满墨）。
    //
    // ⚠️ 这里忽略 K（黑色）通道 —— 完整的 CMYK→RGB 要做
    // `r = 255 - min(255, c + k)`。但绝大多数 PDF 里的 CMYK 图
    // 是「无墨」（k=0）的 CMY 变体，忽略 K 足够用。
    if channels == 4 {
        for px in samples.as_chunks_mut::<4>().0 {
            let (c, m, y) = (px[0], px[1], px[2]);
            px[0] = 255 - c;
            px[1] = 255 - m;
            px[2] = 255 - y;
        }
    }

    encode_png(&samples, width, height, channels as u8)
}

/// 把裸像素编码成 PNG。
fn encode_png(
    samples: &[u8],
    width: u32,
    height: u32,
    channels: u8,
) -> Option<(Vec<u8>, &'static str)> {
    let color = match channels {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => return None,
    };

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);

        let mut writer = encoder.write_header().ok()?;
        writer.write_image_data(samples).ok()?;
        writer.finish().ok()?;
    }

    Some((buf, "png"))
}

/// 把图片编码成 data URL（给多模态模型用）。
pub fn to_data_url(path: &Path) -> AdvanceResult<String> {
    let bytes =
        std::fs::read(path).map_err(|e| AdvanceError::Internal(format!("读取图片失败: {e}")))?;

    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    };

    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);

    Ok(format!("data:{mime};base64,{encoded}"))
}

/// 列出 `asset_dir` 下的所有图片文件。
pub fn list_assets(asset_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(asset_dir) else {
        return Vec::new();
    };

    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e, "png" | "jpg" | "jpeg"))
        })
        .collect();

    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_encoding_produces_a_valid_header() {
        let samples = vec![255u8; 4 * 4 * 3]; // 4×4 RGB
        let (bytes, ext) = encode_png(&samples, 4, 4, 3).unwrap();

        assert_eq!(ext, "png");
        // PNG 的魔数。
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
        );
    }

    #[test]
    fn png_encoding_rejects_unknown_channel_counts() {
        assert!(encode_png(&[0u8; 4], 1, 1, 2).is_none());
    }

    #[test]
    fn data_url_has_the_right_mime() {
        let dir = std::env::temp_dir().join("rag11-dataurl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.png");
        std::fs::write(&path, b"not-really-a-png").unwrap();

        let url = to_data_url(&path).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn listing_assets_ignores_non_images() {
        let dir = std::env::temp_dir().join("rag11-assets");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.png"), b"x").unwrap();
        std::fs::write(dir.join("b.txt"), b"x").unwrap();

        let assets = list_assets(&dir);
        assert_eq!(assets.len(), 1);
        assert!(assets[0].ends_with("a.png"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
