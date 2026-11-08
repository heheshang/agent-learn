//! 测试用的极简 PDF 构造器。
//!
//! ## 为什么手写 PDF 语法
//!
//! 测试夹具必须是**纯文本**的：
//!
//! - 二进制夹具在 git diff 里看不出来，改坏了也没人知道
//! - 没法 review
//!
//! 而且 CI 上不一定有 Python —— `data/*.py` 只是给人看的，
//! 测试必须能在任何机器上跑。
//!
//! ## 手写 PDF 的两个坑
//!
//! ### ① 每行文字要一个独立的 `Tj`
//!
//! 如果一行文字被拆成多个 `Tj` 操作，文本层后端在还原阅读顺序时
//! 会把相邻行交错在一起 —— 这不是解析器的 bug，是 PDF 编码方式
//! 决定的。所以这里**每行一个 `BT ... ET`**。
//!
//! ### ② 坐标要严格递增
//!
//! y 坐标必须单调递减（页面坐标原点在左下角，y 向上）。
//! 不递增的话，阅读顺序还原会乱。

use std::io::Write;

/// 从一组内容流操作构造一份单页 PDF。
///
/// `ops` 是 PDF 内容流的操作字符串，例如
/// `"BT /F1 10 Tf 72 720 Td (Hello) Tj ET"`。
pub fn pdf_from_ops(ops: &[String]) -> Vec<u8> {
    let content = ops.join("\n").into_bytes();
    let compressed = flate2_compress(&content);

    let mut objects: Vec<Vec<u8>> = Vec::new();

    objects.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
    objects.push(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec());
    objects.push(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
    );

    let mut stream_obj = Vec::new();
    stream_obj.extend_from_slice(
        format!(
            "<< /Length {} /Filter /FlateDecode >>\nstream\n",
            compressed.len()
        )
        .as_bytes(),
    );
    stream_obj.extend_from_slice(&compressed);
    stream_obj.extend_from_slice(b"\nendstream");
    objects.push(stream_obj);

    objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());

    assemble(objects)
}

/// 组装 PDF 对象 / xref / trailer。
fn assemble(objects: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n");

    let mut offsets: Vec<usize> = Vec::new();

    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }

    let xref_pos = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );

    out
}

/// zlib 压缩（内容流用）。
///
/// 测试里不想拉一个 flate2 依赖 —— 用「存储模式」的 deflate 就够：
/// PDF 的 `FlateDecode` 接受 zlib 包装，而 zlib 的存储块
/// （`78 01` 头 + 未压缩的 deflate 块 + Adler-32 尾）
/// 是合法的。
fn flate2_compress(data: &[u8]) -> Vec<u8> {
    // zlib 头：CM = 8 (deflate), CINFO = 7 (32K window)
    let mut out = vec![0x78, 0x01];

    // 存储块：BFINAL = 1, BTYPE = 00（不压缩）
    // LEN / NLEN 是长度和它的按位取反。
    let len = data.len() as u16;
    out.push(0x01);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);

    // Adler-32 校验和。
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for byte in data {
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&(((b << 16) | a).to_be_bytes()));

    out
}

/// 让 `Write` 的 import 不被当成死代码（保留给将来写文件的夹具）。
#[allow(dead_code)]
fn _unused(_: &mut impl Write) {}
