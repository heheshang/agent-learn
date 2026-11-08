#!/usr/bin/env python3
"""生成一个带**内嵌位图**的测试 PDF。

PDF 里的图片有两种：
- **矢量图**：路径指令画出来的（线、矩形）—— 没有「图片」这个概念
- **位图**：一个 XObject（`/Subtype /Image`），带压缩后的像素数据

本脚本生成第二种 —— 一张真正的 PNG 位图，嵌进 PDF。
这样才能测出 lopdf 的图片抽取路径。
"""

import struct
import sys
import zlib

PAGE_W, PAGE_H = 612, 792
MARGIN_X, MARGIN_Y = 72, 700


def esc(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_raw_rgb(width: int, height: int) -> bytes:
    """造一张有渐变和色块的**裸 RGB 像素**（不是 PNG）。

    PDF 的 `/DeviceRGB` + `/BitsPerComponent 8` 图像流里装的就是裸像素：
    每行 `width * 3` 字节，没有 PNG 那种逐行 filter 字节。
    """
    raw = bytearray()
    for y in range(height):
        for x in range(width):
            if height // 3 <= y < 2 * height // 3:
                raw += bytes((250, 220, 40))  # 黄色带
            else:
                r = int(255 * x / max(1, width - 1))
                b = int(255 * y / max(1, height - 1))
                raw += bytes((r, 40, b))
    return bytes(raw)


def build_pdf() -> bytes:
    pixels = make_raw_rgb(160, 120)
    compressed_pixels = zlib.compress(pixels, 9)

    # 图注文字
    caption = "Figure 1. A synthetic bitmap embedded in the PDF."

    content_ops = [
        # 标题
        f"BT /F1 16 Tf {MARGIN_X} {MARGIN_Y} Td (Document With An Embedded Bitmap) Tj ET",
        # 说明段落
        f"BT /F1 10 Tf {MARGIN_X} {MARGIN_Y - 40} Td "
        f"(The figure below is a real raster image stored as a PDF image XObject.) Tj ET",
        # 画图片：缩放到 240x180 放在页面中部
        "q 240 0 0 180 72 480 cm /Im1 Do Q",
        # 图注
        f"BT /F1 10 Tf {MARGIN_X} {MARGIN_Y - 260} Td ({esc(caption)}) Tj ET",
    ]
    content = "\n".join(content_ops).encode("latin-1")
    compressed = zlib.compress(content)

    objs: list[bytes] = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> "
        b"/Contents 4 0 R >>"
    )
    objs.append(
        b"<< /Length "
        + str(len(compressed)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + compressed
        + b"\nendstream"
    )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # 图片 XObject：FlateDecode 的裸 RGB 像素
    objs.append(
        b"<< /Type /XObject /Subtype /Image /Width 160 /Height 120 "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
        b"/Length "
        + str(len(compressed_pixels)).encode()
        + b" >>\nstream\n"
        + compressed_pixels
        + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()

    return bytes(out)


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "with_image.pdf"
    with open(out_path, "wb") as f:
        f.write(build_pdf())
    print(f"wrote {out_path}")
