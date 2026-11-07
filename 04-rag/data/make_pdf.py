#!/usr/bin/env python3
"""生成一个布局规范的测试 PDF（无第三方依赖，手写 PDF 语法）。

为什么要手写而不是用 cupsfilter？
cupsfilter 生成的 PDF 每行文字被拆成多个独立的 show-text 操作，
行与行之间的 y 坐标差很小，PDF 文本层的后端在还原阅读顺序时
会把相邻行交错在一起 —— 这不是 loader 的 bug，是 PDF 本身的编码方式
决定的。手写 PDF 可以保证每行一个完整的 Tj 操作，坐标严格递增。
"""

import sys
import zlib

PAGE_W, PAGE_H = 612, 792          # US Letter, 单位 pt
MARGIN_X, MARGIN_Y = 72, 720
LINE_H = 15

# (字号, 文本) —— 用来渲染一个标题
BODY = 10
HEAD = 16
SUBHEAD = 13


def esc(s: str) -> str:
    """转义 PDF 字符串里的特殊字符。"""
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def wrap(text: str, width: int) -> list[str]:
    """按最大宽度折行（等宽近似：字号越大每行字数越少）。"""
    limit = max(20, int(width / (BODY * 0.5)))
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > limit:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def build_pdf(sections: list[tuple[str, list[str]]]) -> bytes:
    """sections: [(标题, 段落列表), ...]，返回一个完整的 PDF 字节流。"""
    # ---- 1. 组装内容流 ----
    ops: list[str] = []
    y = MARGIN_Y

    def emit_text(size: int, x: int, yy: int, text: str) -> None:
        # 每行一次 BT/ET + 一个 Tj，坐标严格递增 —— 保证阅读顺序可还原。
        ops.append(f"BT /F1 {size} Tf {x} {yy} Td ({esc(text)}) Tj ET")

    for heading, paragraphs in sections:
        # 标题
        emit_text(HEAD, MARGIN_X, y, heading)
        y -= LINE_H + 8

        # 段落
        for para in paragraphs:
            for line in wrap(para, PAGE_W - 2 * MARGIN_X):
                emit_text(BODY, MARGIN_X, y, line)
                y -= LINE_H
            y -= LINE_H * 0.5  # 段间距

        y -= 10  # 小节间距

    content = "\n".join(ops).encode("latin-1", errors="replace")

    # ---- 2. 组装 PDF 对象 ----
    objs: list[bytes] = []

    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
    )

    # 内容流用 Flate 压缩
    compressed = zlib.compress(content)
    objs.append(
        b"<< /Length "
        + str(len(compressed)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + compressed
        + b"\nendstream"
    )

    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

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
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()

    return bytes(out)


SECTIONS = [
    (
        "Retrieval-Augmented Generation Engineering Notes",
        [
            "This document is a plain-text PDF used to exercise the PDF loading path. "
            "It contains no scanned images and no complex layout, so pure text "
            "extraction is sufficient.",
        ],
    ),
    (
        "Section 1. Why chunking matters",
        [
            "Retrieval quality is dominated by chunk granularity. Chunks that are too "
            "large mix several topics into one vector and dilute the semantic signal. "
            "Chunks that are too small fragment the answer and leave the language model "
            "without enough context to compose a coherent reply.",
        ],
    ),
    (
        "Section 2. Chunk size guidance",
        [
            "A practical default is three hundred to five hundred tokens per chunk, "
            "with an overlap of ten to twenty percent of the chunk size. The overlap "
            "exists so that a sentence landing exactly on a cut boundary still appears "
            "whole in one of the adjacent chunks.",
        ],
    ),
    (
        "Section 3. Structure aware chunking",
        [
            "Documents that carry explicit structure, such as Markdown headings or PDF "
            "chapter titles, give you free semantic boundaries. Prefer splitting on "
            "those boundaries over splitting on arbitrary character counts, because "
            "the author of the document already told you where the topics change.",
        ],
    ),
    (
        "Section 4. Metadata and citations",
        [
            "Every chunk should record its source file, its heading path, and its "
            "ordinal position. The heading path is the single most valuable piece of "
            "metadata: two sentences with identical wording mean different things in "
            "different chapters. Prefixing the heading path onto the chunk text "
            "measurably improves recall.",
        ],
    ),
    (
        "Section 5. Evaluation pitfalls",
        [
            "Evaluating a retrieval pipeline requires held out questions whose answers "
            "are not literally present in the corpus. If the answer appears verbatim in "
            "the document, recall metrics become meaningless because the retriever can "
            "match on surface form alone rather than on meaning.",
        ],
    ),
]


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "rag_notes.pdf"
    with open(out_path, "wb") as f:
        f.write(build_pdf(SECTIONS))
    print(f"wrote {out_path}")
