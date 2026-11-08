#!/usr/bin/env python3
"""生成一个「复杂」测试 PDF：标题层级 + 表格 + 矢量图 + 纯文本。

手写 PDF 语法（和 04-rag/data/make_pdf.py 同款思路），但额外支持：
- 表格：用线段画格线 + 逐格放文字（PDF 里没有「表格对象」，
  表格是排版出来的，解析器只能靠几何位置还原）
- 矢量图：circle / line 等路径指令（用来验证图片节点的产出）
"""

import sys
import zlib

PAGE_W, PAGE_H = 612, 792
MARGIN_X, MARGIN_Y = 72, 720
LINE_H = 15

BODY, HEAD, SUBHEAD = 10, 16, 13


def esc(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def wrap(text: str, width: int, size: int = BODY) -> list[str]:
    limit = max(20, int(width / (size * 0.5)))
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > limit:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


class Canvas:
    def __init__(self):
        self.ops: list[str] = []
        self.y = MARGIN_Y

    def text(self, size, x, y, s):
        self.ops.append(f"BT /F1 {size} Tf {x} {y} Td ({esc(s)}) Tj ET")

    def heading(self, s, size=HEAD):
        self.text(size, MARGIN_X, self.y, s)
        self.y -= LINE_H + 8

    def subheading(self, s):
        self.text(SUBHEAD, MARGIN_X, self.y, s)
        self.y -= LINE_H + 4

    def para(self, s, size=BODY):
        for line in wrap(s, PAGE_W - 2 * MARGIN_X, size):
            self.text(size, MARGIN_X, self.y, line)
            self.y -= LINE_H
        self.y -= LINE_H * 0.4

    def table(self, headers: list[str], rows: list[list[str]], col_width=150):
        """用线段画表格边框，逐格放文字。

        表格没有任何 PDF 语义 —— 它的结构完全来自这些线的位置。
        """
        left = MARGIN_X
        n_cols = len(headers)
        row_h = LINE_H + 6
        top = self.y + row_h

        # 单元格坐标
        def cell_xy(ci, ri):
            x = left + ci * col_width
            y = top - ri * row_h - row_h
            return x, y

        total_rows = len(rows) + 1
        # 画横线
        for ri in range(total_rows + 1):
            y = top - ri * row_h
            self.ops.append(f"{left} {y} m {left + n_cols * col_width} {y} l S")
        # 画竖线
        for ci in range(n_cols + 1):
            x = left + ci * col_width
            self.ops.append(f"{x} {top} m {x} {top - total_rows * row_h} l S")

        # 表头（加粗字号）
        for ci, h in enumerate(headers):
            x, y = cell_xy(ci, 0)
            self.text(BODY, x + 4, y + 5, h)
        # 数据行
        for ri, row in enumerate(rows):
            for ci, v in enumerate(row):
                x, y = cell_xy(ci, ri + 1)
                self.text(BODY, x + 4, y + 5, v)

        self.y = top - total_rows * row_h - LINE_H

    def vector_figure(self, caption: str):
        """画一个简单的矢量图（矩形 + 两条折线），模拟示意图。"""
        left = MARGIN_X
        bottom = self.y - 120
        w, h = 240, 100
        self.ops.append(
            f"{left} {bottom} m {left + w} {bottom} l {left + w} {bottom + h} l "
            f"{left} {bottom + h} l h S"
        )
        # 一条折线
        self.ops.append(
            f"{left + 10} {bottom + 20} m {left + 70} {bottom + 80} l "
            f"{left + 130} {bottom + 45} l {left + w - 10} {bottom + 90} l S"
        )
        # 一条虚线轴
        self.ops.append(
            f"{left + 10} {bottom + 10} m {left + w - 10} {bottom + 10} l S"
        )
        self.y = bottom - LINE_H
        self.text(BODY, MARGIN_X, self.y, caption)
        self.y -= LINE_H * 1.5


def build_pdf() -> bytes:
    c = Canvas()

    c.heading("Advanced Data Processing for Retrieval")
    c.para(
        "This document exercises the parsing pipeline for complex documents. "
        "It contains a nested heading hierarchy, a real table drawn with vector "
        "lines, and a vector figure. None of these carry PDF semantics, so a "
        "parser can only recover them from geometry."
    )

    c.subheading("1. Why flat text extraction fails")
    c.para(
        "A flat text extractor reads the drawing operators in order and treats "
        "every glyph run as a paragraph. Tables therefore collapse into a "
        "run of values with no row or column meaning, and figure labels end up "
        "mixed into the surrounding prose. Retrieval on top of such text cannot "
        "answer questions that require reading across a row."
    )

    c.subheading("2. Measured layout drift")
    c.para(
        "The table below compares three extraction strategies on the same "
        "document. The column order reflects the reading order the parser "
        "recovered, not the logical order of the source."
    )
    c.table(
        ["Strategy", "Tables", "Reading order", "Cost"],
        [
            ["Flat text", "none", "column major", "low"],
            ["Column aware", "partial", "row major", "medium"],
            ["Layout model", "full", "row major", "high"],
        ],
    )

    c.para(
        "Figure 1 shows the drift of the recovered reading order as the "
        "document gets denser. The horizontal axis is the number of regions "
        "on a page and the vertical axis is the fraction of regions that came "
        "back in the wrong order."
    )
    c.vector_figure("Figure 1. Reading order drift against region density.")

    c.subheading("3. What a table has to preserve")
    c.para(
        "A table has to preserve three things at once: the header row, the "
        "row boundaries, and the cell to column mapping. Losing any one of "
        "them makes the table worse than useless, because it looks structured "
        "to a reader while carrying no recoverable meaning for a retriever."
    )
    c.table(
        ["Property", "Why it matters", "Loss symptom"],
        [
            ["Header row", "Names the columns", "Values lose their axis"],
            ["Row boundary", "Groups related cells", "Rows merge into prose"],
            ["Column map", "Enables cell lookup", "Cross row answers fail"],
        ],
    )

    c.para(
        "The last paragraph is deliberately plain prose so that a parser "
        "which only handles tables has something to fail on."
    )

    content = "\n".join(c.ops).encode("latin-1", errors="replace")

    objs: list[bytes] = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
    )
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
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()

    return bytes(out)


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "complex_layout.pdf"
    with open(out_path, "wb") as f:
        f.write(build_pdf())
    print(f"wrote {out_path}")
