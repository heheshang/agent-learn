#!/usr/bin/env python3
"""检查 README 目录里的锚点链接是否与实际章节标题一致。

GitHub / Cursor 的锚点规则：
- 去掉 markdown 语法标记
- 小写化
- 空格 → 连字符
- 去掉标点（保留中日韩、字母、数字、连字符、下划线）
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def slugify(text: str) -> str:
    """模拟 GitHub 的锚点生成规则。"""
    # 去掉行内的 markdown 链接/强调/代码标记
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = text.strip().lower()
    # 去掉标点，保留字母数字中日韩、连字符、下划线、空格
    text = re.sub(r"[^\w\s\-一-鿿]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def check(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # 收集所有章节标题的锚点
    anchors = set()
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            anchors.add(slugify(m.group(1)))

    # 收集目录里的链接
    problems = []
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in re.finditer(r"\[([^\]]*)\]\(#([^)]+)\)", line):
            target = m.group(2)
            if target not in anchors:
                problems.append(f"  行 {i}: [{m.group(1)}](#{target}) → 找不到对应标题")

    return problems


def main() -> int:
    files = [ROOT / "README.md"]
    files += sorted(ROOT.glob("0*/README.md"))
    files += sorted(ROOT.glob("1*/README.md"))

    total = 0
    for f in files:
        if not f.exists():
            continue
        problems = check(f)
        if problems:
            total += len(problems)
            rel = f.relative_to(ROOT)
            print(f"✗ {rel}")
            for p in problems:
                print(p)
        else:
            print(f"✓ {f.relative_to(ROOT)}")

    print()
    if total == 0:
        print("✓ 全部目录锚点链接有效")
        return 0
    print(f"✗ {total} 个失效锚点")
    return 1


if __name__ == "__main__":
    sys.exit(main())
