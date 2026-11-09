#!/usr/bin/env python3
"""
把 JSON 里的 `observation` 字段里的 `@@DATA:` 行剥掉，只留给人看的正文。

## 为什么需要这个

工具的 observation 里除了可见正文，还**最后跟一行**结构化数据：

    [1] # 标题
    正文……

    @@DATA:{"query":"...","hits":[...]}

那行 `@@DATA:` 是给**证据层**用的（`EvidenceCollector::add_from_tool_data`
靠它把结构化结果解回来）。但：

- 人看的时候是噪音 —— 一坨 JSON 挤在正文后面；
- 喂给模型的时候也是噪音 —— 占 token 还干扰模型。

所以 Agent 循环里会在发给模型前剥掉它（`agent.rs:render_observation`），
但**直接调工具的 HTTP 端点**返回的是原始 observation。

## 用法

    curl -s ... | python3 scripts/strip_data.py

读 stdin 的 JSON，改写 `observation` 字段，输出到 stdout。
非 JSON 输入原样透传（不吞错误）。
"""

import json
import sys


def strip_data(text):
    """剥掉 observation 末尾的 `@@DATA:` 行。"""
    if not isinstance(text, str):
        return text
    lines = text.split("\n")
    # 只删**最后**一个 @@DATA: 段 —— 它总是追加在末尾。
    out = []
    for line in lines:
        if line.startswith("@@DATA:"):
            break
        out.append(line)
    # 去掉尾部空行（@@DATA 前面通常有一个空行）
    while out and out[-1].strip() == "":
        out.pop()
    return "\n".join(out)


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 不是 JSON —— 原样输出，不吞错误。
        sys.stdout.write(raw)
        return

    if isinstance(data, dict) and "observation" in data:
        data["observation"] = strip_data(data["observation"])

    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
