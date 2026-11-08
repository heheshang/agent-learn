#!/usr/bin/env bash
# ============================================================================
# 校验所有 README 里的 mermaid 代码块能被 mermaid-cli 正确渲染。
#
# 用途：mermaid 语法错误**不会**让 markdown 报错，只会在 GitHub / VSCode 上
# 显示成一段红字。所以必须靠工具验证。
#
# 用法：
#   scripts/check-mermaid.sh          # 校验全部 README
#   scripts/check-mermaid.sh <file>   # 校验指定文件
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! command -v mmdc >/dev/null 2>&1; then
  echo "✗ 找不到 mmdc（mermaid-cli）。安装：npm i -g @mermaid-js/mermaid-cli"
  exit 1
fi

# ── Puppeteer 配置 ──────────────────────────────────────────────────────────
#
# ⚠️ 为什么需要它：mmdc 默认的 protocolTimeout 只有 30s，
#    而串行渲染几十张图时 Chromium 复用同一个会话，内存压力会让
#    Runtime.callFunctionOn 随机超时 —— 表现为「这张图渲染失败」，
#    但同一张图单独重跑又完全正常。
#
#    也就是说：那不是 mermaid 语法错误，是渲染器 flaky。
#    提高 protocolTimeout + 加大 /dev/shm 可以大幅降低误报。
PUPPETEER_CONFIG="$ROOT/scripts/puppeteer-config.json"
MMDC_ARGS=()
if [ -f "$PUPPETEER_CONFIG" ]; then
  MMDC_ARGS+=(--puppeteerConfigFile "$PUPPETEER_CONFIG")
fi

# 重试次数：只有「超时类」错误才重试（语法错误重试也没用）。
RETRIES=2

# 收集要校验的文件。
# ⚠️ 用 while read 而不是 mapfile —— macOS 自带的是 bash 3.2，没有 mapfile。
if [ $# -gt 0 ]; then
  FILES=()
  while IFS= read -r line; do
    [ -n "$line" ] && FILES+=("$line")
  done <<< "$*"
else
  FILES=()
  while IFS= read -r line; do
    [ -n "$line" ] && FILES+=("$line")
  done < <(
    cd "$ROOT" && ls README.md 0*/README.md 1*/README.md 2>/dev/null
  )
fi

TOTAL=0
FAILED=0
FAILED_FILES=()

for f in "${FILES[@]}"; do
  path="$ROOT/$f"
  [ -f "$path" ] || continue

  # ⚠️ 每个文件必须用**唯一前缀** —— 否则多个 README 都叫 README.md，
  #    临时文件会互相覆盖，导致「校验通过」但其实漏掉了大部分图。
  slug=$(printf '%s' "$f" | tr '/' '_')

  # 提取所有 ```mermaid 块，逐块写文件。
  python3 - "$path" "$TMP" "$slug" <<'PY'
import re, sys, pathlib
src, tmp, slug = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
text = src.read_text(encoding="utf-8")
blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
for i, b in enumerate(blocks):
    (tmp / f"{slug}.{i}.mmd").write_text(b, encoding="utf-8")
print(len(blocks))
PY

  n=$(ls "$TMP"/"$slug".*.mmd 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && continue

  echo "── $f ($n 张图)"
  for mmd in "$TMP"/"$slug".*.mmd; do
    TOTAL=$((TOTAL + 1))
    idx=$(basename "$mmd" | sed 's/.*\.\([0-9]*\)\.mmd/\1/')

    # ── 重试逻辑 ────────────────────────────────────────────────────────────
    # 只有「超时 / 浏览器崩溃」这类**基础设施类**错误才值得重试。
    # 真正的语法错误（ParseError / Expecting ...）重试多少次都一样。
    ok=0
    out=""
    for attempt in $(seq 0 $RETRIES); do
      if out=$(mmdc "${MMDC_ARGS[@]}" -i "$mmd" -o "$TMP/out.png" 2>&1); then
        ok=1
        break
      fi
      # 语法类错误 —— 直接判失败，不重试。
      if echo "$out" | grep -qiE "ParseError|Expecting|no viable alternative|error:" ; then
        if echo "$out" | grep -qiE "timed out|ProtocolError|Target closed|SIGSEGV|ENOMEM"; then
          : # 同时命中错误与超时 —— 仍按超时处理
        else
          break
        fi
      fi
      [ "$attempt" -lt "$RETRIES" ] && sleep 2
    done

    if [ "$ok" -eq 1 ]; then
      printf '  ✓ 图 %s\n' "$idx"
    else
      FAILED=$((FAILED + 1))
      FAILED_FILES+=("$f#图$idx")
      printf '  ✗ 图 %s 渲染失败\n' "$idx"
      echo "$out" | grep -iE "error|expecting|got|timed out" | head -5 | sed 's/^/      /'
      # 打印出错的图，便于定位。
      sed 's/^/      | /' "$mmd" | head -20
    fi
  done
done

echo
echo "═══════════════════════════════════════"
if [ "$FAILED" -eq 0 ]; then
  echo "✓ 全部 $TOTAL 张图渲染通过"
  exit 0
else
  echo "✗ $FAILED / $TOTAL 张图渲染失败："
  for x in "${FAILED_FILES[@]}"; do echo "    - $x"; done
  exit 1
fi
