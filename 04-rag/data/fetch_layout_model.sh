#!/usr/bin/env bash
# 下载 Docling 版面分析模型（`layout` feature 需要）。
#
# 用法：
#   ./data/fetch_layout_model.sh          # 下载到 04-rag/.models/
#   DOCLING_RS_MODELS_URL=... ./data/fetch_layout_model.sh   # 换镜像源
#
# 为什么需要它：
#   docling 的完整 PDF 管线用 ONNX 跑版面分析模型，
#   模型不下载下来 PDF 标题就没法被识别成 Markdown 标题。
#
# 模型来源：https://github.com/docling-project/docling.rs/releases/tag/models-v1
set -euo pipefail

BASE_URL="${DOCLING_RS_MODELS_URL:-https://github.com/docling-project/docling.rs/releases/download/models-v1}"
MODEL="layout_heron_int8.onnx"
TARGET_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/.models}"
TARGET="${TARGET_DIR}/${MODEL}"

mkdir -p "${TARGET_DIR}"

if [ -f "${TARGET}" ]; then
  echo "模型已存在: ${TARGET}"
  exit 0
fi

echo "下载 ${MODEL} 到 ${TARGET_DIR} ..."
curl -fsSL --retry 3 -o "${TARGET}" "${BASE_URL}/${MODEL}"

echo "完成: ${TARGET} ($(du -h "${TARGET}" | cut -f1))"
echo
echo "提示：还需要 OCR 模型才能处理扫描件/图片里的文字："
echo "  curl -fsSL https://raw.githubusercontent.com/docling-project/docling.rs/master/scripts/install/download_dependencies.sh | sh"
