# ============================================================================
# 04-rag —— RAG Part 1：文档加载 + 文本分块
#
# 用法：
#   make help          查看所有目标
#   make demo          跑全部 demo（load / split / compare / pipeline）
#   make compare       对比三种 splitter 的效果差异
# ============================================================================

PKG      := rag-04
CARGO   ?= cargo

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean \
        demo load split compare pipeline all

## ── 通用 ────────────────────────────────────────────────────────────────

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 运行（默认 all demo）
	$(CARGO) run -p $(PKG)

test: ## 跑单元测试
	$(CARGO) test -p $(PKG)

check: ## 编译检查（不产出二进制）
	$(CARGO) check -p $(PKG)

fmt: ## 格式化
	$(CARGO) fmt -p $(PKG)

fmt-check: ## 检查格式
	$(CARGO) fmt -p $(PKG) -- --check

lint: ## clippy 静态检查
	$(CARGO) clippy -p $(PKG) --all-targets -- -D warnings

clean: ## 清理构建产物
	$(CARGO) clean -p $(PKG)

## ── Demo ────────────────────────────────────────────────────────────────

demo: all ## 跑全部 demo

load: build ## 对比三种 loader 的输出
	$(CARGO) run -p $(PKG) -- load

split: build ## 对比三种 splitter 的输出
	$(CARGO) run -p $(PKG) -- split

compare: build ## 量化不同策略的差异
	$(CARGO) run -p $(PKG) -- compare

pipeline: build ## 全流程：加载 → 分割 → 统计
	$(CARGO) run -p $(PKG) -- pipeline

all: build ## load + split + compare + pipeline
	$(CARGO) run -p $(PKG) -- all
