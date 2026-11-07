# ============================================================================
# 05-rag —— RAG Part 2：向量化 + 向量存储
#
# 用法：
#   make help          查看所有目标
#   make demo          跑全部 demo
#   make compare       对比 Flat / IVF / HNSW 三种索引
#
# ⚠️ 需要系统装好 FAISS 的 C++ 库（macOS: brew install faiss）
# ============================================================================

PKG      := rag-05
CARGO   ?= cargo

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean deps \
        demo embed build-index search compare persist chroma all

## ── 通用 ────────────────────────────────────────────────────────────────

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

deps: ## 检查系统依赖（FAISS）
	@echo "==> 检查 libfaiss_c ..."
	@ldconfig -p 2>/dev/null | grep -q faiss \
		|| ls /opt/homebrew/lib/libfaiss_c* /usr/local/lib/libfaiss_c* >/dev/null 2>&1 \
		|| { echo "❌ 找不到 libfaiss_c，请先执行: brew install faiss"; exit 1; }
	@echo "✓ FAISS 就绪"

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 运行（默认 all demo）
	$(CARGO) run -p $(PKG)

test: ## 跑测试
	$(CARGO) test -p $(PKG)

check: ## 编译检查
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

embed: build ## 向量化：看向量长什么样
	$(CARGO) run -p $(PKG) -- embed

build-index: build ## 建索引：块 → 向量 → FAISS
	$(CARGO) run -p $(PKG) -- build

search: build ## 检索：FAISS 召回 top-k
	$(CARGO) run -p $(PKG) -- search

compare: build ## 对比 Flat / IVF / HNSW
	$(CARGO) run -p $(PKG) -- compare

persist: build ## 持久化：存盘再加载
	$(CARGO) run -p $(PKG) -- persist

chroma: build ## Chroma：写入 + 服务端元数据过滤（需先起 Chroma）
	$(CARGO) run -p $(PKG) -- chroma

all: build ## 跑全部 demo
	$(CARGO) run -p $(PKG) -- all
