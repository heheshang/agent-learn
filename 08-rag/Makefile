# ============================================================================
# 08-rag —— 手撕混合检索（Hybrid Search）+ 重排（Rerank）
#
# 用法：
#   make help          查看所有目标
#   make compare       多模式对比（vector / bm25 / hybrid）★ 核心产出
#
# ⚠️ 需要 FAISS（brew install faiss）+ DEEPSEEK_API_KEY
# ============================================================================

PKG      := rag-08
CARGO   ?= cargo
ADDR    ?= 127.0.0.1:3020

.DEFAULT_GOAL := help
.PHONY: help build run serve demo compare test check fmt fmt-check lint clean \
        reindex smoke

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 运行（默认 serve）
	$(CARGO) run -p $(PKG)

serve: build ## 启动 HTTP 服务
	$(CARGO) run -p $(PKG) -- serve

demo: build ## 端到端 demo（不起服务）
	$(CARGO) run -p $(PKG) -- demo

compare: build ## 多模式对比：vector / bm25 / hybrid
	$(CARGO) run -p $(PKG) -- compare

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

## ── 运维 ────────────────────────────────────────────────────────────────

reindex: ## 重建索引（FAISS + BM25）
	rm -f index/index.faiss index/index.faiss.meta.json
	$(CARGO) run -p $(PKG) -- serve & echo $$! > /tmp/.rag08.pid; \
	sleep 8; curl -s -X POST http://$(ADDR)/api/reindex; echo; \
	kill `cat /tmp/.rag08.pid` 2>/dev/null; rm -f /tmp/.rag08.pid

smoke: build ## 冒烟测试：起服务 → /api/compare → 关服务
	@echo "==> 启动服务 ..."
	@$(CARGO) run -p $(PKG) -- serve & echo $$! > /tmp/.rag08.pid; \
	sleep 8; \
	echo "==> POST /api/compare"; curl -s -X POST http://$(ADDR)/api/compare \
		-H 'Content-Type: application/json' \
		-d '{"question":"BM25 打分公式是什么","modes":["vector","bm25","hybrid"],"expected":["BM25 打分算法"]}'; echo; \
	kill `cat /tmp/.rag08.pid` 2>/dev/null; rm -f /tmp/.rag08.pid
