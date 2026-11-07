# ============================================================================
# 06-rag —— 手撕 Naive RAG 系统（检索 + 生成 + HTTP API + 自主 Agent）
#
# 用法：
#   make help          查看所有目标
#   make serve         启动 HTTP 服务（127.0.0.1:3000）
#   make demo          端到端 demo（不起服务）
#
# ⚠️ 需要 FAISS（brew install faiss）+ DEEPSEEK_API_KEY
# ============================================================================

PKG      := rag-06
CARGO   ?= cargo
ADDR    ?= 127.0.0.1:3000

.DEFAULT_GOAL := help
.PHONY: help build run serve demo test check fmt fmt-check lint clean \
        reindex ask search smoke

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

test: ## 跑测试（单元 + 集成）
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

reindex: ## 重建索引（删除旧索引文件后重新生成）
	rm -f index/index.faiss index/index.faiss.meta.json
	$(CARGO) run -p $(PKG) -- serve & echo $$! > /tmp/.rag06.pid; \
	sleep 8; curl -s -X POST http://$(ADDR)/api/reindex; echo; \
	kill `cat /tmp/.rag06.pid` 2>/dev/null; rm -f /tmp/.rag06.pid

smoke: build ## 冒烟测试：起服务 → ask → 关服务
	@echo "==> 启动服务 ..."
	@$(CARGO) run -p $(PKG) -- serve & echo $$! > /tmp/.rag06.pid; \
	sleep 8; \
	echo "==> POST /api/ask"; curl -s -X POST http://$(ADDR)/api/ask \
		-H 'Content-Type: application/json' -d '{"question":"什么是 RAG？"}'; echo; \
	kill `cat /tmp/.rag06.pid` 2>/dev/null; rm -f /tmp/.rag06.pid
