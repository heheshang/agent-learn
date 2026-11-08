# ============================================================================
# 10-rag-milvus —— 生产级向量数据库（Milvus）
#
# 用法：
#   make help          查看所有目标
#   make up            启动 Milvus + Qdrant（docker compose）★
#   make demo          端到端：建索引 + 稠密检索 + 服务端混合检索
#   make compare       Milvus ↔ Qdrant 同题对照 ★ 核心产出
#   make serve         启动 HTTP 服务（:3030）
#
# ⚠️ 需要 Docker（Milvus standalone = etcd + minio + milvus 三个容器）
# ⚠️ 首次跑会下载嵌入模型（约 470MB，缓存在 .fastembed_cache/）
# ============================================================================

PKG      := rag-10
CARGO   ?= cargo
COMPOSE ?= docker compose
DOCKER_DIR := docker

.DEFAULT_GOAL := help
.PHONY: help build run serve demo compare test test-integration check fmt fmt-check lint clean \
        up down down-v logs ps health reset-collection

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 运行（默认 serve）
	$(CARGO) run -p $(PKG)

serve: build ## 启动 HTTP 服务（:3030）
	$(CARGO) run -p $(PKG) -- serve

demo: build ## 端到端 demo：建索引 + 检索验证
	$(CARGO) run -p $(PKG) -- demo

compare: build ## Milvus ↔ Qdrant 同题对照 ★
	$(CARGO) run -p $(PKG) -- compare

test: ## 跑全部测试（单元 + 集成）
	$(CARGO) test -p $(PKG)

test-integration: ## 只跑集成测试（需要 Docker 已启动）
	RAG10_REQUIRE_BACKEND=1 $(CARGO) test -p $(PKG) --test integration -- --test-threads=1

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

# ── Docker 运维 ────────────────────────────────────────────────────────
up: ## 启动 Milvus standalone + Qdrant ★
	cd $(DOCKER_DIR) && $(COMPOSE) up -d
	@echo "等待 Milvus 就绪（首次启动约 60~90 秒）..."
	@cd $(DOCKER_DIR) && $(COMPOSE) ps

down: ## 停止容器（保留数据）
	cd $(DOCKER_DIR) && $(COMPOSE) down

down-v: ## 停止容器并删除数据卷（⚠️ 数据全没）
	cd $(DOCKER_DIR) && $(COMPOSE) down -v

logs: ## 跟踪日志
	cd $(DOCKER_DIR) && $(COMPOSE) logs -f

ps: ## 查看容器状态
	cd $(DOCKER_DIR) && $(COMPOSE) ps

health: ## 检查两个后端是否就绪
	@echo "--- Milvus ---"
	@curl -sf http://localhost:9091/healthz && echo " OK" || echo " ✗ 不可达"
	@echo "--- Qdrant ---"
	@curl -sf http://localhost:6333/readyz && echo " OK" || echo " ✗ 不可达"

reset-collection: ## 删掉测试用 collection（重置检索结果）
	@curl -sf -X POST http://localhost:19530/v2/vectordb/collections/drop \
		-H 'Content-Type: application/json' -d '{"collectionName":"rag_chunks"}' \
		&& echo "已删除 rag_chunks"
