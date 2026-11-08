# ============================================================================
# 09-rag-eval —— RAG 评估与优化前后对比
#
# 用法：
#   make help          查看所有目标
#   make eval          跑完整评估（多变体对比）★ 核心产出
#   make compare       只跑「优化前 vs 优化后」对比 ★
#   make demo          快速演示：跑一条评估集，打印详细结果
#   make serve         启动 HTTP 服务（:3030）
#
# ⚠️ 需要 DEEPSEEK_API_KEY（生成 + judge 都要调模型）
# ⚠️ 需要 FAISS（brew install faiss）
# ============================================================================

PKG      := rag-09
CARGO   ?= cargo

.DEFAULT_GOAL := help
.PHONY: help build run serve demo eval compare test check fmt fmt-check lint clean \
        dataset-check

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 运行（默认 eval）
	$(CARGO) run -p $(PKG) -- eval

serve: build ## 启动 HTTP 服务（:3030）
	$(CARGO) run -p $(PKG) -- serve

demo: build ## 快速演示：跑一条评估集
	$(CARGO) run -p $(PKG) -- demo

eval: build ## 完整评估（所有变体）
	$(CARGO) run -p $(PKG) -- eval

compare: build ## 优化前后对比（vector vs hybrid）★
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

dataset-check: ## 检查评估集标注是否有效（不发网络请求）
	@$(CARGO) run -p $(PKG) -- demo 2>&1 | head -30
