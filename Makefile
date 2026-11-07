# ============================================================================
# agent-learn —— workspace 级统一入口
#
# 用法：
#   make help          查看所有目标
#   make build         编译全部
#   make test          跑全部测试
#   make lint          clippy 全量检查
#   make rag           RAG 主线一链跑（04 → 05 → 06）
#   make serve-07      启动 07-query 服务
# ============================================================================

CARGO := cargo
PKGS  := axum-01 rig-02 genai-03 rag-04 rag-05 rag-06 rag-07 rag-08

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean \
        01 02 03 04 05 06 07 08 \
        axum rig genai rag \
        serve-06 serve-07 serve-08 \
        demo-04 demo-05 demo-06 demo-07 demo-08 \
        compare-07 compare-08 \
        deps faiss chroma

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## ── Workspace 通用 ─────────────────────────────────────────────────────

build: ## 编译全部
	$(CARGO) build --workspace

run: ## 运行（未指定则跑 04-rag 的 all demo）
	$(CARGO) run -p rag-04

test: ## 跑全部测试
	$(CARGO) test --workspace

check: ## 编译检查
	$(CARGO) check --workspace

fmt: ## 格式化
	$(CARGO) fmt --all

fmt-check: ## 检查格式
	$(CARGO) fmt --all -- --check

lint: ## clippy 全量检查（零警告门槛）
	$(CARGO) clippy --workspace --all-targets -- -D warnings

clean: ## 清理构建产物
	$(CARGO) clean

## ── 单项目快捷入口（等价于进入子目录 make xxx）─────────────────────────

01: ## 01-axum：启动 HTTP 服务
	$(MAKE) -C 01-axum run

02: ## 02-rig：跑全部 demo
	$(MAKE) -C 02-rig run

03: ## 03-genai：跑全部 demo
	$(MAKE) -C 03-genai run

04: ## 04-rag：加载 + 分割全流程
	$(MAKE) -C 04-rag all

05: ## 05-rag：向量化 + 存储全流程
	$(MAKE) -C 05-rag all

06: ## 06-rag：端到端 demo
	$(MAKE) -C 06-rag demo

07: ## 07-query：端到端 demo
	$(MAKE) -C 07-query demo

08: ## 08-rag：端到端 demo
	$(MAKE) -C 08-rag demo

axum: 01 ## 别名
rig: 02 ## 别名
genai: 03 ## 别名
rag: 04 ## 别名

## ── 服务 ────────────────────────────────────────────────────────────────

serve-06: ## 启动 06-rag 服务（:3000）
	$(MAKE) -C 06-rag serve

serve-07: ## 启动 07-query 服务（:3010）
	$(MAKE) -C 07-query serve

serve-08: ## 启动 08-rag 服务（:3020）
	$(MAKE) -C 08-rag serve

## ── Demo ────────────────────────────────────────────────────────────────

demo-04: ## 04-rag：加载 + 分割
	$(MAKE) -C 04-rag all

demo-05: ## 05-rag：向量化 + 存储
	$(MAKE) -C 05-rag all

demo-06: ## 06-rag：Naive RAG 问答
	$(MAKE) -C 06-rag demo

demo-07: ## 07-query：查询改写
	$(MAKE) -C 07-query demo

demo-08: ## 08-rag：混合检索 + 重排
	$(MAKE) -C 08-rag demo

compare-07: ## 07-query：多策略改写对比 ★
	$(MAKE) -C 07-query compare

compare-08: ## 08-rag：多模式检索对比 ★
	$(MAKE) -C 08-rag compare

## ── 系统依赖 ────────────────────────────────────────────────────────────

deps: ## 检查系统依赖
	@echo "==> FAISS (05/06/07/08 需要)"; $(MAKE) -C 05-rag deps

faiss: ## 安装 FAISS（macOS）
	brew install faiss

chroma: ## 启动 Chroma 服务（05-rag 的 chroma demo 需要）
	docker run --rm -p 8765:8000 -v chroma-data:/chroma/chroma chromadb/chroma:latest
