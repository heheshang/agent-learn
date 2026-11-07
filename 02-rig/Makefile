# ============================================================================
# 02-rig —— Rust 的 AI 编排框架（rig 0.46）
#
# 用法：
#   make help      查看所有目标
#   make run       跑全部 demo
#
# ⚠️ 需要环境变量 DEEPSEEK_API_KEY
# ============================================================================

PKG      := rig-02
CARGO   ?= cargo

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean key

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

key: ## 检查 DEEPSEEK_API_KEY 是否已设置
	@test -n "$$DEEPSEEK_API_KEY" \
		&& echo "✓ DEEPSEEK_API_KEY 已设置" \
		|| { echo "❌ 未设置 DEEPSEEK_API_KEY，请先 export DEEPSEEK_API_KEY=sk-..."; exit 1; }

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build key ## 运行全部 demo
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
