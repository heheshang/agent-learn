# ============================================================================
# 01-axum —— HTTP 服务基础（路由 / 处理器 / 状态 / 错误）
#
# 用法：
#   make help      查看所有目标
#   make run       启动服务（127.0.0.1:3000）
#   make smoke     起服务 → 跑一组 curl → 关服务
# ============================================================================

PKG      := axum-01
CARGO   ?= cargo
ADDR    ?= 127.0.0.1:3000

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean smoke

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 启动 HTTP 服务
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

smoke: build ## 端到端冒烟测试（自动起停服务）
	@echo "==> 启动服务 ..."
	@$(CARGO) run -p $(PKG) & echo $$! > /tmp/.axum01.pid; \
	sleep 2; \
	echo "==> GET /health"; curl -s http://$(ADDR)/health; echo; \
	echo "==> POST /users"; curl -s -X POST http://$(ADDR)/users \
		-H 'Content-Type: application/json' -d '{"name":"Rodriguez","age":30}'; echo; \
	echo "==> GET /users?page=1&size=2"; curl -s 'http://$(ADDR)/users?page=1&size=2'; echo; \
	echo "==> GET /users/1"; curl -s http://$(ADDR)/users/1; echo; \
	echo "==> DELETE /users/1"; curl -s -o /dev/null -w '%{http_code}\n' -X DELETE http://$(ADDR)/users/1; \
	kill `cat /tmp/.axum01.pid` 2>/dev/null; rm -f /tmp/.axum01.pid
