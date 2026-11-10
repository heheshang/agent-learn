# ============================================================================
# agent-guide-python 的 Docker 镜像
#
# 用 uv 的多阶段构建：builder 装依赖，runtime 只带运行时产物。
# ⚠️ 关键点：**不要把 .venv 拷到别的路径** —— venv 里的脚本 shebang 写死了
#    构建时的绝对路径，换路径就全失效（这个坑我们踩过一次）。
# ============================================================================

# ── 构建阶段 ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# ⚠️ 先只拷依赖声明，让「依赖没变」的层能被缓存。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ── 运行阶段 ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml ./

# ⚠️ 骨架阶段还没有 HTTP 服务入口（`agent_guide_python.app` 尚未实现），
#    所以默认 CMD 只是打印版本号；等 01-axum 的 FastAPI 服务落地后，
#    再换成 `uvicorn agent_guide_python.app:create_default_app --factory ...`。
CMD ["python", "-c", "import agent_guide_python; print('agent-guide-python', agent_guide_python.__version__)"]
