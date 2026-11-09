# ============================================================================
# 剖析工具镜像 —— 只装 py-spy
#
# 为什么要容器化（这是一个实测踩出来的坑）：
#
#   py-spy 官方文档明确写着「OSX always requires running as root」——
#   它靠 task_for_pid 读目标进程内存，macOS 的 SIP 不允许普通用户 attach
#   别的进程，**重签名（codesign 加 get-task-allow）也绕不过去**
#   （实测：重签 py-spy 自身 + 重签目标解释器，仍报 requires root）。
#
#   而 Linux 下 py-spy attach 自己 spawn 的子进程**不需要 root**，
#   只需要容器给 SYS_PTRACE 能力。
#
#   所以：**macOS 上要跑 py-spy，只能进容器**。
#
# ⚠️ 镜像里**只装 py-spy**，不装 scalene：
#
#   1. scalene 2.4.0 在 slim 镜像里没有预编译 wheel，pip 会退化成
#      源码编译并失败（实测报错 "Getting requirements to build wheel
#      did not run successfully"），把镜像构建拖到几分钟还失败。
#   2. scalene **不需要容器** —— 它是「插桩运行」型剖析器，
#      在宿主机 venv 里就能跑，能力完全不受影响。
#
#   所以分工是：**py-spy 进容器（因为 macOS attach 限制），
#   cProfile / scalene 留在宿主机 venv（因为根本不需要容器）**。
#
# 构建：
#   docker build -t agent-perf-tools -f docker/Dockerfile .
# ============================================================================
FROM python:3.12-slim

# py-spy：采样剖析器（attach 模式，需 SYS_PTRACE）
#
# ⚠️ 只装 py-spy，不装 scalene（理由见文件头）。
RUN pip install --no-cache-dir py-spy==0.4.3

WORKDIR /work

# 默认跑一个探活脚本，避免 `docker run` 无参数时打印帮助
CMD ["python3", "-c", "print('agent-perf-tools ready')"]
