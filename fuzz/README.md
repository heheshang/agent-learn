# 模糊测试（libFuzzer）

本目录是**独立 Cargo workspace**，专用于 [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
的 libFuzzer 目标。

## 为什么它是独立 workspace

libFuzzer 需要 nightly 的 `-Z sanitizer`：

```text
-Zsanitizer=address
-Cllvm-args=-sanitizer-coverage-level=4
```

而主仓库的 `rust-toolchain.toml` 固定在稳定版 **1.99.1**。
两者不能共存于同一个 workspace，所以 `fuzz/Cargo.toml` 里有：

```toml
[workspace]   # ← 空节，这一行的存在让 cargo 把这里当成独立 workspace
```

⚠️ **少了这个空节会报**：

```text
error: current package believes it's in a workspace when it's not
```

## 准备工具链

```bash
cargo install cargo-fuzz --locked
rustup toolchain install nightly
```

## 五个目标

| 目标 | 被测模块 | 找什么 |
|---|---|---|
| `react_parser` | `agent_13::agent` 的解析器 | panic、解析出不可能的结构 |
| `sql_guard` | `agent_15::guard` | ⭐ **绕过**（不只 panic） |
| `schema_validator` | `agent_14::schema` | panic、错误放行 |
| `cache_key` | `agent_21` 的缓存键 | panic、键不稳定 |
| `tokenizer` | `rag_08::tokenize` | panic、词元异常 |

## 怎么跑

```bash
make fuzz-build       # 编译全部目标
make fuzz-run         # 每个目标跑 30 秒
make fuzz-list        # 列出目标

# 单个目标 + 自定义时长
cargo +nightly fuzz run sql_guard -- -max_total_time=300

# 用已有语料复现某个崩溃
cargo +nightly fuzz run sql_guard fuzz/artifacts/sql_guard/crash-xxx
```

## ⭐ `sql_guard` 特殊：不只找 panic，还找「绕过」

安全护栏的输入是**模型生成的任意 SQL**。只测「会不会 panic」是不够的 ——
护栏**不 panic 但放行了危险 SQL** 是更严重的问题。

所以这个目标的判定条件是：

```text
护栏放行（Ok）且 独立实现的扫描发现裸词危险关键字  →  判定为绕过
```

⚠️ **独立实现是必需的**：如果直接用护栏自己的 `tokenize_sql` 做对照，
就变成「用被测代码验证被测代码」——
护栏的词法分析有 bug 时，两边会**一起错**，检测不出来。

`fuzz_targets/sql_guard.rs` 里的对照实现是一个**刻意简化**的独立词法扫描器
（只认裸词，不处理注释/字符串的完整边界）。它的作用不是替代护栏，
而是提供**第二个视角**来发现分歧。

## 与属性测试的分工

| | 属性测试（proptest） | 模糊测试（libFuzzer） |
|---|---|---|
| 位置 | 各项目的 `tests/properties.rs` | 本目录 |
| 输入生成 | 按**语义**生成（合法 JSON、可读 SQL） | **任意字节序列** |
| 查什么 | 语义正确性（不变量） | 健壮性（panic / 绕过） |
| 工具链 | 稳定版（能进 CI） | nightly（不进 CI） |

两者**互补**：proptest 生成的输入「像真的」，能验证语义；
libFuzzer 生成的输入「不像真的」，能撞出没想过的边界。

## 语料

`fuzz/corpus/<target>/` 存着 libFuzzer 攒下的种子语料，
`fuzz/artifacts/` 存着崩溃复现文件。

⚠️ 这两个目录**都不入库**（见根 `.gitignore`）——
它们是**可再生**的（重跑 fuzz 会重新生成 / 修复后就成了废文件），
提交进去只会制造 diff 噪音。

真正需要版本化的是 `fuzz/fuzz_targets/*.rs`（源码）。
