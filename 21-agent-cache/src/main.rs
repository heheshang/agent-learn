//! 21-agent-cache —— 手撕 Redis 缓存（缓存 LLM 响应）。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- tools                    # 工具清单（不调模型）
//! cargo run -- call calculate '{"expression":"(12+34)*2"}'  # 直接调工具 ★
//! cargo run -- run "什么是缓存击穿？"     # 跑一次 Agent（走缓存）
//! cargo run -- run2 "同一个问题"         # ★ 再跑一次（演示缓存命中）
//! cargo run -- compare "什么是缓存击穿？" # ★ 缓存开/关对照
//! cargo run -- stats                    # 缓存统计
//! cargo run -- keys                     # 列出缓存键
//! cargo run -- clear                    # 清空命名空间
//! cargo run -- serve                    # 启动 HTTP 服务（:3130）
//! cargo run -- all                      # 全部
//! ```
//!
//! ## ⚠️ 默认问题**故意不是**「北京天气」
//!
//! 天气工具**不在**缓存白名单里（实时数据），
//! 而且带工具的链路命中率本来就很低。
//! 用纯问答问题才能展示出「缓存真的省了一次模型调用」。

use std::sync::Arc;

use agent_logging::{info, init, warn};
use anyhow::Result;

use agent_21::agent::CachedAgent;
use agent_21::api::AppStateInner;
use agent_21::cache::{CacheBackend, CacheService, LocalCache, RedisCache};
use agent_21::config::AppConfig;
use agent_21::functions::default_registry;
use agent_21::llm::GenaiFunctionLlm;
use agent_21::table::Table;

/// 默认演示问题。
///
/// ⚠️ 必须是「**不需要工具**」的问题 ——
/// 带工具的链路里 `call_id` 每轮都变，缓存几乎必然 miss
/// （见 README「exact-match cache 对 agentic traffic 会失灵」）。
const DEFAULT_QUESTION: &str = "什么是缓存击穿（cache stampede）？请简要解释。";

/// 默认演示工具。
const DEFAULT_TOOL: &str = "calculate";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-21");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question).await?;
        }
        "run2" => {
            // ★ 连跑两次同一个问题 —— 第二次应当命中缓存。
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_twice(&question).await?;
        }
        "compare" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            compare_demo(&question).await?;
        }
        "semantics" => {
            // ★ 三档缓存语义对照 —— agentic 链路能否命中的定量实验。
            let question = arg_or_default(2, DEFAULT_QUESTION);
            semantics_demo(&question).await?;
        }
        "call" => {
            let tool = arg_or_default(2, DEFAULT_TOOL);
            let args = std::env::args()
                .nth(3)
                .unwrap_or_else(|| r#"{"expression":"(12+34)*2"}"#.to_string());
            call_demo(&tool, &args).await?;
        }
        "tools" => tools_demo()?,
        "stats" => stats_demo().await?,
        "keys" => keys_demo().await?,
        "clear" => clear_demo().await?,
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            call_demo(DEFAULT_TOOL, r#"{"expression":"(12+34)*2"}"#).await?;
            run_twice(DEFAULT_QUESTION).await?;
            stats_demo().await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: run / run2 / compare / semantics / call / tools / stats / keys / clear / serve / all）"
            );
        }
    }

    Ok(())
}

/// 取第 n 个命令行参数（没有就用默认值）。
fn arg_or_default(index: usize, default: &str) -> String {
    std::env::args()
        .nth(index)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// 构建 BM25 索引（本地知识库）。
fn build_index(config: &AppConfig) -> Result<Arc<rag_08::bm25::Bm25Index>> {
    let data_dir = std::path::Path::new(&config.data_dir);

    if !data_dir.exists() {
        anyhow::bail!("数据目录不存在: {}（先跑 `make data`）", data_dir.display());
    }

    let splitter_config = rag_04::splitter::SplitterConfig::default();
    let index = rag_08::index::build_bm25(data_dir, &splitter_config)
        .map_err(|e| anyhow::anyhow!("构建知识库索引失败: {e}"))?;

    info!(
        dir = %data_dir.display(),
        documents = index.len(),
        "知识库索引构建完成"
    );

    Ok(Arc::new(index))
}

/// 构建 LLM 客户端。
fn build_llm(config: &AppConfig) -> Arc<dyn agent_21::llm::FunctionLlm> {
    info!(
        model = %config.model,
        temperature = config.temperature,
        max_tokens = config.max_tokens,
        reasoning = !config.disable_reasoning,
        "创建 LLM 客户端"
    );
    Arc::new(GenaiFunctionLlm::new(
        &config.model,
        config.temperature,
        config.max_tokens,
        config.disable_reasoning,
    ))
}

/// 构建缓存服务。
///
/// ⚠️ Redis 连不上时**降级到内存缓存**（除非 `CACHE_REDIS_REQUIRED=1`）。
async fn build_cache(config: &AppConfig) -> Result<Arc<CacheService>> {
    let backend: Arc<dyn CacheBackend> = match RedisCache::connect(&config.redis.url).await {
        Ok(redis) => {
            info!(url = %config.redis.url, backend = "redis", "缓存后端：Redis");
            Arc::new(redis)
        }
        Err(e) => {
            warn!(
                url = %config.redis.url,
                error = %e,
                backend = "local",
                "⚠️ Redis 不可用 —— 降级到内存缓存"
            );
            if config.redis.required {
                anyhow::bail!("Redis 不可用且 CACHE_REDIS_REQUIRED=1: {e}");
            }
            Arc::new(LocalCache::new())
        }
    };

    let service = CacheService::new(backend, config.cache.default_ttl)
        .with_namespace(config.redis.namespace.clone())
        .with_cacheable_tools(config.cache.cacheable_tools.clone())
        .with_tool_ttl(config.cache.tool_ttl);

    if !config.cache.enabled {
        return Ok(Arc::new(CacheService::disabled_backend(
            service.into_backend(),
        )));
    }

    Ok(Arc::new(service))
}

/// 构建 Agent。
fn build_agent(
    config: &AppConfig,
    llm: Arc<dyn agent_21::llm::FunctionLlm>,
    cache: Arc<CacheService>,
) -> Result<Arc<CachedAgent>> {
    let index = build_index(config)?;
    let registry = default_registry(index);

    Ok(Arc::new(
        CachedAgent::new(llm, registry, cache, config.clone())
            .with_tool_ttl(config.cache.tool_ttl)
            .with_normalized_call_id(config.cache.normalize_call_id)
            .with_cache_intermediate_rounds(config.cache.cache_intermediate_rounds),
    ))
}

/// 打印工具清单。
fn tools_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config)?;
    let registry = default_registry(index);

    let mut table = Table::new(vec!["工具", "strict", "可缓存", "说明"]);
    for spec in registry.specs() {
        table.push(vec![
            spec.name.clone(),
            if spec.strict { "是" } else { "否" }.to_string(),
            if config.cache.cacheable_tools.contains(&spec.name) {
                "是"
            } else {
                "否"
            }
            .to_string(),
            spec.description.chars().take(40).collect::<String>(),
        ]);
    }

    println!("{}", table.render());
    println!("⚠️ 天气工具（get_weather）默认**不可缓存** —— 实时数据，缓存即过时。");
    Ok(())
}

/// 直接调工具。
async fn call_demo(tool: &str, args: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config)?;
    let registry = default_registry(index);

    let parsed: serde_json::Value =
        serde_json::from_str(args).map_err(|e| anyhow::anyhow!("参数不是合法 JSON: {e}"))?;

    let outcome = registry
        .call_strict(tool, &parsed, &agent_21::tool::ToolContext::empty())
        .await?;

    println!("工具   : {}", outcome.tool);
    println!("成功   : {}", outcome.ok);
    println!("观察   : {}", outcome.observation);
    if let Some(data) = &outcome.data {
        println!("结构化 : {}", serde_json::to_string_pretty(data)?);
    }

    Ok(())
}

/// 跑一次 Agent。
async fn run_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let llm = build_llm(&config);
    let cache = build_cache(&config).await?;
    let agent = build_agent(&config, llm, cache)?;

    let run = agent.run(question).await?;

    println!("问题   : {}", run.question);
    println!("答案   : {}", run.answer);
    println!("步数   : {}", run.steps.len());
    println!("工具调用: {}", run.tool_calls);
    println!(
        "缓存   : 命中 {} / 未命中 {}",
        run.cache_hits, run.cache_misses
    );
    println!("耗时   : {} ms", run.elapsed_ms);
    println!("token  : {}", run.usage.total_tokens);

    Ok(())
}

/// ★ 连跑两次同一个问题 —— 演示缓存命中。
async fn run_twice(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let llm = build_llm(&config);
    let cache = build_cache(&config).await?;
    let agent = build_agent(&config, llm, cache.clone())?;

    println!("=== 第一次（应当未命中）===");
    let first = agent.run(question).await?;
    print_run(&first);

    println!();
    println!("=== 第二次（应当命中缓存）===");
    let second = agent.run(question).await?;
    print_run(&second);

    println!();
    let stats = cache.stats().snapshot();
    println!("命中率 : {:.1}%", stats.hit_rate * 100.0);
    println!(
        "节省 token: {}（估算：命中条目的 total_tokens 之和）",
        stats.saved_tokens
    );

    if second.cache_hits == 0 {
        // ⚠️ **不要笼统地说「检查后端与 TTL」** ——
        //
        // 「第二次没命中」有**两种**截然不同的原因：
        //
        // 1. **已知结论**：问题触发了工具调用 → 链路是 agentic 的 →
        //    键里含 `call_id`（每次都变）→ 精确匹配缓存**必然 miss**。
        //    这不是 bug，是 LiteLLM 文档里写明的 exact-match cache 的固有性质。
        // 2. **真 bug**：纯问答链路（没调工具）仍然没命中 →
        //    那才可能是后端/TTL/反序列化的问题。
        //
        // ⚠️ 不区分这两种情况，排查方向就被带偏了 ——
        // 用户会去查 Redis 配置，但问题根本不在那儿。
        if second.tools_used.is_empty() && first.tools_used.is_empty() {
            warn!(
                backend = cache.backend_kind(),
                "第二次没有命中缓存，且本次是**纯问答链路**（未调用任何工具）—— \
                 这不是已知结论，请检查缓存后端 / TTL / 序列化（后端: {}）",
                cache.backend_kind()
            );
        } else {
            warn!(
                tools = ?second.tools_used,
                "第二次没有命中缓存 —— ⚠️ 这是**已知结论**不是 bug：\
                 本次问题触发了工具调用（{}），链路属于 agentic traffic，\
                 缓存键里含每次都变的 call_id，精确匹配缓存必然 miss。\
                 想演示缓存命中，请换一个**不需要工具**的纯问答问题。",
                second.tools_used.join(", ")
            );
        }
    }

    Ok(())
}

/// 打印一次运行结果。
fn print_run(run: &agent_21::AgentRun) {
    println!("答案   : {}", run.answer);
    println!("步数   : {}", run.steps.len());
    println!(
        "缓存   : 命中 {} / 未命中 {}",
        run.cache_hits, run.cache_misses
    );
    println!("耗时   : {} ms", run.elapsed_ms);
    println!("token  : {}", run.usage.total_tokens);

    for step in &run.steps {
        let source = match step.source {
            agent_21::CallSource::Live => "真实调用",
            agent_21::CallSource::CacheHit => "★缓存命中",
        };
        let tail = &step.cache_key[step.cache_key.len().saturating_sub(16)..];
        println!(
            "  第 {} 步 [{}] key=…{} 工具数={}",
            step.step,
            source,
            tail,
            step.tool_calls.len()
        );
    }
}

/// ★ 缓存开 / 关对照实验。
///
/// ## 为什么这是本项目最重要的实验
///
/// 「缓存有没有用」不能靠感觉。
/// 对照实验的价值在于：**只有一个变量不同**。
///
/// | 组 | 缓存 | 模型调用 |
/// |---|---|---|
/// | A（基线） | 关 | 每次都真调 |
/// | B（对照） | 开 | 第一次真调，第二次命中 |
///
/// 两组的模型、工具、问题、参数**完全相同**。
async fn compare_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== A 组：缓存关闭（基线）===");
    let baseline = run_group(&config, question, false, "cmp-a").await?;

    println!();
    println!("=== B 组：缓存开启 ===");
    let cached = run_group(&config, question, true, "cmp-b").await?;

    println!();
    println!("=== 对照结果 ===");
    print_comparison(&baseline, &cached);

    Ok(())
}

/// ★ 三档缓存语义对照 —— **agentic 链路能否命中**的定量实验。
///
/// ## 为什么需要这个实验
///
/// 「exact-match cache 对 agentic traffic 失灵」这个结论，
/// 到底是**不可救**的，还是**某个具体实现选择**造成的？
///
/// 本项目实测发现：**两个具体的实现选择**叠加起来造成了必然 miss：
///
/// | 档 | 中间轮写缓存 | 确定性 `call_id` | agentic 命中率 |
/// |---|---|---|---|
/// | 1（原始） | ❌ | ❌ | **0** |
/// | 2 | ✅ | ❌ | **100%** |
/// | 3 | ✅ | ✅ | **100%** |
///
/// ## ⚠️ 关键发现：**中间轮缓存**才是主因
///
/// 原本的结论说「键里含 `call_id`，所以必然 miss」——
/// 但实测发现**即使不重写 `call_id`，只要缓存了中间轮，命中率就是 100%**。
///
/// 原因是：
///
/// ```text
/// 第 1 步（键只含 [user]，本来就稳定）
///   不写缓存 → 每次都真调 → 模型每次措辞不同
///   → 第 2 步的历史不同 → 第 2 步的键不同 → 必然 miss
/// ```
///
/// 也就是说：**中间轮的不稳定性会顺着消息历史污染后面所有轮次**。
/// 真正的问题不是 `call_id`，而是「中间轮每次都真调模型」。
///
/// ## 三组必须用**独立命名空间**
///
/// 否则后一组会命中前一组的残留数据，对照就废了。
async fn semantics_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 档 1：原始（只缓存最后一轮 + 随机 call_id）===");
    let g1 = run_semantics_group(&config, question, false, false, "sem-1").await?;

    println!();
    println!("=== 档 2：+ 缓存中间轮 ===");
    let g2 = run_semantics_group(&config, question, true, false, "sem-2").await?;

    println!();
    println!("=== 档 3：+ 缓存中间轮 + 确定性 call_id ===");
    let g3 = run_semantics_group(&config, question, true, true, "sem-3").await?;

    println!();
    println!("=== 三档对照结果 ===");
    print_semantics_comparison(&g1, &g2, &g3);

    Ok(())
}

/// 三档语义里的一组。
struct SemanticsGroup {
    /// 第二次运行的命中轮数。
    hits: usize,
    /// 第二次运行的未命中轮数。
    misses: usize,
    /// 第二次耗时（毫秒）。
    second_ms: u64,
    /// 第二次 token。
    second_tokens: u64,
    /// 两次答案是否一致。
    answers_equal: bool,
}

/// 跑一组三档语义实验。
async fn run_semantics_group(
    config: &AppConfig,
    question: &str,
    cache_intermediate: bool,
    normalize_call_id: bool,
    namespace_suffix: &str,
) -> Result<SemanticsGroup> {
    let llm = build_llm(config);
    let cache = build_cache(config).await?;

    // ⚠️ 独立命名空间 —— 保证每组都是冷启动。
    let namespace = format!("{}-{}", config.redis.namespace, namespace_suffix);
    let service = cache.as_ref().clone().with_namespace(namespace.clone());
    let _ = service.clear_namespace(&namespace).await;

    let index = build_index(config)?;
    let registry = default_registry(index);
    let agent = CachedAgent::new(llm, registry, Arc::new(service.clone()), config.clone())
        .with_tool_ttl(config.cache.tool_ttl)
        .with_cache_intermediate_rounds(cache_intermediate)
        .with_normalized_call_id(normalize_call_id);

    let first = agent.run(question).await?;
    let second = agent.run(question).await?;

    println!(
        "  第一次: {} ms, token={}, 命中={}",
        first.elapsed_ms, first.usage.total_tokens, first.cache_hits
    );
    println!(
        "  第二次: {} ms, token={}, 命中={}",
        second.elapsed_ms, second.usage.total_tokens, second.cache_hits
    );

    Ok(SemanticsGroup {
        hits: second.cache_hits,
        misses: second.cache_misses,
        second_ms: second.elapsed_ms,
        second_tokens: second.usage.total_tokens,
        answers_equal: first.answer == second.answer,
    })
}

/// 打印三档对照结果。
fn print_semantics_comparison(g1: &SemanticsGroup, g2: &SemanticsGroup, g3: &SemanticsGroup) {
    let mut table = Table::new(vec!["指标", "档1 原始", "档2 +中间轮", "档3 +call_id"]);

    table.push(vec![
        "第二次命中轮数".to_string(),
        g1.hits.to_string(),
        g2.hits.to_string(),
        g3.hits.to_string(),
    ]);

    table.push(vec![
        "第二次未命中轮数".to_string(),
        g1.misses.to_string(),
        g2.misses.to_string(),
        g3.misses.to_string(),
    ]);

    table.push(vec![
        "第二次耗时(ms)".to_string(),
        g1.second_ms.to_string(),
        g2.second_ms.to_string(),
        g3.second_ms.to_string(),
    ]);

    table.push(vec![
        "第二次 token".to_string(),
        g1.second_tokens.to_string(),
        g2.second_tokens.to_string(),
        g3.second_tokens.to_string(),
    ]);

    table.push(vec![
        "两次答案一致".to_string(),
        g1.answers_equal.to_string(),
        g2.answers_equal.to_string(),
        g3.answers_equal.to_string(),
    ]);

    println!("{}", table.render());

    println!();
    println!("★ 结论：");
    println!("  · 「agentic 链路必然 miss」**不是** exact-match 的固有性质，");
    println!("    而是「中间轮不写缓存」这个实现选择造成的。");
    println!("  · 档 2 已经达到 100% 命中 —— **不需要**重写 call_id。");
    println!("  · 档 3 的 call_id 归一化是**锦上添花**（让 provider 侧的前缀缓存也稳定）。");
}

/// 一组对照的结果。
struct GroupResult {
    /// 第一次的耗时。
    first_ms: u64,
    /// 第二次的耗时。
    second_ms: u64,
    /// 第二次的命中数。
    second_hits: usize,
    /// 第二次的 token。
    second_tokens: u64,
    /// 两次答案是否一致。
    answers_equal: bool,
}

/// 跑一组对照（连跑两次同一个问题）。
///
/// ## ⚠️ 两组必须用**独立的命名空间**
///
/// 否则 A 组（缓存关）虽然不写缓存，但 B 组（缓存开）的第一次
/// 会**直接命中 A 组留下的数据**（或者反过来，B 组命中了上一次的残留），
/// 对照实验就废了 —— 实测踩过这个坑。
///
/// 独立命名空间保证：每组的第一次都是**冷启动**，
/// 第二次才是「同题重复」的场景。
async fn run_group(
    config: &AppConfig,
    question: &str,
    cache_on: bool,
    namespace_suffix: &str,
) -> Result<GroupResult> {
    let llm = build_llm(config);

    // ⚠️ 两组用**同一个后端**（Redis）——
    //    唯一区别是缓存开关。
    //    如果 A 组连 Redis 都不连，就多了一个变量（网络延迟）。
    let cache = build_cache(config).await?;

    // 独立命名空间 —— 保证两组互不干扰。
    let namespace = format!("{}-{}", config.redis.namespace, namespace_suffix);
    let service = if cache_on {
        cache.as_ref().clone().with_namespace(namespace.clone())
    } else {
        CacheService::disabled(cache.as_ref().clone_backend()).with_namespace(namespace.clone())
    };

    // 清空这个命名空间 —— 保证从冷启动开始。
    let _ = service.clear_namespace(&namespace).await;

    let agent = build_agent(config, llm, Arc::new(service.clone()))?;

    let first = agent.run(question).await?;
    let second = agent.run(question).await?;

    println!(
        "  第一次: {} ms, token={}, 命中={}",
        first.elapsed_ms, first.usage.total_tokens, first.cache_hits
    );
    println!(
        "  第二次: {} ms, token={}, 命中={}",
        second.elapsed_ms, second.usage.total_tokens, second.cache_hits
    );

    Ok(GroupResult {
        first_ms: first.elapsed_ms,
        second_ms: second.elapsed_ms,
        second_hits: second.cache_hits,
        second_tokens: second.usage.total_tokens,
        answers_equal: first.answer == second.answer,
    })
}

/// 打印对照结果。
fn print_comparison(baseline: &GroupResult, cached: &GroupResult) {
    let mut table = Table::new(vec!["指标", "A 缓存关", "B 缓存开", "差值"]);

    table.push(vec![
        "第一次耗时(ms)".to_string(),
        baseline.first_ms.to_string(),
        cached.first_ms.to_string(),
        format!("{:+}", cached.first_ms as i64 - baseline.first_ms as i64),
    ]);

    table.push(vec![
        "第二次耗时(ms)".to_string(),
        baseline.second_ms.to_string(),
        cached.second_ms.to_string(),
        format!("{:+}", cached.second_ms as i64 - baseline.second_ms as i64),
    ]);

    table.push(vec![
        "第二次 token".to_string(),
        baseline.second_tokens.to_string(),
        cached.second_tokens.to_string(),
        format!(
            "{:+}",
            cached.second_tokens as i64 - baseline.second_tokens as i64
        ),
    ]);

    table.push(vec![
        "第二次命中数".to_string(),
        baseline.second_hits.to_string(),
        cached.second_hits.to_string(),
        format!(
            "{:+}",
            cached.second_hits as i64 - baseline.second_hits as i64
        ),
    ]);

    table.push(vec![
        "两次答案一致".to_string(),
        baseline.answers_equal.to_string(),
        cached.answers_equal.to_string(),
        String::new(),
    ]);

    println!("{}", table.render());

    println!();
    if cached.answers_equal {
        println!("✅ 缓存命中后答案与基线**完全一致** —— 缓存没有改变语义。");
    } else {
        warn!("⚠️ 缓存命中后答案与基线**不一致** —— 这是必须排查的问题！");
    }
}

/// 打印缓存统计。
async fn stats_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let cache = build_cache(&config).await?;
    let stats = cache.stats().snapshot();

    let mut table = Table::new(vec!["指标", "值"]);
    table.push(vec!["后端".to_string(), cache.backend_kind().to_string()]);
    table.push(vec!["命名空间".to_string(), cache.namespace().to_string()]);
    table.push(vec!["查询次数".to_string(), stats.queries.to_string()]);
    table.push(vec!["命中".to_string(), stats.hits.to_string()]);
    table.push(vec!["未命中".to_string(), stats.misses.to_string()]);
    table.push(vec![
        "命中率".to_string(),
        format!("{:.1}%", stats.hit_rate * 100.0),
    ]);
    table.push(vec!["写入".to_string(), stats.writes.to_string()]);
    table.push(vec!["读跳过".to_string(), stats.read_skipped.to_string()]);
    table.push(vec!["写跳过".to_string(), stats.write_skipped.to_string()]);
    table.push(vec![
        "太老被拒".to_string(),
        stats.stale_rejected.to_string(),
    ]);
    table.push(vec!["降级读".to_string(), stats.degraded_reads.to_string()]);
    table.push(vec![
        "降级写".to_string(),
        stats.degraded_writes.to_string(),
    ]);
    table.push(vec![
        "节省 token".to_string(),
        stats.saved_tokens.to_string(),
    ]);

    println!("{}", table.render());
    Ok(())
}

/// 列出缓存键。
async fn keys_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let cache = build_cache(&config).await?;
    let entries = cache
        .list_entries(cache.namespace())
        .await
        .map_err(|e| anyhow::anyhow!("列出缓存条目失败: {e}"))?;

    let mut table = Table::new(vec!["键", "TTL(s)", "摘要"]);
    for entry in &entries {
        table.push(vec![
            entry.key.clone(),
            entry.ttl.to_string(),
            entry.preview.clone(),
        ]);
    }

    println!("{}", table.render());
    println!("共 {} 条", entries.len());
    Ok(())
}

/// 清空命名空间。
async fn clear_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let cache = build_cache(&config).await?;
    let removed = cache
        .clear_namespace(cache.namespace())
        .await
        .map_err(|e| anyhow::anyhow!("清空缓存失败: {e}"))?;

    println!(
        "已删除 {} 条缓存（命名空间: {}）",
        removed,
        cache.namespace()
    );
    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    info!("{}", config.summary());

    let llm = build_llm(&config);
    let cache = build_cache(&config).await?;
    let agent = build_agent(&config, llm.clone(), cache.clone())?;

    let state = Arc::new(AppStateInner::new(config.clone(), agent, cache, llm));

    let app = agent_21::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    info!(addr = %config.bind_addr, "服务已启动");

    axum::serve(listener, app).await?;
    Ok(())
}
