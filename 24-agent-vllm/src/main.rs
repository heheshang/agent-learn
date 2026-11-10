//! CLI 入口。
//!
//! ```text
//! agent-24 <命令> [参数]
//!
//!   modes              ★ 打印两种批处理模式的说明
//!   bench              ★ ★ ★ 三组对照实验（模拟轨）
//!   bench-mode         ★ 实验 A：静态 vs 连续
//!   bench-seqs         ★ 实验 B：max_num_seqs 扫描
//!   bench-chunked      ★ 实验 C：chunked prefill 对照
//!   simulate           ★ 用调度器跑一次负载
//!   serve              启动 HTTP 服务（:3160）
//!   mock               启动 mock vLLM 服务（:8000，OpenAI 兼容）
//!   bench-vllm         ★ ★ ★ ★ 打真 vLLM 测吞吐（需要 vLLM 已启动）
//!   health             ★ 检查 vLLM 服务是否可达
//!   docker-up          ★ 启动 vLLM 容器（需要 NVIDIA GPU + docker）
//!   docker-down        停掉 vLLM 容器
//! ```

use std::process::ExitCode;

use agent_24::api::handlers::AppState;
use agent_24::api::router;
use agent_24::bench;
use agent_24::client::{BenchRequest, VllmClient};
use agent_24::config::AppConfig;
use agent_24::metrics::{Goodput, Slo};
use agent_24::scheduler::{BatchingMode, simulate};
use agent_24::server::{MockState, routes as mock_routes};
use agent_24::table::Table;
use agent_24::workload;
use agent_logging::info;
use futures::StreamExt;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");

    match command {
        "modes" => {
            print_modes();
            ExitCode::SUCCESS
        }
        "bench" => {
            let config = AppConfig::from_env();
            run_all_experiments(&config);
            ExitCode::SUCCESS
        }
        "bench-mode" => {
            let config = AppConfig::from_env();
            let result =
                bench::experiment_batching_mode(&config.workload, config.rounds, config.slo);
            print_experiment(&result);
            ExitCode::SUCCESS
        }
        "bench-seqs" => {
            let config = AppConfig::from_env();
            let result =
                bench::experiment_max_num_seqs(&config.workload, config.rounds, config.slo);
            print_experiment(&result);
            ExitCode::SUCCESS
        }
        "bench-chunked" => {
            let config = AppConfig::from_env();
            let result =
                bench::experiment_chunked_prefill(&config.workload, config.rounds, config.slo);
            print_experiment(&result);
            ExitCode::SUCCESS
        }
        "simulate" => {
            let config = AppConfig::from_env();
            run_simulate(&config);
            ExitCode::SUCCESS
        }
        "serve" => {
            let config = AppConfig::from_env();
            serve(config).await;
            ExitCode::SUCCESS
        }
        "mock" => {
            let config = AppConfig::from_env();
            serve_mock(config).await;
            ExitCode::SUCCESS
        }
        "health" => {
            let config = AppConfig::from_env();
            match check_health(&config).await {
                Ok(models) => {
                    println!("vLLM 可达，模型清单：");
                    for m in models {
                        println!("  - {m}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("vLLM 不可达：{e}");
                    eprintln!("提示：先跑 `make docker-up`（需要 NVIDIA GPU + docker）");
                    ExitCode::FAILURE
                }
            }
        }
        "bench-vllm" => {
            let config = AppConfig::from_env();
            match bench_real_vllm(&config).await {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("压测失败：{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "docker-up" => match run_docker("up") {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        },
        "docker-down" => match run_docker("down") {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            print_help();
            ExitCode::SUCCESS
        }
    }
}

/// 打印帮助。
fn print_help() {
    println!("agent-24 · 手撕批处理优化（vLLM）");
    println!();
    println!("用法：agent-24 <命令>");
    println!();
    println!("  modes          ★ 打印两种批处理模式的说明");
    println!("  bench          ★ ★ ★ 三组对照实验（模拟轨）");
    println!("  bench-mode     ★ 实验 A：静态批处理 vs 连续批处理");
    println!("  bench-seqs     ★ 实验 B：max_num_seqs 扫描");
    println!("  bench-chunked  ★ 实验 C：chunked prefill 对照");
    println!("  simulate       ★ 用调度器跑一次负载");
    println!("  serve          启动 HTTP 服务（:3160）");
    println!("  mock           启动 mock vLLM 服务（:8000，OpenAI 兼容）");
    println!("  health         ★ 检查 vLLM 服务是否可达");
    println!("  bench-vllm     ★ ★ ★ ★ 打真 vLLM 测吞吐");
    println!("  docker-up      启动 vLLM 容器（需要 NVIDIA GPU）");
    println!("  docker-down    停掉 vLLM 容器");
}

/// 打印两种批处理模式的说明。
fn print_modes() {
    println!("=== 静态批处理（static batching）===");
    println!("  做法  ：凑够一批再跑，跑完这一批才换下一批");
    println!("  优势  ：实现简单，一次前向的固定开销摊得最薄");
    println!("  劣势  ：★ 新请求要等**整个批**跑完才能被接纳");
    println!("         ★ 短请求被同批的长请求拖着（head-of-line blocking）");
    println!("  对应  ：HF generate / TGI 老版本默认行为");
    println!();
    println!("=== 连续批处理（continuous batching）===");
    println!("  做法  ：有空位就加入，谁先完成谁腾出位置");
    println!("  优势  ：★ 批永远更满 → 吞吐更高");
    println!("         ★ 排队等待更短");
    println!("  代价  ：TTFT 可能变差（要等位置而不是等整批）");
    println!("  对应  ：vLLM / SGLang / TensorRT-LLM / LMDeploy 的默认行为");
    println!();
    println!("=== 三个关键调参 ===");
    println!("  --max-num-seqs              同时处理的序列数上限（批的大小）");
    println!("  --max-num-batched-tokens    单个 forward 的 token 预算");
    println!("  --enable-chunked-prefill    长 prompt 切块，避免霸占整个 forward");
}

/// 跑一次模拟。
fn run_simulate(config: &AppConfig) {
    let arrivals = workload::generate(&config.workload);

    for mode in [BatchingMode::Static, BatchingMode::Continuous] {
        let sim = simulate(&config.engine, mode, &arrivals);
        let metrics = sim.metrics();
        let goodput = Goodput::from_outcomes(&sim.outcomes, sim.wall_ms, config.slo);

        println!();
        println!("=== {} ===", mode.label());
        println!("  总耗时          : {:.1} ms", sim.wall_ms);
        println!(
            "  请求吞吐        : {:.2} req/s",
            metrics.request_throughput_per_sec
        );
        println!(
            "  token 吞吐      : {:.1} tok/s",
            metrics.output_token_throughput_per_sec
        );
        println!(
            "  TTFT p50/p95/p99: {:.1} / {:.1} / {:.1} ms",
            metrics.ttft_p50_ms, metrics.ttft_p95_ms, metrics.ttft_p99_ms
        );
        println!(
            "  TPOT p50/p95    : {:.3} / {:.3} ms",
            metrics.tpot_p50_ms, metrics.tpot_p95_ms
        );
        println!("  平均排队等待    : {:.1} ms", metrics.avg_queue_wait_ms);
        println!(
            "  GOODPUT         : {:.2} req/s（达标率 {:.0}%，SLO: TTFT<={}ms TPOT<={}ms）",
            goodput.goodput_per_sec,
            goodput.satisfaction_rate * 100.0,
            goodput.slo.ttft_ms,
            goodput.slo.tpot_ms
        );
    }
}

/// 跑全部三组实验。
fn run_all_experiments(config: &AppConfig) {
    println!("{}", "=".repeat(78));
    println!("配置：");
    println!("{}", config.summary());
    println!("{}", "=".repeat(78));

    let a = bench::experiment_batching_mode(&config.workload, config.rounds, config.slo);
    print_experiment(&a);
    print_comparison(&a);

    let b = bench::experiment_max_num_seqs(&config.workload, config.rounds, config.slo);
    print_experiment(&b);
    print_comparison(&b);

    let c = bench::experiment_chunked_prefill(&config.workload, config.rounds, config.slo);
    print_experiment(&c);
    print_comparison(&c);
}

/// 打印一组实验的结果表。
fn print_experiment(result: &bench::ExperimentResult) {
    println!();
    println!("=== {} ===", result.name);

    let mut table = Table::new(&[
        result.param.as_str(),
        "请求吞吐",
        "token吞吐",
        "TTFT p50",
        "TTFT p95",
        "TPOT p50",
        "排队",
        "GOODPUT",
    ]);

    for v in &result.variants {
        table.row(&[
            v.value.as_str(),
            &format!("{:.2}", v.metrics.request_throughput_per_sec),
            &format!("{:.1}", v.metrics.output_token_throughput_per_sec),
            &format!("{:.1}", v.metrics.ttft_p50_ms),
            &format!("{:.1}", v.metrics.ttft_p95_ms),
            &format!("{:.3}", v.metrics.tpot_p50_ms),
            &format!("{:.1}", v.metrics.avg_queue_wait_ms),
            &format!("{:.2}", v.goodput.goodput_per_sec),
        ]);
    }

    println!("{}", table.render());
}

/// 打印相对基线的对比。
fn print_comparison(result: &bench::ExperimentResult) {
    let suite = bench::ExperimentSuite::new(vec![result.clone()], Slo::default());
    let comparisons = suite.compare_with_first(result);
    if comparisons.is_empty() {
        return;
    }

    println!("  相对「{}」的倍数：", result.variants[0].value);
    let mut table = Table::new(&["档位", "请求吞吐", "token吞吐", "TTFT p95", "GOODPUT"]);
    for c in &comparisons {
        table.row(&[
            c.variant_name.as_str(),
            &format!("{:.2}x", c.request_throughput_ratio),
            &format!("{:.2}x", c.token_throughput_ratio),
            &format!("{:.2}x", c.ttft_p95_ratio),
            &format!("{:.2}x", c.goodput_ratio),
        ]);
    }
    println!("{}", table.render());
    println!("  ⚠️ TTFT 倍数 > 1 表示**变差**（延迟上升）。");
}

/// 启动 HTTP 服务。
async fn serve(config: AppConfig) {
    let addr = config.bind_addr.clone();
    let state = AppState {
        config: config.clone(),
    };

    let app = router(state);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("绑定 {addr} 失败：{e}");
            return;
        }
    };

    info!(addr = %addr, "24-agent-vllm 服务已启动");
    println!("24-agent-vllm 服务已启动：http://{addr}");
    println!("  GET  /            服务信息");
    println!("  GET  /health      健康检查");
    println!("  GET  /api/modes   ★ 批处理模式说明");
    println!("  POST /api/simulate ★ 用调度器跑一次负载");
    println!("  POST /api/bench    ★ ★ ★ 三组对照实验");

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("服务异常退出：{e}");
    }
}

/// 启动 mock vLLM 服务。
async fn serve_mock(config: AppConfig) {
    // ⚠️ `base_url` 是给客户端用的**完整 URL**（含 `http://`），
    //    但 `TcpListener::bind` 要的是**纯地址**（`host:port`）。
    //    直接 bind 整个 URL 会报
    //    「nodename nor servname provided, or not known」。
    let addr = strip_scheme(&config.client.base_url);

    let state = MockState::new(BatchingMode::Continuous, config.engine);

    let app = mock_routes(state);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("绑定 {addr} 失败：{e}");
            return;
        }
    };

    info!(addr = %addr, "mock vLLM 已启动");
    println!("mock vLLM 已启动：http://{addr}");
    println!("  GET  /v1/models            模型清单");
    println!("  POST /v1/chat/completions  ★ OpenAI 兼容的对话接口");
    println!();
    println!("⚠️ 这是**模拟**服务，不是真 vLLM —— 它的延迟是模型算出来的。");
    println!("   用它验证客户端与指标口径可以，但**绝不能**拿它的绝对数字代表 vLLM。");

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("服务异常退出：{e}");
    }
}

/// 检查 vLLM 是否可达。
async fn check_health(config: &AppConfig) -> Result<Vec<String>, String> {
    let client = VllmClient::new(config.client.clone()).map_err(|e| e.to_string())?;
    client.health().await.map_err(|e| e.to_string())
}

/// ★ 打真 vLLM 测吞吐。
async fn bench_real_vllm(config: &AppConfig) -> Result<(), String> {
    let client = VllmClient::new(config.client.clone()).map_err(|e| e.to_string())?;

    // 先确认服务可达 —— 否则报错很难懂。
    let models = client.health().await.map_err(|e| {
        format!("{e}\n提示：先启动 vLLM（`make docker-up`），或用 `make mock` 起一个模拟服务")
    })?;
    println!("vLLM 可达，模型清单：{models:?}");
    println!();

    let requests: Vec<BenchRequest> = (0..config.workload.num_requests)
        .map(|i| BenchRequest {
            id: i,
            prompt: format!("请用一句话解释什么是批处理优化。第 {} 号请求。", i),
            max_tokens: config.workload.max_output_tokens,
        })
        .collect();

    // ⚠️ 客户端并发是**显式变量** ——
    //    并发 1 的话服务端永远攒不出批（坑 2）。
    let concurrency = config.client.concurrency;
    println!(
        "并发度 {}　请求条数 {}　每条最多 {} token",
        concurrency,
        requests.len(),
        config.workload.max_output_tokens
    );
    println!();

    let started = std::time::Instant::now();
    let mut responses = Vec::with_capacity(requests.len());

    // ⚠️ 用 `buffer_unordered` 限流并发 ——
    //    `tokio::spawn` 要求 future 是 `'static`，而这里要借用 `&client`。
    let stream = futures::stream::iter(requests.into_iter().map(|r| {
        let client = client.clone();
        async move { client.chat(&r).await }
    }))
    .buffer_unordered(concurrency);

    futures::pin_mut!(stream);

    while let Some(response) = stream.next().await {
        responses.push(response);
    }

    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 汇总度量。
    let ok: Vec<_> = responses.iter().filter(|r| r.ok).collect();
    let failed = responses.len() - ok.len();
    let total_tokens: usize = ok.iter().map(|r| r.completion_tokens).sum();
    let latencies: Vec<f64> = ok.iter().map(|r| r.elapsed_ms).collect();
    let total_prompt_tokens: usize = ok.iter().map(|r| r.prompt_tokens).sum();

    let secs = wall_ms / 1000.0;
    let request_throughput = ok.len() as f64 / secs.max(f64::EPSILON);
    let token_throughput = total_tokens as f64 / secs.max(f64::EPSILON);

    println!("=== 真 vLLM 压测结果 ===");
    println!("  请求数        : {}（失败 {}）", ok.len(), failed);
    println!("  总耗时        : {:.1} ms", wall_ms);
    println!("  请求吞吐      : {:.2} req/s", request_throughput);
    println!("  token 吞吐    : {:.1} tok/s", token_throughput);
    println!(
        "  prompt token  : {}　生成 token：{}",
        total_prompt_tokens, total_tokens
    );
    println!(
        "  端到端 p50/p95: {:.1} / {:.1} ms",
        agent_24::metrics::percentile(&latencies, 50),
        agent_24::metrics::percentile(&latencies, 95)
    );

    if failed > 0 {
        println!();
        println!("  ⚠️ 有 {} 条失败，前 3 条错误：", failed);
        for r in responses.iter().filter(|r| !r.ok).take(3) {
            println!(
                "    #{}: {}",
                r.id,
                r.error.as_deref().unwrap_or("未知错误")
            );
        }
    }

    Ok(())
}

/// 去掉 URL 的 scheme（`http://` / `https://`），只留 `host:port`。
///
/// ⚠️ `TcpListener::bind` 要的是纯地址；
/// 传 `http://127.0.0.1:8000` 会报
/// 「nodename nor servname provided, or not known」。
fn strip_scheme(url: &str) -> String {
    url.split_once("://")
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| url.to_string())
}

/// 启停 vLLM 容器。
fn run_docker(action: &str) -> Result<(), String> {
    let script = match action {
        "up" => "docker/docker-compose.yml",
        _ => "docker/docker-compose.yml",
    };

    let cmd = format!(
        "docker compose -f {script} {}",
        if action == "up" { "up -d" } else { "down" }
    );

    println!("执行：{cmd}");
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .status()
        .map_err(|e| format!("执行 docker 失败：{e}"))?;

    if !status.success() {
        return Err(format!("docker 命令退出码 {:?}", status.code()));
    }

    if action == "up" {
        println!();
        println!("vLLM 容器已启动。等待模型就绪（首次要下载模型，可能几分钟）：");
        println!("  docker compose -f {script} logs -f");
        println!();
        println!("就绪后可以跑：");
        println!("  make health-vllm    # 检查服务");
        println!("  make bench-vllm     # ★ 压测吞吐");
    } else {
        println!("vLLM 容器已停止。");
    }

    Ok(())
}
