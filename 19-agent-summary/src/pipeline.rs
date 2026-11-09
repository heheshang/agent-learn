//! 流水线 —— 把「研究」和「成文」串起来。
//!
//! # 为什么分成两个阶段
//!
//! ```text
//! 阶段 A：研究（Agent 循环，多轮工具调用）
//!   └─ 产出：证据集合（E1、E2…）
//! 阶段 B：成文（一次纯文本调用）
//!   └─ 产出：结构化 JSON（带证据引用）
//! 阶段 C：校验（代码）
//!   └─ 产出：grounding 状态
//! ```
//!
//! ## 为什么不能「一轮搞定」
//!
//! 如果在一个提示词里同时要求「查资料」和「写报告」，
//! 模型会倾向于**先写**（写作的 token 收益更高），
//! 然后**顺手编几个引用编号** ——
//! 这正是 15 踩过的「模型不查库就作答」的坑。
//!
//! 分成两阶段后，写报告时的输入**只有证据**，
//! 模型想编也编不出细节（它没看到原始资料）。
//!
//! ## 为什么阶段 B 不带工具
//!
//! 带工具的话，模型可能又想调工具 —— 然后**又回到阶段 A**。
//! 阶段 B 传**空 tools 列表**，它就只能写。

use std::sync::Arc;

use agent_logging::{debug, info, warn};

use crate::agent::{Agent, AgentRun};
use crate::error::{AppError, AppResult};
use crate::llm::{FunctionLlm, generate_text};
use crate::prompt;
use crate::report::{ProjectPlan, WeeklySummary, parse_project_plan, parse_weekly_summary};

/// 生成周报的结果。
#[derive(Debug, Clone)]
pub struct WeeklyResult {
    /// 结构化周报。
    pub summary: WeeklySummary,

    /// 渲染好的 Markdown。
    pub markdown: String,

    /// 研究阶段的轨迹。
    pub run: AgentRun,
}

/// 生成项目方案的结果。
#[derive(Debug, Clone)]
pub struct PlanResult {
    /// 结构化方案。
    pub plan: ProjectPlan,

    /// 渲染好的 Markdown。
    pub markdown: String,

    /// 研究阶段的轨迹。
    pub run: AgentRun,
}

/// 周报流水线。
pub struct WeeklyPipeline {
    agent: Arc<Agent>,
    llm: Arc<dyn FunctionLlm>,
    max_evidence_per_section: usize,
}

impl std::fmt::Debug for WeeklyPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeeklyPipeline")
            .field("max_evidence", &self.max_evidence_per_section)
            .finish_non_exhaustive()
    }
}

impl WeeklyPipeline {
    /// 创建流水线。
    pub fn new(
        agent: Arc<Agent>,
        llm: Arc<dyn FunctionLlm>,
        max_evidence_per_section: usize,
    ) -> Self {
        Self {
            agent,
            llm,
            max_evidence_per_section,
        }
    }

    /// 跑一次完整流程。
    ///
    /// ## ⚠️ 阶段 A 失败的处理
    ///
    /// 致命错误（鉴权失败等）→ 直接往上抛（`Err`）。
    /// 因为**没有证据就没有周报** —— 硬写一份等于编造。
    ///
    /// 但**非致命**的降级（个别工具失败）→ 继续走，
    /// 只是证据少一些，grounding 状态会体现出来。
    pub async fn run(&self, question: &str) -> AppResult<WeeklyResult> {
        // ── 阶段 A：研究 ────────────────────────────────────────
        let system = prompt::research_system();
        let (run, collector) = self.agent.research(question, &system).await?;

        info!(
            evidence = collector.len(),
            tools = run.tools_used.len(),
            "研究阶段完成"
        );

        // ── 阶段 B：成文 ────────────────────────────────────────
        let user = prompt::weekly_prompt(question, &collector, self.max_evidence_per_section);
        let raw = generate_text(self.llm.as_ref(), "你是资深技术负责人，负责写周报。", &user)
            .await
            .map_err(AppError::from)?;

        debug!(chars = raw.chars().count(), "周报正文生成完成");

        // ── 阶段 C：校验 ────────────────────────────────────────
        let summary = parse_weekly_summary(&raw, &collector)?;
        let markdown = crate::report::render_weekly_summary(&summary);

        if summary.grounding == crate::report::Grounding::Ungrounded {
            // ⚠️ 没接地**不报错** —— 但要警告。
            //
            // 因为「模型确实没引用证据」有可能是**证据本来就少**
            // （比如离线环境 + 空知识库），这时周报是「如实说明查不到」，
            // 不是幻觉。
            //
            // 真正的幻觉是「引用了不存在的编号」——
            // 那种情况 grounding 会是 PartiallyGrounded。
            warn!(
                evidence = collector.len(),
                "周报没有引用任何证据 —— 检查是否有可用资料"
            );
        }

        Ok(WeeklyResult {
            summary,
            markdown,
            run,
        })
    }
}

/// 项目方案流水线。
pub struct PlanPipeline {
    agent: Arc<Agent>,
    llm: Arc<dyn FunctionLlm>,
    max_evidence_per_section: usize,
}

impl std::fmt::Debug for PlanPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanPipeline")
            .field("max_evidence", &self.max_evidence_per_section)
            .finish_non_exhaustive()
    }
}

impl PlanPipeline {
    /// 创建流水线。
    pub fn new(
        agent: Arc<Agent>,
        llm: Arc<dyn FunctionLlm>,
        max_evidence_per_section: usize,
    ) -> Self {
        Self {
            agent,
            llm,
            max_evidence_per_section,
        }
    }

    /// 跑一次完整流程。
    pub async fn run(&self, requirement: &str) -> AppResult<PlanResult> {
        if requirement.trim().is_empty() {
            return Err(AppError::BadRequest("需求描述不能为空".to_string()));
        }

        let system = prompt::research_system();
        let (run, collector) = self.agent.research(requirement, &system).await?;

        info!(
            evidence = collector.len(),
            tools = run.tools_used.len(),
            "方案研究阶段完成"
        );

        let user = prompt::plan_prompt(requirement, &collector, self.max_evidence_per_section);
        let raw = generate_text(
            self.llm.as_ref(),
            "你是资深技术架构师，负责输出项目构建方案。",
            &user,
        )
        .await
        .map_err(AppError::from)?;

        let plan = parse_project_plan(&raw, &collector)?;
        let markdown = crate::report::render_project_plan(&plan);

        Ok(PlanResult {
            plan,
            markdown,
            run,
        })
    }
}

/// 把周报写到文件。
///
/// ## ⚠️ 为什么文件名带日期和随机后缀
///
/// 同一周可能生成多次 —— 如果固定文件名，会**静默覆盖**前一次的结果。
/// 所以文件名带 `日期 + 递增序号`。
pub fn write_weekly_report(dir: &std::path::Path, markdown: &str) -> AppResult<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;

    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let mut path = dir.join(format!("weekly-{stamp}.md"));

    // 同一秒内多次 → 加后缀。
    let mut counter = 1u32;
    while path.exists() {
        path = dir.join(format!("weekly-{stamp}-{counter}.md"));
        counter += 1;
        if counter > 100 {
            break;
        }
    }

    std::fs::write(&path, markdown)?;
    info!(path = %path.display(), "周报已写入");

    Ok(path)
}

/// 把项目方案写到文件。
pub fn write_project_plan(dir: &std::path::Path, markdown: &str) -> AppResult<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;

    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let mut path = dir.join(format!("plan-{stamp}.md"));
    let mut counter = 1u32;
    while path.exists() {
        path = dir.join(format!("plan-{stamp}-{counter}.md"));
        counter += 1;
        if counter > 100 {
            break;
        }
    }

    std::fs::write(&path, markdown)?;
    info!(path = %path.display(), "项目方案已写入");

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{EvidenceCollector, EvidenceKind};

    #[test]
    fn writes_weekly_report_with_unique_name() {
        let dir = tempfile::tempdir().unwrap();

        let a = write_weekly_report(dir.path(), "# 周报 A").unwrap();
        let b = write_weekly_report(dir.path(), "# 周报 B").unwrap();

        assert_ne!(a, b);
        assert!(a.exists());
        assert!(b.exists());
        assert!(std::fs::read_to_string(&a).unwrap().contains("周报 A"));
    }

    #[test]
    fn writes_project_plan() {
        let dir = tempfile::tempdir().unwrap();

        let path = write_project_plan(dir.path(), "# 方案").unwrap();

        assert!(path.exists());
        assert!(path.to_string_lossy().contains("plan-"));
    }

    #[test]
    fn evidence_collector_is_reusable() {
        // 冒烟测试：确保证据层与流水线类型能对上。
        let mut c = EvidenceCollector::new();
        c.add(EvidenceKind::Kb, "kb_search", "a.md", "内容");
        assert_eq!(c.len(), 1);
    }
}
