//! **数据模型** —— 活动记录、查询窗口、周报。
//!
//! # 为什么自己写而不用 serde 直接 derive
//!
//! 因为周报要**可复现**：同样的输入必须产出**逐字节相同**的 Markdown。
//! 任何依赖 `HashMap` 迭代顺序的地方都会破坏这一点，
//! 所以本模块的所有集合都是 `Vec` / `BTreeMap`（有序）。

use serde::{Deserialize, Serialize};

/// 一条活动记录。
///
/// 这是整个系统的**输入单元** —— 一次提交、一个任务、一次事件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Activity {
    /// 稳定 ID（去重时保留「先遇到的那条」）。
    pub id: String,

    /// 时间戳（Unix 秒）。
    pub ts: i64,

    /// 活动类型（`feature` / `fix` / `incident` / …）。
    pub kind: String,

    /// 负责人。
    pub author: String,

    /// 所属模块。
    pub scope: String,

    /// 标题。
    pub title: String,

    /// 权重（影响统计）。
    pub weight: i32,

    /// 是否是风险项。
    pub risk: bool,
}

/// 一个查询窗口。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Window {
    /// 起点（Unix 秒，含）。
    pub start: i64,

    /// 终点（Unix 秒，含）。
    pub end: i64,
}

impl Window {
    /// 构造一个窗口。
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    /// 窗口的天数。
    ///
    /// ## ⚠️ 必须按「自然日边界」算
    ///
    /// 直接 `(end - start) / 86400` 会在起点不是零点时算错
    /// （比如 10:00 → 次日 09:00 实际跨了 2 个自然日，但除法只给出 0.95 天）。
    pub fn days(&self) -> usize {
        let start_day = self.start.div_euclid(86_400);
        let end_day = self.end.div_euclid(86_400);
        ((end_day - start_day) + 1).max(1) as usize
    }

    /// 是否包含某个时间戳。
    pub fn contains(&self, ts: i64) -> bool {
        ts >= self.start && ts <= self.end
    }
}

/// 一个自然日的键（UTC）。
///
/// ## ⚠️ 必须是**常量时间**的
///
/// 这是本项目最重要的优化点之一 —— 见 [`crate::pipeline`]。
/// 朴素写法是 `chrono` 的格式化（要分配字符串 + 做时区换算），
/// 而 Howard Hinnant 的算法是**纯整数运算**。
pub fn day_key(ts: i64) -> i64 {
    // 向下取整到当天零点（对负数也正确 —— `div_euclid` 而不是 `/`）。
    ts.div_euclid(86_400)
}

/// 一个自然日的键渲染成 `YYYY-MM-DD`（仅用于展示）。
pub fn day_label(ts: i64) -> String {
    let day = day_key(ts);
    chrono::DateTime::from_timestamp(day * 86_400, 0)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| format!("day-{}", day))
}

/// ★ 一次聚合的**汇总结论**。
///
/// ## ⚠️ 这是三档策略的**等价性证据**
///
/// `naive` / `pre-agg` / `optimized` 必须产出**完全相同**的 `Digest` ——
/// 否则测的是「不同的功能」而不是「优化」。
///
/// 这由回归测试 `naive_and_preagg_produce_identical_digests` 锁住。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Digest {
    /// 过滤后的记录数。
    pub total: usize,

    /// 去重后的记录数。
    pub deduped: usize,

    /// 覆盖的模块数。
    pub scopes: usize,

    /// 风险项数。
    pub risks: usize,

    /// 覆盖的天数。
    pub days: usize,

    /// 按类型的分布。
    pub by_kind: Vec<(String, usize)>,

    /// 按负责人的分布。
    pub by_author: Vec<(String, usize)>,

    /// 按模块的分布。
    pub by_scope: Vec<(String, usize)>,

    /// 按天的分布。
    pub by_day: Vec<(String, usize)>,

    /// 亮点。
    pub highlights: Vec<Highlight>,

    /// 风险项。
    pub risks_list: Vec<Highlight>,
}

/// 一条亮点或风险。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Highlight {
    /// 活动 ID。
    pub id: String,

    /// 标题。
    pub title: String,

    /// 模块。
    pub scope: String,

    /// 负责人。
    pub author: String,

    /// 类型。
    pub kind: String,

    /// 权重。
    pub weight: i32,
}
