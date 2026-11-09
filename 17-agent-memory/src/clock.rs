//! 时间戳工具 —— 把「现在几点」从业务代码里抽出来。
//!
//! ## 为什么要单独一个模块
//!
//! 两个理由：
//!
//! 1. **可测试** —— 测「事实合并后 `updated_at` 更新了」时，
//!    需要能控制时间。集中在一个地方最容易替换。
//! 2. **格式统一** —— RFC 3339（如 `2026-11-09T10:30:00+08:00`）
//!    是**可排序**的 ISO 8601 变体，
//!    直接字符串比较就等价于时间比较（同一时区内）。
//!
//! ## ⚠️ 时区
//!
//! 本项目用**本地时区**（`chrono::Local`）。
//! 记忆文件按天组织（`daily/YYYY-MM-DD.md`），
//! 「今天」应该是**用户所在时区的今天**，不是 UTC 的今天。

use chrono::{Local, Utc};

/// 当前时间的 RFC 3339 字符串（本地时区）。
pub fn now_rfc3339() -> String {
    Local::now().to_rfc3339()
}

/// 当前时间的 UTC RFC 3339 字符串。
pub fn now_utc_rfc3339() -> String {
    Utc::now().to_rfc3339()
}

/// 今天的日期（`YYYY-MM-DD`，本地时区）。
pub fn today() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

/// 当前时间戳（毫秒）。
///
/// 用于耗时统计。
pub fn now_millis() -> u128 {
    Local::now().timestamp_millis() as u128
}

/// 生成一个事实 id。
///
/// ## 格式
///
/// ```text
/// fact_<unix毫秒>_<序号>
/// ```
///
/// ## ⚠️ 为什么不用随机 uuid
///
/// 1. **可排序** —— 同前缀下按字典序 = 按创建时间序
/// 2. **不占地方** —— 日志里可读
/// 3. **够用** —— 事实 id 只需要「同一 owner 内唯一」
pub fn new_fact_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("fact_{}_{n:04}", Utc::now().timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_is_sortable() {
        let a = now_rfc3339();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = now_rfc3339();
        // 同一时区内，RFC 3339 的字典序 == 时间序。
        assert!(a <= b, "{a} 应该 <= {b}");
    }

    #[test]
    fn today_has_the_right_shape() {
        let t = today();
        assert_eq!(t.len(), 10);
        assert_eq!(&t[4..5], "-");
        assert_eq!(&t[7..8], "-");
    }

    #[test]
    fn fact_ids_are_unique() {
        let a = new_fact_id();
        let b = new_fact_id();
        assert_ne!(a, b);
        assert!(a.starts_with("fact_"));
    }
}
