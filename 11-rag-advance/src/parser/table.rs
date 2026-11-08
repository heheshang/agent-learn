//! 表格还原 —— 用**纯几何**把散落的文字块还原成网格。
//!
//! ## 问题
//!
//! PDF 里没有「表格」这种对象。一个表格在文件里是：
//!
//! - 一堆文字块（每个单元格一个），**带坐标**
//! - 一些画线的指令（格线）
//! - 可能还有合并单元格的说明
//!
//! 纯文本抽取拿到的是「一行一行的字」：
//!
//! ```text
//! Strategy  Tables  Reading order  Cost
//! Flat text none    column major  low
//! ```
//!
//! 这段文本**丢了列的对应关系**。读者知道有 4 列，但不知道
//! 「low」属于哪一列。检索更糟 —— 语义完全糊掉。
//!
//! ## 做法：不用模型，用坐标
//!
//! 关键洞察：**表格的本质是「网格」**。既然我们有每个单元格的坐标，
//! 那就把坐标聚类成行和列：
//!
//! ```text
//! ① 按垂直中心把单元格聚成行    （同一行的单元格 y 几乎一样）
//! ② 按左边缘把单元格聚成列      （同一列的单元格 x 几乎一样）
//! ③ 交叉成网格                  （行 × 列）
//! ```
//!
//! 这就是 docling 内部 `reconstruct_table` 做的事，本模块重新实现一遍 ——
//! 因为 docling 那份是内部函数，而且它的容差是硬编码的。
//!
//! ## 为什么不用模型
//!
//! MinerU / TableFormer 这类模型能处理**病态表格**（合并单元格、跨行表头、
//! 无框线表格）。但它们的代价是：
//!
//! - 要跑 ONNX，几十 MB 的模型
//! - 要把整页渲染成图片
//! - 慢（秒级 per 页）
//!
//! 而**规整表格**（有框线、列对齐）在企业文档里占绝大多数。
//! 对这类表格，纯几何方案是**毫秒级**且精度够用。
//!
//! 本项目给出**两条路**并让它们可以对照：
//!
//! | 策略 | 优点 | 代价 |
//! |---|---|---|
//! | [`TableStrategy::Geometric`] | 毫秒级、零模型依赖 | 处理不了合并单元格 |
//! | [`TableStrategy::Flat`] | 永远不会错 | 丢掉表格结构（当对照基线） |
//!
//! ## 关键工程细节
//!
//! ### 行列聚类的容差为什么要相对行高
//!
//! 不同文档的字号不同，固定的「差 2 个单位算同一行」在 8pt 正文里
//! 太宽、在 20pt 标题里太窄。所以容差取**行高的倍数**。
//!
//! ### 为什么要「列要有实际内容」
//!
//! 几何聚类经常把一列拆成两列 —— 尤其当单元格里是居中对齐的数字时。
//! 如果一列在所有行里都只有 1 个单元格有内容，那它几乎肯定是
//! 聚类误差产生的**伪列**，必须丢掉。

use serde::{Deserialize, Serialize};

use crate::parser::geometry::{BBox, PositionedText};

/// 表格还原策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableStrategy {
    /// 纯几何还原成网格（推荐）。
    Geometric,

    /// 压平成「一行一个字符串」的伪表格（**对照基线**）。
    ///
    /// 它的意义在于：让「结构化到底值不值」变成一个**可测量的对照实验**。
    Flat,
}

impl TableStrategy {
    /// 字符串形式。
    pub fn as_str(&self) -> &'static str {
        match self {
            TableStrategy::Geometric => "geometric",
            TableStrategy::Flat => "flat",
        }
    }

    /// 从字符串解析。
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "geometric" | "grid" => Some(TableStrategy::Geometric),
            "flat" | "baseline" => Some(TableStrategy::Flat),
            _ => None,
        }
    }

    /// 所有策略。
    pub fn all() -> Vec<Self> {
        vec![TableStrategy::Geometric, TableStrategy::Flat]
    }
}

/// 一个还原出来的表格。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// 网格内容：`grid[row][col]`。缺失的单元格是空字符串。
    pub grid: Vec<Vec<String>>,

    /// 表格在页面上的位置。
    pub bbox: BBox,

    /// 页码。
    pub page_no: usize,

    /// 还原策略。
    pub strategy: TableStrategy,

    /// 表头（网格的第一行）。
    ///
    /// 空表示第一行不是表头（那它就只是一行数据）。
    pub header: Vec<String>,
}

impl Table {
    /// 行数。
    pub fn row_count(&self) -> usize {
        self.grid.len()
    }

    /// 列数。
    pub fn column_count(&self) -> usize {
        self.grid.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// 渲染成 Markdown 表格。
    ///
    /// 这是表格**进入检索语料**的形态 —— Markdown 表格保留了
    /// 「哪一格是哪一列」的信息，纯文本没有。
    pub fn to_markdown(&self) -> String {
        if self.grid.is_empty() {
            return String::new();
        }

        let ncols = self.column_count();
        let mut out = String::new();

        // 表头行：优先用 `header`，否则用网格第一行。
        let head: Vec<String> = if self.header.is_empty() {
            self.grid[0].clone()
        } else {
            self.header.clone()
        };

        let render_row = |cells: &[String], out: &mut String| {
            out.push('|');
            for i in 0..ncols {
                let cell = cells.get(i).map(|s| s.as_str()).unwrap_or("");
                // 竖线和换行会破坏表格结构，必须转义。
                let escaped = cell.replace('|', "\\|").replace('\n', " ");
                out.push(' ');
                out.push_str(&escaped);
                out.push_str(" |");
            }
            out.push('\n');
        };

        render_row(&head, &mut out);

        // 分隔行。
        out.push('|');
        for _ in 0..ncols {
            out.push_str(" --- |");
        }
        out.push('\n');

        // 数据行：跳过表头行（如果它就是 grid[0]）。
        let skip = if self.header.is_empty() {
            0
        } else if self
            .grid
            .first()
            .map(|r| r == &self.header)
            .unwrap_or(false)
        {
            1
        } else {
            0
        };

        for row in self.grid.iter().skip(skip) {
            render_row(row, &mut out);
        }

        out
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "表格 {} 行 × {} 列 @p{} ({})",
            self.row_count(),
            self.column_count(),
            self.page_no,
            self.strategy.as_str()
        )
    }
}

/// 表格还原参数。
#[derive(Debug, Clone)]
pub struct TableOptions {
    /// 最少多少个单元格才认为是一个表格。
    pub min_cells: usize,

    /// 行聚类容差（行高的倍数）。
    pub row_tolerance: f32,

    /// 列聚类容差（行高的倍数）。
    pub column_tolerance: f32,
}

impl Default for TableOptions {
    fn default() -> Self {
        Self {
            min_cells: 4,
            row_tolerance: 0.7,
            column_tolerance: 1.5,
        }
    }
}

/// 还原网格。
///
/// 返回 `None` 表示这堆文字**不像表格**（单元格太少）。
///
/// ## 算法
///
/// ```text
/// ① 过滤：丢掉空文本的单元格
/// ② 行聚类：按垂直中心排序，中心差 < 行高×行容差 的归到同一行
/// ③ 列聚类：按左边缘排序，左边缘差 < 行高×列容差 的归到同一列
/// ④ 建网格：行 × 列
/// ⑤ 清伪列：丢掉「整列只有 ≤1 行有内容」的列
/// ⑥ 抽表头：网格第一行
/// ```
///
/// ## 第 ⑤ 步为什么必要
///
/// 居中对齐的数字列是最容易出问题的场景：
///
/// ```text
///        10        20        30
///        40        50        60
/// ```
///
/// 如果 10 和 20 的左边缘差了 3 个单位而容差是 2，就会被拆成两列，
/// 于是 3 列的表变成 6 列，其中 3 列只有一半有内容。
/// 丢掉「稀疏列」正好修正了这种误差。
fn build_grid(cells: &[PositionedText], options: &TableOptions) -> Option<Vec<Vec<String>>> {
    // ① 过滤空单元格。
    let cells: Vec<&PositionedText> = cells.iter().filter(|c| !c.text.trim().is_empty()).collect();

    if cells.len() < options.min_cells {
        return None;
    }

    // 典型行高（用来定容差）。用中位数而不是平均值 ——
    // 一个巨大的标题单元格会把平均值拉偏。
    let mut heights: Vec<f32> = cells.iter().map(|c| c.bbox.height()).collect();
    heights.sort_by(f32::total_cmp);
    let median_height = heights[heights.len() / 2].max(4.0);

    // ② 行聚类：按垂直中心排序后贪心分组。
    let mut by_y: Vec<&PositionedText> = cells.clone();
    by_y.sort_by(|a, b| a.bbox.center_y().total_cmp(&b.bbox.center_y()));

    let row_tol = median_height * options.row_tolerance;
    let mut rows: Vec<Vec<&PositionedText>> = Vec::new();
    let mut row_centers: Vec<f32> = Vec::new();

    for cell in by_y {
        let cy = cell.bbox.center_y();
        // 复用最后一行：它的中心离当前单元格足够近就算同一行。
        if let Some(last) = rows.last_mut()
            && let Some(&rc) = row_centers.last()
            && (cy - rc).abs() <= row_tol
        {
            last.push(cell);
            // 用两行的平均中心更新，避免行内偏移累积。
            let n = last.len() as f32;
            *row_centers.last_mut().unwrap() = rc + (cy - rc) / n;
            continue;
        }
        rows.push(vec![cell]);
        row_centers.push(cy);
    }

    // ③ 列聚类：按左边缘排序后贪心分组。
    let mut lefts: Vec<f32> = cells.iter().map(|c| c.bbox.left).collect();
    lefts.sort_by(f32::total_cmp);

    let col_tol = median_height * options.column_tolerance;
    let mut col_starts: Vec<f32> = Vec::new();
    for left in lefts {
        if col_starts.last().is_none_or(|&last| left - last > col_tol) {
            col_starts.push(left);
        }
    }

    if col_starts.is_empty() {
        return None;
    }

    let ncols = col_starts.len();

    // ④ 建网格。
    let mut grid: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in &rows {
        // 行内按左边缘排序，保证列序稳定。
        let mut sorted: Vec<&PositionedText> = row.clone();
        sorted.sort_by(|a, b| a.bbox.left.total_cmp(&b.bbox.left));

        let mut cells_in_row = vec![String::new(); ncols];
        for cell in sorted {
            let ci = col_starts
                .iter()
                .rposition(|&start| cell.bbox.left + col_tol * 0.5 >= start)
                .unwrap_or(0)
                .min(ncols - 1);

            let text = cell.text.trim();
            if cells_in_row[ci].is_empty() {
                cells_in_row[ci] = text.to_string();
            } else {
                cells_in_row[ci].push(' ');
                cells_in_row[ci].push_str(text);
            }
        }
        grid.push(cells_in_row);
    }

    // ⑤ 清伪列：丢掉「在所有行里只有 ≤1 行有内容」的列。
    //
    // 只有一列有内容的列几乎肯定是聚类误差的产物。
    // 但要注意：如果整张表只有 1 行有内容（比如一个跨行的标题行），
    // 那就不能清 —— 至少保留 2 列。
    if grid.len() >= 3 {
        let mut keep = vec![true; ncols];
        for (ci, keep_col) in keep.iter_mut().enumerate().take(ncols) {
            let used = grid
                .iter()
                .filter(|r| r.get(ci).is_some_and(|c| !c.trim().is_empty()))
                .count();
            if used <= 1 {
                *keep_col = false;
            }
        }

        let kept_count = keep.iter().filter(|k| **k).count();
        if kept_count >= 2 {
            for row in &mut grid {
                let mut new_row = Vec::with_capacity(kept_count);
                for (ci, cell) in row.iter().enumerate() {
                    if keep[ci] {
                        new_row.push(cell.clone());
                    }
                }
                *row = new_row;
            }
        }
    }

    if grid.is_empty() || grid[0].is_empty() {
        return None;
    }

    Some(grid)
}

/// 从一堆文字块里还原表格。
///
/// `cells` 应该已经**按阅读顺序**排好（或者至少是同一页的）。
pub fn reconstruct_table(
    cells: &[PositionedText],
    page_no: usize,
    options: &TableOptions,
) -> Option<Table> {
    let grid = build_grid(cells, options)?;

    let bbox = cells
        .iter()
        .map(|c| c.bbox)
        .reduce(|a, b| a.union(&b))
        .unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));

    let header = grid.first().cloned().unwrap_or_default();

    Some(Table {
        grid,
        bbox,
        page_no,
        strategy: TableStrategy::Geometric,
        header,
    })
}

/// 压平成「一行一个字符串」的伪表格（对照基线）。
///
/// 这就是纯文本抽取看到的样子 —— 它**没有**任何列结构。
pub fn flatten_table(cells: &[PositionedText], page_no: usize) -> Option<Table> {
    let non_empty: Vec<&PositionedText> =
        cells.iter().filter(|c| !c.text.trim().is_empty()).collect();

    if non_empty.len() < 2 {
        return None;
    }

    // 按行分组（和几何策略同样的容差逻辑），但**不**分列。
    let mut by_y: Vec<&PositionedText> = non_empty.clone();
    by_y.sort_by(|a, b| a.bbox.center_y().total_cmp(&b.bbox.center_y()));

    let mut heights: Vec<f32> = non_empty.iter().map(|c| c.bbox.height()).collect();
    heights.sort_by(f32::total_cmp);
    let median_height = heights[heights.len() / 2].max(4.0);

    let mut grid: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_center = 0.0f32;

    for cell in by_y {
        let cy = cell.bbox.center_y();
        if !current.is_empty() && (cy - current_center).abs() > median_height * 0.7 {
            grid.push(vec![current.join(" ")]);
            current.clear();
        }
        if current.is_empty() {
            current_center = cy;
        }
        current.push(cell.text.trim().to_string());
    }
    if !current.is_empty() {
        grid.push(vec![current.join(" ")]);
    }

    let bbox = cells
        .iter()
        .map(|c| c.bbox)
        .reduce(|a, b| a.union(&b))
        .unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));

    Some(Table {
        grid,
        bbox,
        page_no,
        strategy: TableStrategy::Flat,
        header: Vec::new(),
    })
}

/// 判断这堆单元格**像不像**表格。
///
/// ## 启发式依据
///
/// 真实表格的特征：
///
/// 1. 至少 2 行 2 列
/// 2. 单元格**填充率高** —— 真实表格大部分格子有内容
/// 3. 没有「超长文本」单元格 —— 表格里的单元格通常很短
///
/// 第 3 条是关键：**段落被误判成表格**是最常见的失败模式。
/// 两栏排版的两段长文字，每行 2 个单元格 —— 数量上像表格，
/// 但单元格里是整段话，那不是表格。
pub fn looks_like_table(cells: &[PositionedText]) -> bool {
    let non_empty: Vec<&PositionedText> =
        cells.iter().filter(|c| !c.text.trim().is_empty()).collect();

    if non_empty.len() < 4 {
        return false;
    }

    // 至少 2 行（垂直方向上有明显分离）。
    let mut ys: Vec<f32> = non_empty.iter().map(|c| c.bbox.center_y()).collect();
    ys.sort_by(f32::total_cmp);
    let mut distinct_ys = 1usize;
    for w in ys.windows(2) {
        if (w[1] - w[0]).abs() > 4.0 {
            distinct_ys += 1;
        }
    }
    if distinct_ys < 2 {
        return false;
    }

    // 至少 2 列（水平方向上有明显分离）。
    let mut xs: Vec<f32> = non_empty.iter().map(|c| c.bbox.left).collect();
    xs.sort_by(f32::total_cmp);
    let mut distinct_xs = 1usize;
    for w in xs.windows(2) {
        if (w[1] - w[0]).abs() > 8.0 {
            distinct_xs += 1;
        }
    }
    if distinct_xs < 2 {
        return false;
    }

    // 单元格不能太长 —— 段落不是表格。
    //
    // 阈值取 120 个字符：一个表格单元格 rarely 超过这个长度，
    // 而一个段落几乎总是超过。
    let max_len = non_empty
        .iter()
        .map(|c| c.text.trim().chars().count())
        .max()
        .unwrap_or(0);
    if max_len > 120 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个 3 行 3 列的表格单元格（坐标按 100px 的列宽、20px 的行高排布）。
    fn grid_cells<const N: usize, const M: usize>(rows: [[&str; M]; N]) -> Vec<PositionedText> {
        let mut out = Vec::new();
        for (ri, row) in rows.iter().enumerate() {
            for (ci, text) in row.iter().enumerate() {
                let left = 100.0 + ci as f32 * 100.0;
                let top = 100.0 + ri as f32 * 20.0;
                out.push(PositionedText::new(
                    *text,
                    BBox::new(left, top, left + 90.0, top + 16.0),
                ));
            }
        }
        out
    }

    #[test]
    fn reconstructs_a_clean_grid() {
        let cells = grid_cells([
            ["策略", "表格", "成本"],
            ["平铺", "无", "低"],
            ["版面", "全", "高"],
        ]);

        let table = reconstruct_table(&cells, 1, &TableOptions::default()).unwrap();

        assert_eq!(table.row_count(), 3);
        assert_eq!(table.column_count(), 3);
        assert_eq!(table.grid[0], vec!["策略", "表格", "成本"]);
        assert_eq!(table.grid[1], vec!["平铺", "无", "低"]);
        assert_eq!(table.grid[2], vec!["版面", "全", "高"]);
    }

    #[test]
    fn header_is_the_first_row() {
        let cells = grid_cells([["A", "B", "C"], ["1", "2", "3"]]);
        let table = reconstruct_table(&cells, 1, &TableOptions::default()).unwrap();
        assert_eq!(table.header, vec!["A", "B", "C"]);
    }

    #[test]
    fn markdown_keeps_the_column_mapping() {
        let cells = grid_cells([["Strategy", "Cost"], ["Flat", "low"]]);
        let table = reconstruct_table(&cells, 1, &TableOptions::default()).unwrap();
        let md = table.to_markdown();

        // 表头行 + 分隔行 + 数据行。
        assert!(md.contains("| Strategy | Cost |"));
        assert!(md.contains("| Flat | low |"));
        // 关键：分隔行必须存在，否则 Markdown 表格不成立。
        assert!(md.contains("| --- | --- |"));
    }

    #[test]
    fn markdown_escapes_pipes() {
        let cells = grid_cells([["a|b", "c"], ["d", "e"]]);
        let table = reconstruct_table(&cells, 1, &TableOptions::default()).unwrap();
        let md = table.to_markdown();
        // 未转义的竖线会多切出一列。
        assert!(md.contains("a\\|b"));
    }

    #[test]
    fn rejects_a_few_cells() {
        let cells = grid_cells([["A", "B"], ["1", "2"]]);
        // 4 个单元格刚好过阈值 —— 但只有 2 列 2 行。
        assert!(reconstruct_table(&cells, 1, &TableOptions::default()).is_some());

        let tiny = vec![PositionedText::new("A", BBox::new(0.0, 0.0, 10.0, 10.0))];
        assert!(reconstruct_table(&tiny, 1, &TableOptions::default()).is_none());
    }

    #[test]
    fn long_paragraphs_are_not_tables() {
        // 两个并排的长段落 —— 数量像表格，但内容是段落。
        let long = "这是一段很长的正文，用来测试段落不应该被误判成表格的启发式规则。".repeat(3);
        let cells = vec![
            PositionedText::new(long.clone(), BBox::new(0.0, 0.0, 400.0, 60.0)),
            PositionedText::new(long, BBox::new(420.0, 0.0, 600.0, 60.0)),
        ];
        assert!(!looks_like_table(&cells));
    }

    #[test]
    fn a_real_table_passes_the_heuristic() {
        let cells = grid_cells([
            ["Strategy", "Tables", "Cost"],
            ["Flat", "none", "low"],
            ["Layout", "full", "high"],
        ]);
        assert!(looks_like_table(&cells));
    }

    #[test]
    fn flat_strategy_has_no_column_structure() {
        let cells = grid_cells([["A", "B", "C"], ["1", "2", "3"]]);
        let flat = flatten_table(&cells, 1).unwrap();

        // 压平后每行只有 1 格 —— 列信息彻底丢失。
        assert_eq!(flat.column_count(), 1);
        assert_eq!(flat.row_count(), 2);
        assert!(flat.grid[0][0].contains(' '));
    }

    #[test]
    fn sparse_columns_are_dropped() {
        // 3 列，但第 3 列只有 1 行有内容 —— 是聚类误差产生的伪列。
        let mut cells = grid_cells([["A", "B", "C"], ["1", "2", "3"], ["4", "5", ""]]);
        // 把第三列的第一行挪远一点，制造出「只有一行用它」的稀疏列。
        cells[2].bbox.left = 400.0;

        let table = reconstruct_table(&cells, 1, &TableOptions::default()).unwrap();
        assert!(
            table.column_count() <= 2,
            "稀疏列应该被丢掉，实际列数 {}",
            table.column_count()
        );
    }
}
