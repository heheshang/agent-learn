//! 几何基础类型 —— 所有解析器的公共词汇。
//!
//! PDF 里**没有**「表格」「标题」「图片」这些概念，只有一堆带坐标的绘制指令。
//! 所谓「识别」表格，本质就是**从坐标里反推出结构**。
//!
//! 所以本模块先定义几个和坐标有关的公共类型，后面三个解析器
//! （表格 / 图片 / 版面）都建立在它们之上。

use serde::{Deserialize, Serialize};

/// 一个轴对齐的矩形框。
///
/// 坐标系约定：**左上角为原点，y 轴向下为正**。
///
/// ⚠️ PDF 内部用的是「左下角为原点、y 轴向上」的坐标系。
/// docling 给的 `TextCell` 是后者（`t` 是上边，`b` 是下边，且 `t > b`），
/// 而本项目统一转成前者（`top` 小于 `bottom`）——
/// 否则排序逻辑里到处要写翻转，迟早写错。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    /// 左边界。
    pub left: f32,
    /// 上边界。
    pub top: f32,
    /// 右边界。
    pub right: f32,
    /// 下边界。
    pub bottom: f32,
}

impl BBox {
    /// 构造一个框。
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// 从 docling 的 `TextCell` 坐标构造。
    ///
    /// docling 的约定：`l` / `r` 是左右，`t` 是上边，`b` 是下边，
    /// 且 **y 轴向上**（所以 `t > b`）。
    pub fn from_cell(l: f32, t: f32, r: f32, b: f32) -> Self {
        Self {
            left: l.min(r),
            top: t.min(b),
            right: l.max(r),
            bottom: t.max(b),
        }
    }

    /// 宽度。
    pub fn width(&self) -> f32 {
        (self.right - self.left).max(0.0)
    }

    /// 高度。
    pub fn height(&self) -> f32 {
        (self.bottom - self.top).max(0.0)
    }

    /// 面积。
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// 水平中心。
    pub fn center_x(&self) -> f32 {
        (self.left + self.right) / 2.0
    }

    /// 垂直中心。
    pub fn center_y(&self) -> f32 {
        (self.top + self.bottom) / 2.0
    }

    /// 和另一个框的交集面积。
    pub fn intersection_area(&self, other: &BBox) -> f32 {
        let w = (self.right.min(other.right) - self.left.max(other.left)).max(0.0);
        let h = (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0);
        w * h
    }

    /// 「交集 ÷ 自身面积」—— 这个框有多少比例落在另一个框里。
    ///
    /// 这是判断「某个文字块是不是在这个区域内」的标准做法。
    ///
    /// 为什么用交集除以**自身**面积，而不是并集（IoU）？
    /// 因为我们要问的是「这个单元格是不是**在**这个区域里」，
    /// 而不是一个「两个框有多像」。小单元格落在大区域里时，
    /// IoU 会很低（面积差太远），但重叠率是 1.0 —— 这才是我们要的答案。
    pub fn coverage_in(&self, other: &BBox) -> f32 {
        let own = self.area();
        if own <= f32::EPSILON {
            return 0.0;
        }
        self.intersection_area(other) / own
    }

    /// 合并两个框（取外接矩形）。
    pub fn union(&self, other: &BBox) -> BBox {
        BBox {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    /// 向下取整到整数网格 —— 用于**对齐**时消除浮点噪声。
    ///
    /// PDF 里的坐标是浮点数，同一个表格的不同单元格可能有 `0.0001` 的差异。
    /// 直接比较会漏判，所以比较之前先对齐到网格。
    pub fn snap(&self, grid: f32) -> BBox {
        if grid <= 0.0 {
            return *self;
        }
        BBox {
            left: (self.left / grid).round() * grid,
            top: (self.top / grid).round() * grid,
            right: (self.right / grid).round() * grid,
            bottom: (self.bottom / grid).round() * grid,
        }
    }
}

/// 一个带位置的文本单元。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionedText {
    /// 文本内容。
    pub text: String,

    /// 位置。
    pub bbox: BBox,
}

impl PositionedText {
    /// 构造。
    pub fn new(text: impl Into<String>, bbox: BBox) -> Self {
        Self {
            text: text.into(),
            bbox,
        }
    }
}

/// 布局元素的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    /// 标题（层级由 `level` 给出）。
    Heading,
    /// 正文段落。
    Paragraph,
    /// 列表项。
    ListItem,
    /// 表格。
    Table,
    /// 图片。
    Image,
    /// 页眉 / 页脚（会被丢弃 —— 它们对检索没有价值）。
    Furniture,
}

/// 文档里的一个布局元素。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutElement {
    /// 元素类型。
    pub kind: ElementKind,

    /// 位置。
    pub bbox: BBox,

    /// 元素文本。
    ///
    /// 表格元素存的是**渲染成 Markdown 的表格**；
    /// 图片元素存的是图片的 alt 描述（没有描述就是空）。
    pub text: String,

    /// 标题层级（1~6）。非标题元素是 `None`。
    pub level: Option<u8>,

    /// 页码。
    pub page_no: usize,
}

impl LayoutElement {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        let preview: String = self.text.chars().take(50).collect();
        format!(
            "[{:?} p{} lvl={:?}] {}",
            self.kind,
            self.page_no,
            self.level,
            preview.replace('\n', " ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_normalizes_inverted_y_axis() {
        // docling 的 t > b（y 轴向上），构造后应该变成 top < bottom。
        let bbox = BBox::from_cell(10.0, 100.0, 50.0, 80.0);
        assert_eq!(bbox.left, 10.0);
        assert_eq!(bbox.top, 80.0);
        assert_eq!(bbox.right, 50.0);
        assert_eq!(bbox.bottom, 100.0);
        assert!(bbox.top < bbox.bottom);
    }

    #[test]
    fn coverage_is_about_the_inner_box() {
        let small = BBox::new(10.0, 10.0, 20.0, 20.0);
        let big = BBox::new(0.0, 0.0, 100.0, 100.0);
        // 小框完全落在 大框 里 → 覆盖率 1.0。
        assert!((small.coverage_in(&big) - 1.0).abs() < 1e-6);
        // 反过来是大框落在大框里 → 覆盖率是 0.01。
        assert!((big.coverage_in(&small) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn disjoint_boxes_have_zero_coverage() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(50.0, 50.0, 60.0, 60.0);
        assert_eq!(a.coverage_in(&b), 0.0);
    }

    #[test]
    fn snap_removes_float_noise() {
        let bbox = BBox::new(10.001, 20.002, 30.003, 40.004);
        let snapped = bbox.snap(1.0);
        assert_eq!(snapped.left, 10.0);
        assert_eq!(snapped.top, 20.0);
    }

    #[test]
    fn union_takes_the_outer_rectangle() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 20.0, 20.0);
        let u = a.union(&b);
        assert_eq!(u.left, 0.0);
        assert_eq!(u.right, 20.0);
        assert_eq!(u.bottom, 20.0);
    }
}
