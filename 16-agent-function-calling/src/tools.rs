//! 工具定义层 —— 把「工具」翻译成 OpenAI Function Calling 的规范形态。
//!
//! ## 为什么需要一个单独的模块
//!
//! 13/14/15 三个项目的工具是**自描述**的：工具定义只活在提示词文本里，
//! 模型输出 `Action: get_weather` 这样的**字符串**，我们再用正则解析。
//!
//! 本项目走的是**结构化协议** —— 工具定义必须以 **JSON Schema** 的形式
//! 放进 API 请求的 `tools` 字段，模型返回**结构化的** `tool_calls`。
//!
//! 这带来一个硬要求：**工具的 Schema 必须能被序列化、能被模型读懂**。
//! 提示词里的文本爱怎么写怎么写，Schema 里错一个字段模型就填错参数。
//!
//! ## 本模块的三个产出
//!
//! | 产出 | 说明 |
//! |---|---|
//! | [`ToolSpec`] | 工具的「四要素」：名字 / 描述 / 参数 Schema / 严格模式 |
//! | [`ToolSpec::to_genai_tool`] | 翻译成 `genai::chat::Tool`（**原样透传**，消毒由适配器做） |
//! | [`strict_schema`] | ★ **薄封装** [`genai::chat::sanitize_json_schema`]（OpenAI strict 方言） |
//!
//! ## ⚠️★ Schema 消毒**不自己实现**
//!
//! strict 模式的改写（所有字段进 `required`、每层补
//! `additionalProperties: false`、内联 `$ref`）由 **genai 适配器**
//! 在发请求前按 **provider 方言**完成。
//!
//! **自己改写 Schema = 与 provider 的校验规则漂移** ——
//! 本项目早先有一版手写 `strict_schema`（只改根层，**嵌套对象完全没处理**），
//! 正是这个问题的实例。
//!
//! ## strict 模式的硬约束（来自 OpenAI 文档）
//!
//! 官方文档明确写了 strict 模式的两个前提：
//!
//! 1. `additionalProperties` 必须设为 `false`
//! 2. `properties` 里**所有**字段都必须出现在 `required` 里
//!
//! 可选字段的正确写法是 `["string", "null"]` —— 类型数组里加 `null`。
//!
//! ## ⚠️ 但 DeepSeek 不强制执行 strict
//!
//! 实测（`deepseek-flash` / `deepseek-chat`）：
//!
//! ```text
//! strict=true + required 只写一个字段
//!   → 请求被接受
//!   → 模型照样可能不填另一个字段
//! ```
//!
//! 也就是说：**strict 是「锦上添花」，不是「安全保证」**。
//! 所以本项目**仍然**在本地做 Schema 校验（复用 14 的 `agent_14::schema`）——
//! 这正是 OpenAI 文档里「strict 模式下 schema 会被 sanitize」的另一种实现。
//!
//! 换句话说：**别把参数正确性外包给模型**。

use serde_json::Value;

/// 工具名约定。
///
/// OpenAI 文档说工具名「应当清晰且描述性强」，
/// 但同时 `a-zA-Z0-9_-` 是各厂商的**事实标准**（带空格会被部分厂商拒绝）。
/// 本项目统一用 `snake_case`。
pub const TOOL_NAME_MAX_LEN: usize = 64;

/// 一个工具的定义。
///
/// ## 「四要素」而不是「三要素」
///
/// 13/14 的 `ToolSpec` 是三要素（名字 / 描述 / Schema）。
/// 本项目多了一个 **`strict`** —— 因为它必须能被序列化进 API 请求。
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// 工具名（`snake_case`）。
    pub name: String,

    /// 工具描述。
    ///
    /// ⚠️ **描述是模型行为的一部分**，不是文档。
    /// OpenAI 的「最佳实践」里第一条就是：
    /// 「明确描述函数的目的和每个参数（以及它的格式），以及输出代表什么」。
    pub description: String,

    /// 参数的 JSON Schema。
    pub schema: Value,

    /// 是否开启 strict 模式。
    ///
    /// ## 为什么要保留这个开关
    ///
    /// DeepSeek **接受** `strict` 但**不强制**（实测）。
    /// 而 strict 模式要求「所有字段都进 required」——
    /// 这对「参数多且有默认值」的工具反而是负担。
    ///
    /// 所以这里做成开关：让调用方能对照实验。
    pub strict: bool,
}

impl ToolSpec {
    /// 构造一个工具定义（**不开** strict）。
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            strict: false,
        }
    }

    /// 构造一个工具定义并**开启** strict 模式。
    ///
    /// ## ⚠️ 开 strict 前必须想清楚
    ///
    /// strict 模式要求「**所有**字段都进 `required`」。
    /// 如果你的工具有**带默认值**的可选参数
    /// （比如 `days: integer = 1`），开 strict 之后
    /// 模型就**每次都得填**它 —— 因为 strict 不认 `default`，
    /// 它只认「字段在不在 `required` 里」。
    ///
    /// OpenAI 官方的解法是把可选字段的类型写成 `["string", "null"]`，
    /// 意思是「这个字段必填，但可以是 null」。
    ///
    /// 本项目的三个工具都**不开** strict —— 它们用 `default` 表达可选参数。
    pub fn strict(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            strict: true,
            ..Self::new(name, description, schema)
        }
    }

    /// 切换 strict 模式。
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// 翻译成 `genai::chat::Tool`。
    ///
    /// ## ⚠️★ Schema 的消毒**交给 genai 适配器**（不再自己改写）
    ///
    /// OpenAI 的 strict 模式要求 `additionalProperties: false`
    /// 且所有字段进 `required`。这件事**由适配器在发请求前完成**：
    ///
    /// - OpenAI / DeepSeek 走 OpenAI 适配器 → `sanitize_json_schema(schema, OpenAiStrictTool)`
    ///   （所有 `properties` 进 `required`、递归补 `additionalProperties: false`、内联 `$ref`）
    /// - Anthropic → `sanitize_json_schema(schema, AnthropicStructured)`
    ///
    /// ## ★ 为什么**不自己**改写
    ///
    /// **自己改写 = 与 provider 的校验规则漂移。**
    /// 手写版只覆盖了「根层 + 一层嵌套」这种已知形状；
    /// provider 升级了规则（比如加了 `$ref` 内联、tuple 类型），
    /// 本地手写版就过时了 —— 症状是 provider 报一个看不懂的 400。
    ///
    /// 本项目早先有一版手写 `strict_schema`（只改根层 `required` /
    /// `additionalProperties`，**嵌套对象完全没处理**），
    /// 正是这个问题的实例。现在它已被
    /// [`genai::chat::sanitize_json_schema`] 取代。
    ///
    /// ⚠️ 本地**仍然**要做业务级校验（复用 14 的 `agent_14::schema`）——
    /// strict 只是「锦上添花」，DeepSeek 实测**接受但不强制**它，
    /// 参数正确性不能外包给模型。
    pub fn to_genai_tool(&self) -> genai::chat::Tool {
        // ⚠️ 这里**原样**透传 Schema —— 消毒由适配器按 provider 方言做。
        //
        //    如果在这里再 sanitize 一次，等于「改写两遍」：
        //    第一遍按 OpenAI 方言，第二遍适配器又按 provider 方言 ——
        //    对 OpenAI 幂等，对 Anthropic 就**方言冲突**了。
        genai::chat::Tool::new(self.name.clone())
            .with_description(self.description.clone())
            .with_schema(self.schema.clone())
            .with_strict(self.strict)
    }

    /// 参数说明（渲染成人类可读文本，给 API 层展示用）。
    pub fn render_parameters(&self) -> String {
        agent_14::schema::render_schema(&self.schema)
    }
}

/// 把 Schema 改写成 **strict 模式**合规的形态。
///
/// ## ⚠️★ 本函数现在是 [`genai::chat::sanitize_json_schema`] 的**薄封装**
///
/// ## 为什么不再自己实现
///
/// 手写版只覆盖了「根层 + 一层嵌套」这种**已知形状**：
///
/// 1. 根对象加 `additionalProperties: false`
/// 2. 根对象 `required` 补齐**所有** `properties` 的键
/// 3. 每个子对象同样处理（递归）
/// 4. `type` 是数组时（`["string","null"]`）**不动**
///
/// ⚠️ **自己改写 Schema = 与 provider 的校验规则漂移。**
/// 手写版不知道 provider 后来加的规则（`$ref` 内联、tuple 类型、
/// `unevaluatedProperties`…），症状是 provider 报一个看不懂的 400。
///
/// crate 版按 **provider 方言**改写，三种方言：
///
/// | 方言 | 用在哪 |
/// |---|---|
/// | `OpenAiStrictTool` | OpenAI / DeepSeek 的 function 参数（要求显式 object 根） |
/// | `OpenAiStrict` | OpenAI 的结构化输出（`response_format`） |
/// | `AnthropicStructured` | Anthropic 的 JSON 输出与工具 |
///
/// ## 为什么仍然保留这个函数（而不是直接删掉）
///
/// ① 它是**本项目的公开 API**（`agent_16::tools::strict_schema`），
///    18-agent-error 直接 `pub use` 它；
/// ② 它让「strict 模式下 Schema 长什么样」在**本地**可测 ——
///
///    ⚠️ **本地可测 ≠ 本地生效**。
///    真正发给 provider 的 Schema 是适配器消毒后的版本，
///    本函数只是让测试能断言「按 OpenAI strict 方言应该长什么样」。
///
/// ## 递归是**必须**的
///
/// 手写版最大的缺陷：只改了根层。嵌套对象的 `required` 没补、
/// `additionalProperties` 没加 —— 而 strict 方言要求**每一层**都满足。
/// crate 版是递归的。
pub fn strict_schema(schema: &Value) -> Value {
    genai::chat::sanitize_json_schema(
        schema,
        genai::chat::JsonSchemaDialect::OpenAiStrictTool,
    )
}

/// 检查一个工具名是否合法。
///
/// OpenAI 文档说工具名「应当清晰且描述性强」，
/// 但本项目统一限制在 `a-zA-Z0-9_-` —— 这是各厂商的事实标准。
pub fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= TOOL_NAME_MAX_LEN
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// 检查一个 Schema 的**基础形状**是否合规。
///
/// ## 为什么和 strict 检查分开
///
/// 两种模式的要求**不一样**：
///
/// | | `additionalProperties: false` | 所有字段进 `required` |
/// |---|---|---|
/// | 非 strict | 建议有（拒绝未知参数） | 不要求 |
/// | strict | **硬要求** | **硬要求** |
///
/// 本项目的三个工具都**不开** strict —— 因为它们用 `default`
/// 表达可选参数（`days`、`top_k`），
/// 一旦开 strict，模型就**每次都得填**这些可选字段。
///
/// 所以「形状检查」和「strict 检查」必须分开：
/// 前者看 `additionalProperties`，后者才看 `required`。
pub fn check_schema_shape(schema: &Value) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();

    if schema.get("type").and_then(Value::as_str) != Some("object") {
        issues.push("根节点必须是 `object`".to_string());
    }

    if schema.get("additionalProperties") != Some(&Value::Bool(false)) {
        issues.push("缺少 `\"additionalProperties\": false`（拒绝未知参数）".to_string());
    }

    if !schema
        .get("properties")
        .map(Value::is_object)
        .unwrap_or(false)
    {
        issues.push("缺少 `properties`".to_string());
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// 检查一个 Schema 是否满足 strict 模式的硬约束。
///
/// 这个函数的用途是**自检** —— 单元测试里断言我们的工具 Schema 合规，
/// 避免线上才发现「OpenAI 拒绝了整个请求」。
pub fn check_strict_compliance(schema: &Value) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    check_object(schema, "", &mut issues);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// 递归检查 strict 合规性。
fn check_object(schema: &Value, path: &str, issues: &mut Vec<String>) {
    let Some(obj) = schema.as_object() else {
        return;
    };

    let is_object =
        obj.get("type").and_then(Value::as_str) == Some("object") || obj.contains_key("properties");

    if !is_object {
        // 非对象节点：检查嵌套的 items。
        if let Some(items) = obj.get("items") {
            check_object(items, &join(path, "items"), issues);
        }
        return;
    }

    let here = if path.is_empty() {
        "<root>".to_string()
    } else {
        path.to_string()
    };

    // ① additionalProperties 必须是 false。
    if obj.get("additionalProperties") != Some(&Value::Bool(false)) {
        issues.push(format!(
            "`{here}` 缺少 `\"additionalProperties\": false`（strict 模式硬要求）"
        ));
    }

    // ② 所有 properties 都必须在 required 里。
    if let Some(Value::Object(props)) = obj.get("properties") {
        let required: Vec<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        for key in props.keys() {
            if !required.contains(&key.as_str()) {
                issues.push(format!(
                    "`{here}` 的字段 `{key}` 没在 `required` 里（strict 模式要求所有字段必填）"
                ));
            }
        }

        for (key, sub) in props {
            check_object(sub, &join(path, key), issues);
        }
    }

    if let Some(items) = obj.get("items") {
        check_object(items, &join(path, "items"), issues);
    }
}

/// 拼接路径。
fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_schema_adds_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });

        let out = strict_schema(&schema);
        assert_eq!(out["additionalProperties"], json!(false));
    }

    #[test]
    fn strict_schema_fills_required_with_all_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "days": { "type": "integer" }
            },
            "required": ["city"]
        });

        let out = strict_schema(&schema);
        let required = out["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
        assert!(required.iter().any(|v| v.as_str() == Some("city")));
        assert!(required.iter().any(|v| v.as_str() == Some("days")));
    }

    #[test]
    fn strict_schema_keeps_all_properties_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "string" },
                "c": { "type": "string" }
            },
            "required": ["c", "a"]
        });

        let out = strict_schema(&schema);

        // ★ crate 版按 **`properties` 的键序**排列 `required`，
        // **不保留**原 `required` 的顺序（原顺序里的成员也不会丢）。
        //
        // ⚠️ 手写版是「保留原顺序 + 追加新字段」——
        //    两者的**顺序**不同，但**集合**相同。
        //    对 provider 而言顺序无语义（`required` 是集合），
        //    所以这是**行为等价**而非行为变更。
        //
        // ⚠️ 差异的来源：手写版是「在原 required 上追加」，
        //    crate 版是「从 properties 的键重新生成，再并入原 required 的多余项」。
        let required: Vec<&str> = out["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();

        // 集合相同 —— 三个字段都必须必填。
        assert_eq!(required.len(), 3);
        assert!(required.contains(&"a"));
        assert!(required.contains(&"b"));
        assert!(required.contains(&"c"));
    }

    #[test]
    fn strict_schema_recurses_into_nested_objects() {
        let schema = json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "country": { "type": "string" }
                    }
                }
            }
        });

        let out = strict_schema(&schema);
        assert_eq!(
            out["properties"]["location"]["additionalProperties"],
            json!(false)
        );
        assert_eq!(
            out["properties"]["location"]["required"],
            json!(["city", "country"])
        );
    }

    #[test]
    fn strict_schema_keeps_nullable_type_arrays() {
        // ⚠️ `["string","null"]` 是 strict 模式下「可选字段」的官方写法 ——
        // 绝不能被改写掉。
        let schema = json!({
            "type": "object",
            "properties": {
                "note": { "type": ["string", "null"], "description": "可选" }
            }
        });

        let out = strict_schema(&schema);
        assert_eq!(out["properties"]["note"]["type"], json!(["string", "null"]));
    }

    #[test]
    fn strict_schema_leaves_non_object_roots_alone() {
        let schema = json!({ "type": "string" });
        let out = strict_schema(&schema);
        assert_eq!(out, schema);
    }

    #[test]
    fn check_schema_shape_requires_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });

        let issues = check_schema_shape(&schema).unwrap_err();
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("additionalProperties"));
    }

    #[test]
    fn check_schema_shape_accepts_a_well_formed_schema() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });

        assert!(check_schema_shape(&schema).is_ok());
    }

    #[test]
    fn check_strict_compliance_accepts_a_compliant_schema() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city": { "type": "string" },
                "days": { "type": "integer" }
            },
            "required": ["city", "days"]
        });

        assert!(check_strict_compliance(&schema).is_ok());
    }

    #[test]
    fn check_strict_compliance_reports_missing_additional_properties() {
        let schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });

        let issues = check_strict_compliance(&schema).unwrap_err();
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("additionalProperties"));
    }

    #[test]
    fn check_strict_compliance_reports_missing_required() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city": { "type": "string" },
                "days": { "type": "integer" }
            },
            "required": ["city"]
        });

        let issues = check_strict_compliance(&schema).unwrap_err();
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("days"));
    }

    #[test]
    fn check_strict_compliance_recurses() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "location": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } }
                }
            },
            "required": ["location"]
        });

        let issues = check_strict_compliance(&schema).unwrap_err();
        // 嵌套对象缺 additionalProperties + 缺 required。
        assert!(issues.len() >= 2);
        assert!(issues.iter().any(|i| i.contains("location")));
    }

    #[test]
    fn validates_tool_names() {
        assert!(is_valid_tool_name("get_weather"));
        assert!(is_valid_tool_name("get-weather2"));
        assert!(!is_valid_tool_name(""));
        assert!(!is_valid_tool_name("get weather"));
        assert!(!is_valid_tool_name("天气"));
        assert!(!is_valid_tool_name(&"a".repeat(65)));
    }

    #[test]
    fn translates_to_genai_tool() {
        let spec = ToolSpec::strict(
            "get_weather",
            "查询天气",
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        );

        let tool = spec.to_genai_tool();
        assert_eq!(tool.name.as_str(), "get_weather");
        assert_eq!(tool.strict, Some(true));
        // ⚠️★ Schema **原样透传** —— 消毒由适配器在发请求前做。
        //
        //    如果这里断言 schema 已被改写，就等于把「本地改写」钉死了，
        //    正是本次重构要移除的行为。
        assert_eq!(
            tool.schema.unwrap(),
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            })
        );
    }

    /// ★ strict 的 Schema 消毒**由 genai 适配器**完成（本地不预改写）。
    ///
    /// 回归测试：修复前 `to_genai_tool` 会**自己**调 `strict_schema` 改写一遍，
    /// 导致「改写两遍」—— 对 OpenAI 幂等，对 Anthropic 就是**方言冲突**。
    #[test]
    fn to_genai_tool_does_not_presanitize_the_schema() {
        let spec = ToolSpec::strict(
            "t",
            "d",
            json!({
                "type": "object",
                "properties": { "a": { "type": "string" }, "b": { "type": "string" } },
                "required": ["a"]
            }),
        );

        let tool = spec.to_genai_tool();

        // ⚠️ `b` 仍在 `required` 之外 —— 说明本地**没有**预改写。
        assert_eq!(tool.schema.unwrap()["required"], json!(["a"]));
        // strict 标记必须透传 —— 适配器据此决定要不要 sanitize。
        assert_eq!(tool.strict, Some(true));
    }

    /// ★ 适配器消毒的效果在本地**可验证**（按 OpenAI strict 方言）。
    ///
    /// 这才是 `strict_schema` 存在的意义：
    /// 让「strict 模式下 Schema 应该长什么样」可在本地断言，
    /// 而**实际发给 provider 的**是适配器消毒后的版本。
    #[test]
    fn strict_schema_matches_the_openai_strict_dialect() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "nested": {
                    "type": "object",
                    "properties": { "x": { "type": "integer" }, "y": { "type": "integer" } },
                    "required": ["x"]
                }
            },
            "required": ["a"]
        });

        let out = strict_schema(&schema);

        // 根层：全部字段必填 + 禁止额外字段。
        let required = out["required"].as_array().unwrap();
        assert!(required.contains(&json!("a")));
        assert!(required.contains(&json!("nested")));
        assert_eq!(out["additionalProperties"], json!(false));

        // ★ 嵌套层也要处理 —— 手写版漏了这一步。
        let nested_required = out["properties"]["nested"]["required"].as_array().unwrap();
        assert!(nested_required.contains(&json!("x")));
        assert!(nested_required.contains(&json!("y")));
        assert_eq!(
            out["properties"]["nested"]["additionalProperties"],
            json!(false)
        );
    }

    /// ★ `$ref` 的内联规则比想象的**窄** —— 只有「带注解兄弟」的才会内联。
    ///
    /// crate 的判据是 `map.len() > 1`（见 `json_schema.rs` 的
    /// `inline_ref_annotations` 分支）：
    ///
    /// - `{"$ref": "#/$defs/X"}`（**裸引用**）→ **保留原样**
    /// - `{"$ref": "#/$defs/X", "description": "..."}`（带注解兄弟）→ **内联**
    ///
    /// ## 为什么裸引用**不**内联
    ///
    /// 因为 OpenAI 拒绝的是「`$ref` **带**兄弟注解」——
    /// 纯 `$ref` 本身是合法的。crate 只在**必须**时才动它，
    /// 避免「为了内联而内联」把 Schema 结构改得面目全非。
    ///
    /// ⚠️ 这正是「自己改写 = 与 provider 规则漂移」的又一个实例 ——
    /// 手写版**根本不知道**这条规则（它压根不处理 `$ref`）。
    #[test]
    fn strict_schema_inlines_refs_only_when_annotations_are_present() {
        let schema = json!({
            "type": "object",
            "properties": {
                "bare": { "$ref": "#/$defs/point" },
                "annotated": { "$ref": "#/$defs/point", "description": "一个点" }
            },
            "required": ["bare", "annotated"],
            "$defs": {
                "point": {
                    "type": "object",
                    "properties": { "x": { "type": "integer" } },
                    "required": ["x"]
                }
            }
        });

        let out = strict_schema(&schema);

        // ⚠️ 裸引用**保留** —— crate 只内联带注解兄弟的。
        assert_eq!(out["properties"]["bare"], json!({"$ref": "#/$defs/point"}));

        // 带注解兄弟的**被内联**，且注解保留。
        let annotated = &out["properties"]["annotated"];
        assert!(annotated.get("$ref").is_none(), "带注解的 $ref 应被内联");
        assert_eq!(annotated["type"], json!("object"));
        assert_eq!(annotated["description"], json!("一个点"));
    }

    #[test]
    fn non_strict_tool_keeps_schema_untouched() {
        let schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });
        let spec = ToolSpec::new("t", "d", schema.clone());
        let tool = spec.to_genai_tool();

        assert_eq!(tool.strict, Some(false));
        assert_eq!(tool.schema.unwrap(), schema);
    }
}
