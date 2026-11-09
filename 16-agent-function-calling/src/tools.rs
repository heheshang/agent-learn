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
//! | [`ToolSpec::to_genai_tool`] | 翻译成 `genai::chat::Tool` |
//! | [`strict_schema`] | 把 Schema 改写成 **strict 模式**合规的形态 |
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
    /// ## 为什么在这里做 strict 改写
    ///
    /// OpenAI 的 strict 模式要求 `additionalProperties: false`
    /// 且所有字段进 `required`。
    /// 如果直接发原始 Schema 给 `strict: true` 的工具，
    /// OpenAI 会**拒绝整个请求**（不是降级，是报错）。
    ///
    /// 所以必须在**发送前**把 Schema 改写成合规形态。
    pub fn to_genai_tool(&self) -> genai::chat::Tool {
        let schema = if self.strict {
            strict_schema(&self.schema)
        } else {
            self.schema.clone()
        };

        genai::chat::Tool::new(self.name.clone())
            .with_description(self.description.clone())
            .with_schema(schema)
            .with_strict(self.strict)
    }

    /// 参数说明（渲染成人类可读文本，给 API 层展示用）。
    pub fn render_parameters(&self) -> String {
        agent_14::schema::render_schema(&self.schema)
    }
}

/// 把 Schema 改写成 **strict 模式**合规的形态。
///
/// ## 改写规则
///
/// 1. 根对象加 `additionalProperties: false`
/// 2. 根对象 `required` 补齐**所有** `properties` 的键
/// 3. 每个子对象同样处理（递归）
/// 4. `type` 是数组时（`["string","null"]`）**不动** —— 那是合法的可选字段写法
///
/// ## 为什么「补齐 required」而不是「报错」
///
/// OpenAI 文档说：schema 不满足 strict 要求时**请求会被拒绝**。
/// 但我们更希望「工具能用，只是模型必须填所有字段」。
///
/// ⚠️ 这里的选择有代价：把一个本来可选的字段变成必填，
/// 模型就必须给一个值 —— 通常是 `null`。
/// 所以**只有真正可选的字段**才应该用 `["string","null"]` 表达。
///
/// ## 不改动 `type` 数组的含义
///
/// `{"type": ["string", "null"]}` 在 strict 模式下是**官方推荐**的
/// 「可选字段」写法 —— 它本来就同时是 required 和可为 null。
pub fn strict_schema(schema: &Value) -> Value {
    let mut out = schema.clone();

    if let Some(obj) = out.as_object_mut() {
        // 根必须是 object 才有 strict 语义。
        let is_object = obj.get("type").and_then(Value::as_str) == Some("object")
            || obj.contains_key("properties");

        if is_object {
            obj.insert("additionalProperties".to_string(), Value::Bool(false));

            // 补齐 required。
            if let Some(Value::Object(props)) = obj.get("properties") {
                let mut required: Vec<Value> = obj
                    .get("required")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();

                for key in props.keys() {
                    if !required.iter().any(|v| v.as_str() == Some(key.as_str())) {
                        required.push(Value::String(key.clone()));
                    }
                }

                obj.insert("required".to_string(), Value::Array(required));
            }
        }

        // 递归处理子对象。
        if let Some(Value::Object(props)) = obj.get_mut("properties") {
            for (_, sub) in props.iter_mut() {
                *sub = strict_schema(sub);
            }
        }
        if let Some(items) = obj.get_mut("items") {
            *items = strict_schema(items);
        }
    }

    out
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
    fn strict_schema_preserves_existing_required_order() {
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
        // 原有顺序保留，新字段追加在后。
        let required: Vec<&str> = out["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(required, vec!["c", "a", "b"]);
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
        // strict 模式下 additionalProperties 被补上。
        let schema = tool.schema.unwrap();
        assert_eq!(schema["additionalProperties"], json!(false));
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
