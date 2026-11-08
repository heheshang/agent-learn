//! JSON Schema 子集校验器 —— 对应 LangChain 的 `args_schema`。
//!
//! ## 为什么需要它
//!
//! LangChain 里工具的参数契约用 **Pydantic 模型**或 **JSON Schema** 表达：
//!
//! ```python
//! class WeatherInput(BaseModel):
//!     location: str = Field(description="City name or coordinates")
//!     units: Literal["celsius", "fahrenheit"] = Field(default="celsius")
//! ```
//!
//! 教程里明确说「**类型标注是必需的**，因为它们定义了工具的输入 Schema」，
//! 而 Schema 同时干两件事：
//!
//! 1. **告诉模型参数怎么填**（这段会渲染进提示词）
//! 2. **本地校验**（模型填错时不发请求，直接告诉模型哪里错了）
//!
//! 第 2 点是本项目 [`crate::tool`] 里 `validate_args` 的依据。
//!
//! ## 支持的 Schema 子集
//!
//! 只实现工具参数真正需要的那部分 —— 刻意**不支持** `oneOf` / `allOf` /
//! `$ref` / 条件分支：
//!
//! | 关键字 | 作用 |
//! |---|---|
//! | `type` | `object` / `string` / `number` / `integer` / `boolean` / `array` / `null` |
//! | `properties` | 子字段定义 |
//! | `required` | 必填字段 |
//! | `additionalProperties: false` | 拒绝未知字段 |
//! | `enum` | 枚举值 |
//! | `default` | 缺省值（校验前填充） |
//! | `minimum` / `maximum` | 数值范围 |
//! | `items` | 数组元素类型 |
//! | `description` | 字段说明（渲染进提示词） |
//!
//! ## 为什么错误信息要「可执行」
//!
//! 校验失败时返回的是给**模型**看的文案，所以必须说清楚：
//! 哪个参数、期望什么、实际是什么。
//!
//! ```text
//! 缺少必填参数 `location`（类型 string，说明：城市名）
//! 参数 `units` 必须是字符串，实际是数字
//! 参数 `days` 超出范围（1..=7），实际是 30
//! ```
//!
//! 这类错误会被渲染成 `Observation` 喂回模型，模型读完就能改对。

use serde_json::{Map, Value};

/// 单条校验问题。
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// 出问题的参数路径（`location`、`forecast.days` 这样的点号路径）。
    pub path: String,

    /// 问题描述（给模型看）。
    pub message: String,
}

impl Issue {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Issue {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}` {}", self.path, self.message)
    }
}

/// 校验结果。
#[derive(Debug, Clone, Default)]
pub struct Validation {
    /// 所有问题（空 = 通过）。
    pub issues: Vec<Issue>,
}

impl Validation {
    /// 是否通过。
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }

    /// 拼成一句给模型看的话。
    pub fn message(&self) -> String {
        self.issues
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("；")
    }
}

/// 用 Schema 校验参数。
///
/// 返回 [`Validation`] 而不是 `Result` —— 因为**要一次性报出所有问题**，
/// 让模型一次改对，而不是修一个报一个。
pub fn validate(schema: &Value, args: &Value) -> Validation {
    let mut issues = Vec::new();

    // 根必须是对象。
    let Some(object) = args.as_object() else {
        issues.push(Issue::new(
            "<root>",
            format!("必须是 JSON 对象，实际是 {}", json_type_name(args)),
        ));
        return Validation { issues };
    };

    validate_object(schema, object, "", &mut issues);

    Validation { issues }
}

/// 校验一个对象（相对某个 Schema）。
fn validate_object(
    schema: &Value,
    object: &Map<String, Value>,
    prefix: &str,
    issues: &mut Vec<Issue>,
) {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        // 没有 properties —— 只检查 required。
        check_required(schema, object, prefix, issues);
        return;
    };

    // ── 必填检查 ────────────────────────────────────────────────
    check_required(schema, object, prefix, issues);

    // ── 逐字段校验 ──────────────────────────────────────────────
    for (key, sub_schema) in props {
        let path = join_path(prefix, key);
        let Some(value) = object.get(key) else {
            // 缺省值由 `fill_defaults` 负责填充，这里不算错。
            continue;
        };
        validate_value(sub_schema, value, &path, issues);
    }

    // ── 未知字段 ────────────────────────────────────────────────
    if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
        for key in object.keys() {
            if !props.contains_key(key) {
                issues.push(Issue::new(
                    join_path(prefix, key),
                    "是未知参数（可用参数: ".to_string()
                        + &props.keys().cloned().collect::<Vec<_>>().join(", ")
                        + "）",
                ));
            }
        }
    }
}

/// 校验 `required` 列表。
fn check_required(
    schema: &Value,
    object: &Map<String, Value>,
    prefix: &str,
    issues: &mut Vec<Issue>,
) {
    let Some(required) = schema.get("required").and_then(Value::as_array) else {
        return;
    };
    for key in required {
        let Some(name) = key.as_str() else { continue };
        if !object.contains_key(name) {
            let path = join_path(prefix, name);
            let desc = schema
                .get("properties")
                .and_then(|p| p.get(name))
                .and_then(|s| s.get("description"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let suffix = if desc.is_empty() {
                String::new()
            } else {
                format!("（说明：{desc}）")
            };
            issues.push(Issue::new(
                path,
                format!("是必填参数，但没有被提供{suffix}"),
            ));
        }
    }
}

/// 校验单个值。
fn validate_value(schema: &Value, value: &Value, path: &str, issues: &mut Vec<Issue>) {
    // 类型。
    if let Some(expected) = schema.get("type").and_then(Value::as_str)
        && !matches_type(expected, value)
    {
        issues.push(Issue::new(
            path,
            format!("类型必须是 {expected}，实际是 {}", json_type_name(value)),
        ));
        // 类型都不对了，后面的检查没意义。
        return;
    }

    // 枚举。
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
        && !allowed.contains(value)
    {
        let rendered = allowed
            .iter()
            .map(|v| match v.as_str() {
                Some(s) => s.to_string(),
                None => v.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" / ");
        issues.push(Issue::new(
            path,
            format!("必须是 [{rendered}] 之一，实际是 {value}"),
        ));
    }

    // 数值范围。
    if let Some(min) = schema.get("minimum").and_then(Value::as_f64)
        && let Some(actual) = value.as_f64()
        && actual < min
    {
        issues.push(Issue::new(path, format!("不能小于 {min}，实际是 {actual}")));
    }
    if let Some(max) = schema.get("maximum").and_then(Value::as_f64)
        && let Some(actual) = value.as_f64()
        && actual > max
    {
        issues.push(Issue::new(path, format!("不能大于 {max}，实际是 {actual}")));
    }

    // 字符串长度。
    if let Some(min_len) = schema.get("minLength").and_then(Value::as_u64)
        && let Some(actual) = value.as_str()
        && (actual.chars().count() as u64) < min_len
    {
        issues.push(Issue::new(
            path,
            format!("长度不能小于 {min_len}，实际是 {}", actual.chars().count()),
        ));
    }

    // 嵌套对象。
    if schema.get("properties").is_some()
        && let Some(obj) = value.as_object()
    {
        validate_object(schema, obj, path, issues);
    }

    // 数组元素。
    if let Some(items) = schema.get("items")
        && let Some(arr) = value.as_array()
    {
        for (i, item) in arr.iter().enumerate() {
            validate_value(items, item, &format!("{path}[{i}]"), issues);
        }
    }
}

/// 按 Schema 填充缺省值。
///
/// ## 为什么在校验**之前**填充
///
/// LangChain 里 `units: Literal["celsius","fahrenheit"] = Field(default="celsius")`
/// 的默认值是在**调用时**生效的 —— 模型不填就用默认值。
/// 所以顺序必须是：填默认值 → 校验。
pub fn fill_defaults(schema: &Value, args: &mut Map<String, Value>) {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };

    for (key, sub_schema) in props {
        // 缺省值：只在字段**完全没给**时填。
        if !args.contains_key(key)
            && let Some(default) = sub_schema.get("default")
        {
            args.insert(key.clone(), default.clone());
            continue;
        }

        // 递归填充嵌套对象。
        if let Some(obj) = args.get_mut(key).and_then(Value::as_object_mut)
            && sub_schema.get("properties").is_some()
        {
            fill_defaults(sub_schema, obj);
        }
    }
}

/// 判断值是否符合给定 JSON 类型。
fn matches_type(expected: &str, value: &Value) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        // ⚠️ JSON 里 `1.0` 也是 number，但 `integer` 要求没有小数部分。
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// JSON 值的类型名（给错误信息用）。
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// 拼接参数路径。
fn join_path(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// 把 Schema 渲染成给模型看的「参数说明」。
///
/// ## 格式
///
/// ```text
/// - location (string, 必填): 城市名，例如 "北京"
/// - units (string, 选填, 默认 celsius): 温度单位
/// ```
///
/// 这段文字会直接进提示词 —— 所以它是**模型行为的一部分**，
/// 不是文档。写得含糊，模型就会乱填。
pub fn render_schema(schema: &Value) -> String {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return String::new();
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let mut lines = Vec::new();
    for (key, sub) in props {
        let ty = sub.get("type").and_then(Value::as_str).unwrap_or("any");
        let mut flags = Vec::new();
        if required.contains(&key.as_str()) {
            flags.push("必填".to_string());
        } else {
            flags.push("选填".to_string());
        }
        if let Some(default) = sub.get("default") {
            flags.push(format!(
                "默认 {}",
                default
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| default.to_string())
            ));
        }
        if let Some(allowed) = sub.get("enum").and_then(Value::as_array) {
            flags.push(format!(
                "取值 {}",
                allowed
                    .iter()
                    .map(|v| v.as_str().unwrap_or_default().to_string())
                    .collect::<Vec<_>>()
                    .join("/")
            ));
        }

        let desc = sub.get("description").and_then(Value::as_str).unwrap_or("");
        let mut line = format!("- {key} ({ty}, {}): {}", flags.join(", "), desc);
        if desc.is_empty() {
            line = format!("- {key} ({ty}, {})", flags.join(", "));
        }
        lines.push(line);
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名，例如 北京"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "温度单位"
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "default": 1,
                    "description": "预报天数"
                }
            },
            "required": ["location"],
            "additionalProperties": false
        })
    }

    #[test]
    fn accepts_valid_args() {
        let v = validate(
            &weather_schema(),
            &json!({ "location": "北京", "units": "celsius", "days": 3 }),
        );
        assert!(v.is_ok(), "不该有问题: {}", v.message());
    }

    #[test]
    fn reports_missing_required_field() {
        let v = validate(&weather_schema(), &json!({}));
        assert!(!v.is_ok());
        assert!(v.message().contains("location"));
        assert!(v.message().contains("必填"));
    }

    #[test]
    fn reports_type_mismatch() {
        let v = validate(&weather_schema(), &json!({ "location": 42 }));
        assert!(!v.is_ok());
        assert!(v.message().contains("类型必须是 string"));
        assert!(v.message().contains("实际是 number"));
    }

    #[test]
    fn reports_enum_violation() {
        let v = validate(
            &weather_schema(),
            &json!({ "location": "北京", "units": "kelvin" }),
        );
        assert!(!v.is_ok());
        assert!(v.message().contains("celsius"));
    }

    #[test]
    fn reports_range_violation() {
        let v = validate(
            &weather_schema(),
            &json!({ "location": "北京", "days": 30 }),
        );
        assert!(!v.is_ok());
        assert!(v.message().contains("不能大于 7"));
    }

    #[test]
    fn rejects_unknown_field_when_additional_properties_is_false() {
        let v = validate(
            &weather_schema(),
            &json!({ "location": "北京", "foo": "bar" }),
        );
        assert!(!v.is_ok());
        assert!(v.message().contains("未知参数"));
    }

    #[test]
    fn reports_all_issues_at_once() {
        // 一次性报出全部问题 —— 让模型一次改对。
        let v = validate(
            &weather_schema(),
            &json!({ "units": "kelvin", "days": 30, "extra": 1 }),
        );
        assert!(v.issues.len() >= 3, "应该一次报出多个问题: {}", v.message());
    }

    #[test]
    fn rejects_non_object_root() {
        let v = validate(&weather_schema(), &json!("北京"));
        assert!(!v.is_ok());
        assert!(v.message().contains("必须是 JSON 对象"));
    }

    #[test]
    fn fills_defaults_before_validation() {
        let mut args = json!({ "location": "北京" }).as_object().unwrap().clone();
        fill_defaults(&weather_schema(), &mut args);

        assert_eq!(args["units"], "celsius");
        assert_eq!(args["days"], 1);

        // 填完默认值后校验必须通过。
        let v = validate(&weather_schema(), &Value::Object(args));
        assert!(v.is_ok(), "{}", v.message());
    }

    #[test]
    fn does_not_override_provided_value() {
        let mut args = json!({ "location": "北京", "units": "fahrenheit" })
            .as_object()
            .unwrap()
            .clone();
        fill_defaults(&weather_schema(), &mut args);

        assert_eq!(args["units"], "fahrenheit");
    }

    #[test]
    fn integer_type_rejects_float() {
        let v = validate(
            &weather_schema(),
            &json!({ "location": "北京", "days": 1.5 }),
        );
        assert!(!v.is_ok());
        assert!(v.message().contains("integer"));
    }

    #[test]
    fn renders_readable_schema() {
        let rendered = render_schema(&weather_schema());
        assert!(rendered.contains("location (string, 必填)"));
        assert!(rendered.contains("城市名"));
        assert!(rendered.contains("默认 celsius"));
        assert!(rendered.contains("取值 celsius/fahrenheit"));
    }
}
