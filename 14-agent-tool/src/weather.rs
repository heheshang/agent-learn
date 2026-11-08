//! 天气工具 —— 本项目的主角：一个**真正需要联网**的自定义工具。
//!
//! ## 为什么这个工具值得单独写
//!
//! 13-agent-core 的工具是**纯本地**的：BM25 检索（内存操作）和手写算术。
//! 它们跑得和读一个文件一样快，不可能失败。
//!
//! 天气工具不一样：
//!
//! - 它要**等外部 HTTP 服务**（Open-Meteo 免 API key）
//! - 它会**失败**：地名查不到、网络不通、上游返回 500
//! - 它有**多个参数**：城市名、单位、预报天数
//!
//! 这三件事让天气工具成为检验「工具契约」设计的最佳载体。
//!
//! ## 数据来源：Open-Meteo
//!
//! 选它有两个原因：
//!
//! 1. **免 API key** —— 不需要额外的环境变量，`cargo run` 立刻能跑
//! 2. **免注册** —— 没有配额限制
//!
//! 两个接口：
//!
//! - `GET /v1/search?name=北京&count=1` —— 地名 → 经纬度
//! - `GET /v1/forecast?latitude=..&longitude=..&current=..&daily=..` —— 预报
//!
//! ## 工具的两段式设计
//!
//! ```text
//! 用户说「北京天气怎么样」
//!        ↓
//!   get_weather(location="北京", units="celsius", days=1)
//!        ↓
//!   ① 地理编码：地名 → (lat, lon, 城市全名, 国家)
//!   ② 取预报：经纬度 → 当前天气 + 每日预报
//!        ↓
//!   渲染成给模型看的 Observation
//! ```
//!
//! **为什么是「地名 → 经纬度 → 预报」两跳**，而不是直接查预报：
//! 预报接口只认经纬度。地名解析是独立的一步。
//!
//! ## 工具失败的三种形态
//!
//! | 情况 | 处理 | 模型看到什么 |
//! |---|---|---|
//! | 地名查不到 | 返回 `ToolError::Execution` | 「找不到城市 `Atlantis`，请换一个城市名」 |
//! | 网络不通 / 超时 | 返回 `ToolError::Upstream` | 「天气服务暂时不可用：...」 |
//! | 参数不合法 | Schema 校验拦下 | 「`days` 必须在 1..=7 之间」 |
//!
//! 三种都能让**模型自我纠正** —— 这才是好的工具错误。

use agent_logging::{debug, info, warn};
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolFuture, ToolOutcome, ToolSpec};

/// `get_weather` 工具的名字。
///
/// ⚠️ 这个名字会**原样出现在提示词里**，模型照着它调工具。
/// 所以它必须和提示词里写的完全一致（这里统一由常量提供）。
pub const GET_WEATHER_TOOL_NAME: &str = "get_weather";

/// 地理编码接口地址。
pub const GEOCODING_URL: &str = "https://geocoding-api.open-meteo.com/v1/search";

/// 预报接口地址。
pub const FORECAST_URL: &str = "https://api.open-meteo.com/v1/forecast";

/// 天气工具的参数 Schema。
///
/// ## 与 LangChain 教程的对应
///
/// ```python
/// class WeatherInput(BaseModel):
///     location: str = Field(description="City name or coordinates")
///     units: Literal["celsius", "fahrenheit"] = Field(default="celsius")
///     include_forecast: bool = Field(default=False)
/// ```
///
/// 本项目把 `include_forecast: bool` 换成了 `days: integer` ——
/// 因为「要不要预报」是个布尔，「要几天」才是真正需要的信息量。
///
/// ## `additionalProperties: false` 的意义
///
/// 模型经常会**自作主张**地加参数（`{"location": "北京", "unit": "C"}`）。
/// 拒绝未知参数能让它**立刻**看到「`unit` 不是合法参数」，
/// 而不是悄悄忽略一个拼错的参数导致查错地方。
pub fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名，例如 \"北京\"、\"Shanghai\"、\"Tokyo\"。必须是城市名，不是省或国家。"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
                "description": "温度单位。celsius 摄氏，fahrenheit 华氏。"
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7,
                "default": 1,
                "description": "预报天数，1 表示只看当前实况，2~7 表示今天起未来几天的预报。"
            }
        },
        "required": ["location"],
        "additionalProperties": false
    })
}

/// 温度单位。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Units {
    /// 摄氏度。
    Celsius,
    /// 华氏度。
    Fahrenheit,
}

impl Units {
    /// 从字符串解析。
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "celsius" | "c" | "centigrade" | "metric" => Some(Units::Celsius),
            "fahrenheit" | "f" | "°f" => Some(Units::Fahrenheit),
            _ => None,
        }
    }

    /// 符号后缀。
    pub fn symbol(self) -> &'static str {
        match self {
            Units::Celsius => "°C",
            Units::Fahrenheit => "°F",
        }
    }

    /// 温度单位名（给模型看的）。
    pub fn label(self) -> &'static str {
        match self {
            Units::Celsius => "摄氏度",
            Units::Fahrenheit => "华氏度",
        }
    }

    /// 摄氏 → 目标单位。
    pub fn convert(self, celsius: f64) -> f64 {
        match self {
            Units::Celsius => celsius,
            Units::Fahrenheit => celsius * 9.0 / 5.0 + 32.0,
        }
    }
}

/// 天气工具。
pub struct WeatherTool {
    client: reqwest::Client,
    geocoding_url: String,
    forecast_url: String,
    default_days: usize,
    max_days: usize,
}

impl std::fmt::Debug for WeatherTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeatherTool")
            .field("geocoding_url", &self.geocoding_url)
            .field("forecast_url", &self.forecast_url)
            .field("max_days", &self.max_days)
            .finish()
    }
}

impl WeatherTool {
    /// 创建天气工具（连接真实的 Open-Meteo 服务）。
    pub fn new(default_days: usize, max_days: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("agent-learn/14-agent-tool")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            client,
            geocoding_url: GEOCODING_URL.to_string(),
            forecast_url: FORECAST_URL.to_string(),
            default_days: default_days.clamp(1, 7),
            max_days: max_days.clamp(1, 7),
        }
    }

    /// 用自定义端点创建（测试用 —— 打到一个假的 HTTP 服务）。
    pub fn with_endpoints(
        geocoding_url: impl Into<String>,
        forecast_url: impl Into<String>,
        default_days: usize,
        max_days: usize,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            client,
            geocoding_url: geocoding_url.into(),
            forecast_url: forecast_url.into(),
            default_days: default_days.clamp(1, 7),
            max_days: max_days.clamp(1, 7),
        }
    }

    /// 默认预报天数。
    pub fn default_days(&self) -> usize {
        self.default_days
    }

    /// 最大预报天数。
    pub fn max_days(&self) -> usize {
        self.max_days
    }

    /// ① 地理编码：地名 → 坐标。
    ///
    /// ## 失败语义
    ///
    /// 查不到城市**不是**上游错误 —— 上游活得好好的，
    /// 是**参数**不对（城市名拼错 / 不存在）。
    /// 所以返回 [`ToolError::Execution`] 而不是 [`ToolError::Upstream`]。
    async fn geocode(&self, location: &str) -> Result<GeoLocation, ToolError> {
        let url = format!(
            "{}?name={}&count=1&language=zh&format=json",
            self.geocoding_url,
            urlencode(location)
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::upstream(format!("连接地理编码服务失败: {e}")))?;

        if !response.status().is_success() {
            return Err(ToolError::upstream(format!(
                "地理编码服务返回 HTTP {}",
                response.status()
            )));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| ToolError::upstream(format!("解析地理编码响应失败: {e}")))?;

        let results = body
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| ToolError::execution(GET_WEATHER_TOOL_NAME, format!("找不到城市 `{location}`。请使用常见的城市名，例如「北京」「上海」「东京」。")))?;

        let first = results.first().ok_or_else(|| {
            ToolError::execution(
                GET_WEATHER_TOOL_NAME,
                format!(
                    "找不到城市 `{location}`。请使用常见的城市名，例如「北京」「上海」「东京」。"
                ),
            )
        })?;

        let latitude = first
            .get("latitude")
            .and_then(Value::as_f64)
            .ok_or_else(|| ToolError::upstream("地理编码响应缺少 latitude 字段".to_string()))?;
        let longitude = first
            .get("longitude")
            .and_then(Value::as_f64)
            .ok_or_else(|| ToolError::upstream("地理编码响应缺少 longitude 字段".to_string()))?;

        Ok(GeoLocation {
            name: first
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(location)
                .to_string(),
            country: first
                .get("country")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            admin1: first
                .get("admin1")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            latitude,
            longitude,
        })
    }

    /// ② 取预报：坐标 → 天气数据。
    async fn fetch_forecast(
        &self,
        geo: &GeoLocation,
        units: Units,
        days: usize,
    ) -> Result<Forecast, ToolError> {
        let current_fields =
            "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m";
        let daily_fields =
            "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max";

        let url = format!(
            "{}?latitude={}&longitude={}&current={}&daily={}&timezone=auto&forecast_days={}",
            self.forecast_url, geo.latitude, geo.longitude, current_fields, daily_fields, days
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::upstream(format!("连接天气服务失败: {e}")))?;

        if !response.status().is_success() {
            return Err(ToolError::upstream(format!(
                "天气服务返回 HTTP {}",
                response.status()
            )));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| ToolError::upstream(format!("解析天气响应失败: {e}")))?;

        Forecast::from_json(&body, geo, units)
    }
}

/// 地理编码结果。
#[derive(Debug, Clone)]
pub struct GeoLocation {
    /// 城市名。
    pub name: String,
    /// 国家。
    pub country: String,
    /// 一级行政区（省 / 州）。
    pub admin1: String,
    /// 纬度。
    pub latitude: f64,
    /// 经度。
    pub longitude: f64,
}

impl GeoLocation {
    /// 完整地名（给模型看）。
    pub fn full_name(&self) -> String {
        let mut parts = vec![self.name.clone()];
        if !self.admin1.is_empty() && self.admin1 != self.name {
            parts.push(self.admin1.clone());
        }
        if !self.country.is_empty() {
            parts.push(self.country.clone());
        }
        parts.join(", ")
    }
}

/// 一天的预报。
#[derive(Debug, Clone)]
pub struct DailyForecast {
    /// 预报日期。
    pub date: String,
    /// 天气代码（WMO）。
    pub weather_code: i32,
    /// 最高温（摄氏度）。
    pub temp_max_c: f64,
    /// 最低温（摄氏度）。
    pub temp_min_c: f64,
    /// 最大降水概率（%）。
    pub precipitation_probability: Option<i32>,
}

/// 天气数据。
#[derive(Debug, Clone)]
pub struct Forecast {
    /// 地点。
    pub location: GeoLocation,
    /// 观测时间。
    pub time: String,
    /// 当前温度（摄氏度）。
    pub temperature_c: f64,
    /// 体感温度（摄氏度）。
    pub apparent_temperature_c: Option<f64>,
    /// 相对湿度（%）。
    pub humidity: Option<i32>,
    /// 天气代码（WMO）。
    pub weather_code: i32,
    /// 风速（km/h）。
    pub wind_speed: Option<f64>,
    /// 每日预报。
    pub daily: Vec<DailyForecast>,
    /// 目标温度单位。
    pub units: Units,
}

impl Forecast {
    /// 从 Open-Meteo 的 JSON 响应里解析。
    fn from_json(body: &Value, geo: &GeoLocation, units: Units) -> Result<Self, ToolError> {
        let current = body
            .get("current")
            .ok_or_else(|| ToolError::upstream("天气响应缺少 current 字段".to_string()))?;

        let temperature_c = current
            .get("temperature_2m")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                ToolError::upstream("天气响应缺少 current.temperature_2m".to_string())
            })?;

        let weather_code = current
            .get("weather_code")
            .and_then(Value::as_i64)
            .unwrap_or(0) as i32;

        let time = current
            .get("time")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let apparent_temperature_c = current.get("apparent_temperature").and_then(Value::as_f64);
        let humidity = current
            .get("relative_humidity_2m")
            .and_then(Value::as_i64)
            .map(|v| v as i32);
        let wind_speed = current.get("wind_speed_10m").and_then(Value::as_f64);

        // 每日预报。
        let mut daily = Vec::new();
        if let Some(daily_block) = body.get("daily")
            && let Some(dates) = daily_block.get("time").and_then(Value::as_array)
        {
            let maxes = daily_block
                .get("temperature_2m_max")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mins = daily_block
                .get("temperature_2m_min")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let codes = daily_block
                .get("weather_code")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let probs = daily_block
                .get("precipitation_probability_max")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            for (i, date) in dates.iter().enumerate() {
                let Some(date) = date.as_str() else { continue };
                daily.push(DailyForecast {
                    date: date.to_string(),
                    weather_code: codes.get(i).and_then(Value::as_i64).unwrap_or(0) as i32,
                    temp_max_c: maxes.get(i).and_then(Value::as_f64).unwrap_or(f64::NAN),
                    temp_min_c: mins.get(i).and_then(Value::as_f64).unwrap_or(f64::NAN),
                    precipitation_probability: probs
                        .get(i)
                        .and_then(Value::as_i64)
                        .map(|v| v as i32),
                });
            }
        }

        Ok(Forecast {
            location: geo.clone(),
            time,
            temperature_c,
            apparent_temperature_c,
            humidity,
            weather_code,
            wind_speed,
            daily,
            units,
        })
    }

    /// 渲染成给模型看的观察文本。
    ///
    /// ## 格式为什么这样设计
    ///
    /// - **第一行就是结论** —— 温度 + 天气状况。
    ///   模型读 Observation 时最先看到的是最有用的信息。
    /// - **字段用 `标签: 值`** —— 模型能引用具体字段，
    ///   最终答案里写「18°C」比写「挺凉的」有用得多。
    /// - **数字统一用目标单位** —— 转换在服务端做，
    ///   模型不用自己算（13 的 MRKL 结论：LLM 不擅长算）。
    pub fn render(&self) -> String {
        let mut out = String::new();
        let unit = self.units.symbol();

        out.push_str(&format!(
            "{} 当前实况：{}，{}{}，体感 {}{}\n",
            self.location.full_name(),
            weather_text(self.weather_code),
            self.units.convert(self.temperature_c).round(),
            unit,
            self.apparent_temperature_c
                .map(|t| self.units.convert(t).round())
                .unwrap_or_else(|| self.units.convert(self.temperature_c).round()),
            unit,
        ));

        if let Some(h) = self.humidity {
            out.push_str(&format!("相对湿度: {h}%\n"));
        }
        if let Some(w) = self.wind_speed {
            out.push_str(&format!("风速: {w} km/h\n"));
        }
        if !self.time.is_empty() {
            out.push_str(&format!("观测时间: {}\n", self.time));
        }

        // 每日预报。
        for (i, day) in self.daily.iter().enumerate() {
            if i == 0 {
                // 第一天就是「今天」—— 当前实况已经说过了，不重复。
                continue;
            }
            let mut line = format!(
                "{} {}：{}，{} ~ {}{}",
                day.date,
                relative_day(i),
                weather_text(day.weather_code),
                self.units.convert(day.temp_min_c).round(),
                self.units.convert(day.temp_max_c).round(),
                unit
            );
            if let Some(p) = day.precipitation_probability {
                line.push_str(&format!("，降水概率 {p}%"));
            }
            out.push_str(&line);
            out.push('\n');
        }

        out.trim_end().to_string()
    }

    /// 结构化数据（进 API 响应，不进模型上下文）。
    pub fn to_json(&self) -> Value {
        json!({
            "location": {
                "name": self.location.name,
                "admin1": self.location.admin1,
                "country": self.location.country,
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
            },
            "time": self.time,
            "units": match self.units {
                Units::Celsius => "celsius",
                Units::Fahrenheit => "fahrenheit",
            },
            "current": {
                "temperature": self.units.convert(self.temperature_c).round(),
                "apparent_temperature": self.apparent_temperature_c
                    .map(|t| self.units.convert(t).round()),
                "humidity": self.humidity,
                "weather_code": self.weather_code,
                "weather_text": weather_text(self.weather_code),
                "wind_speed": self.wind_speed,
            },
            "daily": self.daily.iter().map(|d| json!({
                "date": d.date,
                "weather_code": d.weather_code,
                "weather_text": weather_text(d.weather_code),
                "temp_max": self.units.convert(d.temp_max_c).round(),
                "temp_min": self.units.convert(d.temp_min_c).round(),
                "precipitation_probability": d.precipitation_probability,
            })).collect::<Vec<_>>(),
        })
    }
}

/// 第 n 天（从 1 开始）的相对说法。
fn relative_day(index: usize) -> &'static str {
    match index {
        1 => "明天",
        2 => "后天",
        3 => "3 天后",
        4 => "4 天后",
        5 => "5 天后",
        6 => "6 天后",
        _ => "7 天后",
    }
}

/// WMO 天气代码 → 中文描述。
///
/// ## 为什么在工具里做这个转换
///
/// 模型看到 `weather_code: 61` 没什么用 —— 它不知道那是「小雨」。
/// 转换后模型能直接说「今天小雨」，
/// 最终答案的质量明显更好。
///
/// 这也对应 LangChain 教程里「返回结构化对象让模型推理」的思想 ——
/// 但这里更进一步：**结构化 → 人类可读**的转换也放在工具里，
/// 因为那是数据语义的一部分，不该让模型去猜。
pub fn weather_text(code: i32) -> &'static str {
    match code {
        0 => "晴",
        1 => "大部晴朗",
        2 => "局部多云",
        3 => "阴",
        45 | 48 => "雾",
        51 | 53 | 55 => "毛毛雨",
        56 | 57 => "冻毛毛雨",
        61 => "小雨",
        63 => "中雨",
        65 => "大雨",
        66 | 67 => "冻雨",
        71 => "小雪",
        73 => "中雪",
        75 => "大雪",
        77 => "雪粒",
        80 => "阵雨",
        81 => "强阵雨",
        82 => "暴雨",
        85 | 86 => "阵雪",
        95 => "雷阵雨",
        96 | 99 => "雷阵雨伴冰雹",
        _ => "未知天气",
    }
}

/// 简单的 URL 编码（只处理查询串里常见的字符）。
fn urlencode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char);
            }
            b' ' => out.push('+'),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

impl Tool for WeatherTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            GET_WEATHER_TOOL_NAME,
            "查询某个城市的实时天气和未来几天的预报。\
             需要实时的天气信息（温度、湿度、风力、降水）时必须调用这个工具，\
             不要凭记忆回答天气问题。",
            weather_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let location = args
                .get("location")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if location.is_empty() {
                warn!("get_weather 缺少 location 参数");
                return Err(ToolError::invalid_args(vec![
                    "`location` 不能为空（说明：城市名，例如 \"北京\"）".to_string(),
                ]));
            }

            let units = match args.get("units").and_then(Value::as_str) {
                Some(raw) => Units::parse(raw).unwrap_or(Units::Celsius),
                None => Units::Celsius,
            };

            let days = args
                .get("days")
                .and_then(Value::as_i64)
                .map(|v| v as usize)
                .unwrap_or(self.default_days)
                .clamp(1, self.max_days);

            // ① 地理编码。
            info!(location = %location, units = units.symbol(), days, "get_weather：地理编码");
            let geo = self.geocode(&location).await?;
            debug!(
                location = %geo.full_name(),
                latitude = geo.latitude,
                longitude = geo.longitude,
                "地理编码完成"
            );

            // ② 预报。
            info!("get_weather：取预报");
            let forecast = self.fetch_forecast(&geo, units, days).await?;

            let observation = forecast.render();
            let data = forecast.to_json();

            info!(
                location = %forecast.location.full_name(),
                temperature_c = forecast.temperature_c,
                daily_count = forecast.daily.len(),
                "get_weather 完成"
            );

            // 上下文里的超时提示 —— 目前只用于日志，不改变行为。
            let _ = ctx;

            Ok(ToolOutcome::ok_with_data(
                GET_WEATHER_TOOL_NAME,
                observation,
                data,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_requires_location_and_limits_days() {
        let schema = weather_schema();
        assert_eq!(schema["required"], json!(["location"]));
        assert_eq!(schema["properties"]["days"]["maximum"], json!(7));
        assert_eq!(schema["properties"]["units"]["default"], json!("celsius"));
    }

    #[test]
    fn parses_units() {
        assert_eq!(Units::parse("celsius"), Some(Units::Celsius));
        assert_eq!(Units::parse("Fahrenheit"), Some(Units::Fahrenheit));
        assert_eq!(Units::parse("kelvin"), None);
    }

    #[test]
    fn converts_temperature() {
        assert_eq!(Units::Celsius.convert(20.0), 20.0);
        assert!((Units::Fahrenheit.convert(20.0) - 68.0).abs() < 1e-9);
        assert!((Units::Fahrenheit.convert(0.0) - 32.0).abs() < 1e-9);
    }

    #[test]
    fn maps_wmo_codes_to_chinese() {
        assert_eq!(weather_text(0), "晴");
        assert_eq!(weather_text(61), "小雨");
        assert_eq!(weather_text(95), "雷阵雨");
        assert_eq!(weather_text(9999), "未知天气");
    }

    #[test]
    fn renders_forecast_readable() {
        let geo = GeoLocation {
            name: "北京".to_string(),
            country: "中国".to_string(),
            admin1: "北京市".to_string(),
            latitude: 39.9,
            longitude: 116.4,
        };
        let forecast = Forecast {
            location: geo,
            time: "2026-11-08T12:00".to_string(),
            temperature_c: 18.5,
            apparent_temperature_c: Some(17.0),
            humidity: Some(45),
            weather_code: 0,
            wind_speed: Some(9.7),
            daily: vec![
                DailyForecast {
                    date: "2026-11-08".to_string(),
                    weather_code: 0,
                    temp_max_c: 20.0,
                    temp_min_c: 8.0,
                    precipitation_probability: Some(0),
                },
                DailyForecast {
                    date: "2026-11-09".to_string(),
                    weather_code: 61,
                    temp_max_c: 16.0,
                    temp_min_c: 9.0,
                    precipitation_probability: Some(70),
                },
            ],
            units: Units::Celsius,
        };

        let rendered = forecast.render();

        // 第一行必须有结论。
        assert!(rendered.starts_with("北京, 北京市, 中国 当前实况：晴"));
        assert!(rendered.contains("19°C"));
        assert!(rendered.contains("湿度: 45%"));
        assert!(rendered.contains("风速: 9.7 km/h"));
        // 明天要有降水概率。
        assert!(rendered.contains("明天"));
        assert!(rendered.contains("降水概率 70%"));
        // 今天不重复出现（当前实况已经说了）。
        assert_eq!(rendered.matches("2026-11-08").count(), 1);
    }

    #[test]
    fn renders_fahrenheit_conversion() {
        let geo = GeoLocation {
            name: "纽约".to_string(),
            country: "美国".to_string(),
            admin1: "纽约州".to_string(),
            latitude: 40.7,
            longitude: -74.0,
        };
        let forecast = Forecast {
            location: geo,
            time: "2026-11-08T12:00".to_string(),
            temperature_c: 20.0,
            apparent_temperature_c: None,
            humidity: None,
            weather_code: 3,
            wind_speed: None,
            daily: vec![],
            units: Units::Fahrenheit,
        };

        let rendered = forecast.render();
        assert!(rendered.contains("68°F"));
    }

    #[test]
    fn urlencodes_spaces_and_chinese() {
        assert_eq!(urlencode("New York"), "New+York");
        assert_eq!(urlencode("北京"), "%E5%8C%97%E4%BA%AC");
    }

    #[tokio::test]
    async fn missing_location_is_reported_as_invalid_args() {
        let tool = WeatherTool::new(1, 7);
        let outcome = tool
            .call(&json!({}), &ToolContext::empty())
            .await
            .unwrap_err();

        assert!(outcome.to_string().contains("location"));
    }

    #[test]
    fn structured_data_has_current_and_daily() {
        let geo = GeoLocation {
            name: "上海".to_string(),
            country: "中国".to_string(),
            admin1: "上海市".to_string(),
            latitude: 31.2,
            longitude: 121.5,
        };
        let forecast = Forecast {
            location: geo,
            time: "t".to_string(),
            temperature_c: 12.0,
            apparent_temperature_c: None,
            humidity: Some(60),
            weather_code: 3,
            wind_speed: None,
            daily: vec![],
            units: Units::Celsius,
        };

        let data = forecast.to_json();
        assert_eq!(data["current"]["temperature"], json!(12.0));
        assert_eq!(data["current"]["weather_text"], json!("阴"));
        assert!(data["daily"].is_array());
    }
}
