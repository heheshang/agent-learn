//! 模糊测试目标 —— 21-agent-cache 的 **缓存键构造**。
//!
//! ## 这个目标在找什么
//!
//! 1. **panic** —— 缓存键在每次 LLM 调用前都要算一次，挂了就是全挂。
//! 2. **不稳定** —— 同样输入两次算出不同的键，
//!    症状是「缓存命中率恒为 0」，但没有任何报错，
//!    极难定位（21 的 README 里专门记了这个坑：
//!    不能用 `DefaultHasher`，因为它跨进程不稳定）。
//! 3. **键形态非法** —— Redis 键里含空格 / 换行 / 控制字符会让日志错乱，
//!    含 `:` 会破坏「命名空间:版本:摘要」的三段结构。

#![no_main]

use libfuzzer_sys::fuzz_target;
use agent_21::cache::key::{CACHE_KEY_VERSION, CacheKey};
use genai::chat::{ChatMessage, MessageContent};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    // 造一条 user 消息（消息内容就是 fuzz 输入）。
    let message = ChatMessage::user(MessageContent::from_text(text.to_string()));

    let key = CacheKey::new(
        "agent-cache",
        "deepseek-flash",
        &[message],
        "digest",
        "auto",
        0.0,
        2048,
        true,
    );

    // 1. 不能 panic。
    let redis_key = key.redis_key();

    // 2. 摘要必须稳定。
    assert_eq!(
        key.digest(),
        key.digest(),
        "摘要必须稳定（input={text:?}）"
    );

    // 3. 键必须稳定 —— 同输入两次算出的 Redis 键必须一致。
    let key2 = CacheKey::new(
        "agent-cache",
        "deepseek-flash",
        &[ChatMessage::user(MessageContent::from_text(text.to_string()))],
        "digest",
        "auto",
        0.0,
        2048,
        true,
    );
    assert_eq!(
        redis_key,
        key2.redis_key(),
        "同输入必须得到同键（input={text:?}）"
    );

    // 4. 键形态必须是三段 `{namespace}:{version}:{16 位十六进制}`。
    let parts: Vec<&str> = redis_key.split(':').collect();
    assert_eq!(parts.len(), 3, "键必须是三段结构（input={text:?}）");
    assert_eq!(parts[0], "agent-cache");
    assert_eq!(parts[1], CACHE_KEY_VERSION);
    assert_eq!(parts[2].len(), 16, "摘要段必须 16 位（input={text:?}）");
    assert!(
        parts[2].chars().all(|c| c.is_ascii_hexdigit()),
        "摘要段必须全十六进制（input={text:?}）"
    );

    // 5. 键里不能有空白 / 控制字符（否则日志与 Redis 排障都会很难受）。
    assert!(
        !redis_key.chars().any(|c| c.is_whitespace() || c.is_control()),
        "键里不能有空白或控制字符（input={text:?}）"
    );
});
