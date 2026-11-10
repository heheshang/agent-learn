//! HTTP API 层。
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET | `/` | 服务信息 + 参数说明 |
//! | GET | `/health` | 健康检查 |
//! | GET | `/api/modes` | ★ 批处理模式说明（静态 vs 连续） |
//! | POST | `/api/simulate` | ★ 用调度器模拟跑一次负载 |
//! | POST | `/api/bench` | ★ ★ ★ 三组对照实验 |
//! | POST | `/api/mock` | ★ 启动/切换 mock vLLM 服务 |

pub mod handlers;
pub mod models;
pub mod routes;

pub use routes::router;
