// DS4 (DwarfStar) â€” HTTP server crate.
//
// Public surface:
//   - `EnginePool`, `JobResult`, `SamplingParams` - engine front-end
//   - `run_server` â€” bind TcpListener and serve the routing table
//   - `ServerState`, `serve` â€” hyper service implementation
//   - Re-exports for the wire-format types (OpenAI + Anthropic + DSML)

pub mod anthropic;
pub mod dsml;
pub mod engine_pool;
pub mod id;
pub mod openai;
pub mod routes;
pub mod streaming;

pub use engine_pool::{
    EnginePool, EngineStatus, Job, JobResult, PoolConfig, SamplingParams, TokenUsage,
    MAX_CONTEXT_TOKENS,
};
pub use routes::{serve, ServerState};
