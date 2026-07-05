// DS4 (DwarfStar) â€” HTTP server binary.
//
// Replaces `ds4_server.c` (15875 LoC). Provides:
//   - OpenAI /v1/chat/completions
//   - OpenAI /v1/completions
//   - OpenAI /v1/models
//   - OpenAI /v1/responses
//   - Anthropic /v1/messages
//   - SSE streaming
//   - DSML <-> OpenAI/Anthropic tool-call transcoding
//
// GGUF models serve token generation through the Rust engine pool when
// the selected backend can load them. Requests without a loaded runtime
// model fail closed with explicit errors.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use ds4_server::{serve, EnginePool, PoolConfig, ServerState, MAX_CONTEXT_TOKENS};

const CRATE_NAME: &str = "ds4-server";
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "ds4-server", version, about = "DwarfStar HTTP server", long_about = None)]
struct Cli {
    /// Path to a DeepSeek V4 Flash / PRO GGUF produced for DwarfStar.
    #[arg(long)]
    model: PathBuf,

    /// Path to the MTP head weights (optional).
    #[arg(long)]
    mtp: Option<PathBuf>,

    /// Listen host.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Listen port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Context size in tokens.
    #[arg(long, default_value_t = 8192)]
    ctx: usize,

    /// CPU threads.
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Tokens per prefill chunk.
    #[arg(long, default_value_t = 512)]
    prefill_chunk: usize,

    /// Override the model id reported by /v1/models.
    #[arg(long, default_value = "ds4")]
    model_id: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    log::info!("{CRATE_NAME} {VERSION}");
    log::info!(
        "model={:?} listen={}:{} ctx={} threads={} prefill_chunk={}",
        cli.model,
        cli.host,
        cli.port,
        cli.ctx,
        cli.threads,
        cli.prefill_chunk,
    );

    if cli.ctx == 0 || cli.ctx > MAX_CONTEXT_TOKENS {
        eprintln!(
            "invalid --ctx {}; expected 1..={}",
            cli.ctx, MAX_CONTEXT_TOKENS
        );
        return ExitCode::from(2);
    }
    if cli.prefill_chunk == 0 {
        eprintln!("invalid --prefill-chunk 0; expected a positive value");
        return ExitCode::from(2);
    }

    let pool_cfg = PoolConfig {
        model: cli.model.clone(),
        mtp: cli.mtp.clone(),
        ctx: cli.ctx,
        prefill_chunk: cli.prefill_chunk,
        n_threads: cli.threads,
    };
    let pool = match EnginePool::open(pool_cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to open engine pool: {e}");
            return ExitCode::from(1);
        }
    };
    if pool.is_engine_unavailable() {
        log::warn!(
            "the selected model/backend has no loaded runtime model; inference routes return 501 until a model is loaded"
        );
    }
    let state = Arc::new(ServerState::new(pool, cli.model_id.clone()));

    let listener = match bind_listener(&cli.host, cli.port).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {}:{}: {e:?}", cli.host, cli.port);
            return ExitCode::from(1);
        }
    };
    let local = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| format!("{}:{}", cli.host, cli.port));
    log::info!("listening on {local}");

    if let Err(e) = serve(listener, state).await {
        eprintln!("server error: {e:?}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

async fn bind_listener(host: &str, port: u16) -> Result<tokio::net::TcpListener> {
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    Ok(listener)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_constants_are_sane() {
        assert_eq!(CRATE_NAME, "ds4-server");
        assert!(!VERSION.is_empty());
    }
}
