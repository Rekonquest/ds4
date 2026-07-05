// DS4 (DwarfStar) â€” CLI one-shot mode.
//
// Build a chat prompt, sync it into a fresh `Ds4Session`, sample +
// decode, then print. Same flow the C side runs through `ds4-cli
// --prompt ...`.

use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use ds4_core::{engine::Ds4Engine, session::Ds4Session};
use ds4_types::{Ds4Backend, Ds4EngineOptions, Ds4ErrorKind, Ds4RewriteStatus};
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct OneShotConfig {
    pub model: PathBuf,
    pub mtp: Option<PathBuf>,
    pub ctx: usize,
    pub prefill_chunk: usize,
    pub n_threads: usize,
    pub backend: Ds4Backend,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub n_predict: usize,
    pub system: Option<String>,
    pub prompt: String,
    pub display: bool,
}

impl OneShotConfig {
    pub fn engine_options(&self) -> Ds4EngineOptions {
        Ds4EngineOptions {
            model_path: self.model.clone(),
            mtp_path: self.mtp.clone(),
            backend: self.backend,
            n_threads: self.n_threads,
            prefill_chunk: self.prefill_chunk,
            ..Ds4EngineOptions::default()
        }
    }
}

/// Read a prompt from stdin (one-shot mode `--prompt-stdin`).
pub fn read_prompt_stdin() -> Result<String> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf.trim_end().to_string())
}

/// Entry point: run a single prompt against the engine and print the
/// reply to stdout. Returns an error if the engine fails to open.
pub fn run(cfg: OneShotConfig) -> Result<()> {
    let opts = cfg.engine_options();
    let engine = Arc::new(Ds4Engine::open(opts).map_err(|e| {
        anyhow!(
            "failed to open engine at {:?}: {:?}: {}",
            cfg.model,
            e.kind,
            e.message
        )
    })?);
    let model_name = engine.model_name().to_string();
    log::info!(
        "engine ready (model={model_name:?}, layers={})",
        engine.layer_count()
    );
    let mut session = Ds4Session::create(&engine, cfg.ctx)
        .map_err(|e| anyhow!("failed to create session: {:?}: {}", e.kind, e.message))?;

    let mut prompt = cfg.prompt.clone();
    if prompt.is_empty() {
        prompt = read_prompt_stdin().context("reading prompt from stdin")?;
    }
    if prompt.is_empty() {
        bail!("empty prompt; nothing to do");
    }

    let tokens = engine.chat().encode_prompt(
        cfg.system.as_deref().unwrap_or(""),
        &prompt,
        ds4_types::Ds4ThinkMode::None,
    );
    log::info!("prompt tokens: {}", tokens.len());

    run_decode_loop(&engine, &mut session, &tokens, &cfg)
}

fn run_decode_loop(
    _engine: &Arc<Ds4Engine>,
    session: &mut Ds4Session,
    prompt: &[u32],
    cfg: &OneShotConfig,
) -> Result<()> {
    session
        .sync(prompt)
        .map_err(|e| anyhow!("sync failed: {:?}: {}", e.kind, e.message))?;
    let status = session.rewrite_from_common(prompt, session.common_prefix(prompt));
    if matches!(status, Ds4RewriteStatus::RebuildNeeded) {
        log::warn!("session rewrite reported RebuildNeeded; KV may be partial");
    }
    session
        .refresh_logits()
        .map_err(|e| anyhow!("logits refresh failed: {:?}: {}", e.kind, e.message))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.unwrap_or(0x4453_3452));
    let eos = _engine.tokenizer().eos_id();
    let mut stdout = std::io::stdout();
    for _ in 0..cfg.n_predict {
        if session.pos() >= session.ctx() {
            break;
        }
        let tok = session.sample(cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p, &mut rng);
        if tok == eos {
            break;
        }
        if cfg.display {
            let text = render_token(_engine, tok)?;
            stdout.write_all(text.as_bytes())?;
            stdout.flush()?;
        }
        session
            .eval(tok)
            .map_err(|e| anyhow!("eval failed: {:?}: {}", e.kind, e.message))?;
    }
    if cfg.display {
        stdout.write_all(b"\n")?;
    }
    Ok(())
}

pub(crate) fn render_token(engine: &Arc<Ds4Engine>, token: u32) -> Result<String> {
    engine
        .tokenizer()
        .detokenize(&[token])
        .map_err(|e| anyhow!("detokenize failed: {:?}: {}", e.kind, e.message))
}

pub fn engine_missing_model_message() -> &'static str {
    "the selected model/backend has no loaded runtime model"
}

pub fn is_engine_pending(err: &ds4_types::Ds4Error) -> bool {
    err.kind == Ds4ErrorKind::NotImplemented
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_missing_model_message_mentions_runtime_state() {
        let m = engine_missing_model_message();
        assert!(m.contains("selected model/backend"));
        assert!(m.contains("loaded runtime model"));
    }

    #[test]
    fn engine_options_use_config() {
        let cfg = OneShotConfig {
            model: PathBuf::from("/tmp/m.gguf"),
            mtp: None,
            ctx: 4096,
            prefill_chunk: 256,
            n_threads: 8,
            backend: Ds4Backend::Cpu,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            seed: Some(42),
            n_predict: 128,
            system: None,
            prompt: "hi".to_string(),
            display: true,
        };
        let opts = cfg.engine_options();
        assert_eq!(opts.model_path, PathBuf::from("/tmp/m.gguf"));
        assert_eq!(opts.n_threads, 8);
        assert_eq!(opts.prefill_chunk, 256);
    }

    #[test]
    fn render_token_decodes_through_engine_tokenizer() {
        let dir = std::env::temp_dir().join(format!("ds4-cli-render-token-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("synth.gguf");
        Ds4Engine::write_synthetic_gguf(&model).unwrap();
        let engine = Arc::new(
            Ds4Engine::open(Ds4EngineOptions {
                model_path: model,
                ..Ds4EngineOptions::default()
            })
            .unwrap(),
        );
        assert_eq!(render_token(&engine, 8).unwrap(), "h");
    }
}
