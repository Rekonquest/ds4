// DS4 (DwarfStar) â€” CLI REPL mode.
//
// Persistent `Ds4Session` so successive prompts reuse the live KV
// checkpoint. Backed by `rustyline` for editing + history; we wire
// multiline input via Rustyline's built-in escape sequences and
// keep Ctrl-C / EOF handling explicit.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ds4_core::{engine::Ds4Engine, session::Ds4Session};
use ds4_types::{Ds4Backend, Ds4EngineOptions, Ds4ErrorKind};
use rand::SeedableRng;

use crate::one_shot::OneShotConfig;

const PROMPT: &str = "ds4> ";

/// REPL configuration. Mirrors `OneShotConfig` for the bits the
/// REPL needs; we reuse the engine-opening code path.
#[derive(Debug, Clone)]
pub struct ReplConfig {
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
}

impl ReplConfig {
    pub fn to_one_shot(&self) -> OneShotConfig {
        OneShotConfig {
            model: self.model.clone(),
            mtp: self.mtp.clone(),
            ctx: self.ctx,
            prefill_chunk: self.prefill_chunk,
            n_threads: self.n_threads,
            backend: self.backend,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            seed: self.seed,
            n_predict: self.n_predict,
            system: self.system.clone(),
            prompt: String::new(),
            display: true,
        }
    }
}

/// Run the REPL loop. Returns `Ok(())` on graceful EOF, `Err` on
/// fatal engine failures.
pub fn run(cfg: ReplConfig) -> Result<()> {
    let opts: Ds4EngineOptions = cfg.to_one_shot().engine_options();
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

    if engine_pending(&engine) {
        eprintln!("the selected model/backend has no loaded runtime model");
        eprintln!("REPL prompts require a loaded CPU-compatible model/backend.");
    }

    let mut rl = rustyline::DefaultEditor::new()
        .map_err(|e| anyhow!("failed to open rustyline editor: {e}"))?;
    let history_path = dirs_history_path();
    if let Some(path) = history_path.as_ref() {
        let _ = rl.load_history(path);
    }

    loop {
        let line = match rl.readline(PROMPT) {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                // Ctrl-C: clear current buffer, stay in REPL.
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                // Ctrl-D: exit REPL gracefully.
                break;
            }
            Err(e) => {
                eprintln!("readline error: {e}");
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_repl_command(trimmed) {
            handle_repl_command(trimmed, &mut session, &engine, &cfg)?;
            continue;
        }
        rl.add_history_entry(trimmed).ok();
        if let Err(e) = dispatch_prompt(trimmed, &mut session, &engine, &cfg) {
            eprintln!("error: {e}");
            // Missing runtime models are not fatal: keep the REPL alive.
            if !is_engine_pending_err(&e) {
                // Swallow per-turn errors so the REPL stays
                // usable after a recoverable prompt error.
            }
        }
        println!();
    }

    if let Some(path) = history_path.as_ref() {
        let _ = rl.save_history(path);
    }
    Ok(())
}

fn dispatch_prompt(
    prompt: &str,
    session: &mut Ds4Session,
    engine: &Arc<Ds4Engine>,
    cfg: &ReplConfig,
) -> Result<()> {
    let tokens = engine.chat().encode_prompt(
        cfg.system.as_deref().unwrap_or(""),
        prompt,
        ds4_types::Ds4ThinkMode::None,
    );
    session
        .sync(&tokens)
        .map_err(|e| anyhow!("sync failed: {:?}: {}", e.kind, e.message))?;
    session
        .refresh_logits()
        .map_err(|e| anyhow!("logits refresh failed: {:?}: {}", e.kind, e.message))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.unwrap_or(0x4453_3452));
    let eos = engine.tokenizer().eos_id();
    for _ in 0..cfg.n_predict {
        if session.pos() >= session.ctx() {
            break;
        }
        let tok = session.sample(cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p, &mut rng);
        if tok == eos {
            break;
        }
        print!("{}", crate::one_shot::render_token(engine, tok)?);
        session
            .eval(tok)
            .map_err(|e| anyhow!("eval failed: {:?}: {}", e.kind, e.message))?;
    }
    Ok(())
}

fn is_repl_command(line: &str) -> bool {
    matches!(
        line,
        "/exit" | "/quit" | "/bye" | "/clear" | "/reset" | "/help" | "/?"
    )
}

fn handle_repl_command(
    line: &str,
    session: &mut Ds4Session,
    _engine: &Arc<Ds4Engine>,
    _cfg: &ReplConfig,
) -> Result<()> {
    match line {
        "/exit" | "/quit" | "/bye" => {
            // Treat as EOF for the outer loop.
            std::process::exit(0);
        }
        "/clear" => {
            // Best-effort clear; session reset clears the KV cache.
            session.invalidate();
            println!("session reset.");
        }
        "/reset" => {
            session.rewind(0);
            println!("session rewound to position 0.");
        }
        "/help" | "/?" => {
            println!("REPL commands:");
            println!("  /exit, /quit, /bye    Exit the REPL.");
            println!("  /clear                Reset the session (drops KV).");
            println!("  /reset                Rewind the session to position 0.");
            println!("  /help, /?             Show this help.");
            println!("Enter any other text to send it as a prompt.");
        }
        _ => {}
    }
    Ok(())
}

fn dirs_history_path() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".ds4_history"))
}

fn engine_pending(engine: &Arc<Ds4Engine>) -> bool {
    engine.model().is_none()
}

fn is_engine_pending_err(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<ds4_types::Ds4Error>()
            .map(|de| de.kind == Ds4ErrorKind::NotImplemented)
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repl_command_recognised() {
        assert!(is_repl_command("/exit"));
        assert!(is_repl_command("/quit"));
        assert!(is_repl_command("/help"));
        assert!(is_repl_command("/?"));
        assert!(!is_repl_command("hello"));
        assert!(!is_repl_command("/unknown"));
    }
}
