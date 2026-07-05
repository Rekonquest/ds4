// DS4 (DwarfStar) â€” CLI binary.
//
// Replaces `ds4_cli.c` (1707 LoC) + `ds4_help.c`. Provides:
//   - one-shot mode (single prompt -> response -> exit)
//   - interactive REPL mode with persistent Ds4Session so the CLI
//     reuses the KV cache across turns (unlike the stateless HTTP
//     server)
//   - persistent rustyline history (replaces the embedded linenoise)
//
// GGUF models run through the Rust session path when the selected
// backend can load them. Missing runtime models fail closed and the
// CLI surfaces that state cleanly.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod help;
mod one_shot;
mod repl;

use ds4_types::Ds4Backend;
use one_shot::{engine_missing_model_message, is_engine_pending, OneShotConfig};
use repl::ReplConfig;

const CRATE_NAME: &str = "ds4-cli";
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "ds4", version, about = "DwarfStar CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    mode: Option<Mode>,

    /// Path to a DeepSeek V4 Flash / PRO GGUF produced for DwarfStar.
    #[arg(long, short = 'm', global = true)]
    model: Option<PathBuf>,

    /// Path to the MTP head weights (optional).
    #[arg(long, global = true)]
    mtp: Option<PathBuf>,

    /// Context size in tokens.
    #[arg(long, short = 'c', default_value_t = 8192, global = true)]
    ctx: usize,

    /// Tokens per prefill chunk.
    #[arg(long, default_value_t = 512, global = true)]
    prefill_chunk: usize,

    /// CPU threads.
    #[arg(long, default_value_t = 1, global = true)]
    threads: usize,

    /// Backend to request: cpu, cuda, arc, vulkan, rocm, or metal.
    #[arg(long, default_value = "cpu", value_parser = parse_backend, global = true)]
    backend: Ds4Backend,

    /// Sampling temperature.
    #[arg(long, default_value_t = 1.0, global = true)]
    temperature: f32,

    /// Top-k sampling.
    #[arg(long, default_value_t = 40, global = true)]
    top_k: usize,

    /// Top-p sampling.
    #[arg(long, default_value_t = 0.9, global = true)]
    top_p: f32,

    /// Min-p sampling.
    #[arg(long, default_value_t = 0.0, global = true)]
    min_p: f32,

    /// RNG seed.
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Maximum tokens to predict.
    #[arg(long, default_value_t = 256, global = true)]
    n_predict: usize,

    /// Optional system prompt.
    #[arg(long, global = true)]
    system: Option<String>,

    /// Read the prompt from stdin.
    #[arg(long, global = true)]
    prompt_stdin: bool,

    /// One-shot prompt (positional). If omitted, enter REPL.
    prompt: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Start the interactive REPL (default when no prompt is given).
    Repl,
    /// Run a single prompt and exit.
    OneShot {
        /// Prompt text. If empty and `--prompt-stdin` is set, read stdin.
        prompt: Vec<String>,
    },
}

fn parse_backend(value: &str) -> Result<Ds4Backend, String> {
    match value.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Ds4Backend::Cpu),
        "cuda" => Ok(Ds4Backend::Cuda),
        "arc" | "intel-arc" | "xpu" => Ok(Ds4Backend::Arc),
        "vulkan" | "vulcan" | "vk" => Ok(Ds4Backend::Vulkan),
        "rocm" => Ok(Ds4Backend::Rocm),
        "metal" => Ok(Ds4Backend::Metal),
        other => Err(format!(
            "unknown backend {other:?}; expected cpu, cuda, arc, vulkan, rocm, or metal"
        )),
    }
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    log::info!("{CRATE_NAME} {VERSION}");
    let _help_entry: fn() = help::print_help;
    debug_assert_eq!(help::CRATE_NAME, CRATE_NAME);
    debug_assert_eq!(help::VERSION, VERSION);

    let model_path = match cli.model.clone() {
        Some(p) => p,
        None => {
            eprintln!("error: --model <PATH> is required");
            return ExitCode::from(2);
        }
    };

    let wants_repl = match &cli.mode {
        Some(Mode::Repl) => true,
        Some(Mode::OneShot { prompt }) => {
            prompt.is_empty() && cli.prompt.is_empty() && !cli.prompt_stdin
        }
        None => cli.prompt.is_empty() && !cli.prompt_stdin,
    };

    let result = if wants_repl {
        let cfg = ReplConfig {
            model: model_path.clone(),
            mtp: cli.mtp.clone(),
            ctx: cli.ctx,
            prefill_chunk: cli.prefill_chunk,
            n_threads: cli.threads,
            backend: cli.backend,
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            min_p: cli.min_p,
            seed: cli.seed,
            n_predict: cli.n_predict,
            system: cli.system.clone(),
        };
        repl::run(cfg)
    } else {
        let mut prompt_parts: Vec<String> = match &cli.mode {
            Some(Mode::OneShot { prompt }) => prompt.clone(),
            _ => Vec::new(),
        };
        prompt_parts.extend(cli.prompt.iter().cloned());
        let prompt = prompt_parts.join(" ");
        let cfg = OneShotConfig {
            model: model_path.clone(),
            mtp: cli.mtp.clone(),
            ctx: cli.ctx,
            prefill_chunk: cli.prefill_chunk,
            n_threads: cli.threads,
            backend: cli.backend,
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            min_p: cli.min_p,
            seed: cli.seed,
            n_predict: cli.n_predict,
            system: cli.system.clone(),
            prompt,
            display: true,
        };
        one_shot::run(cfg)
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            for cause in e.chain() {
                if let Some(de) = cause.downcast_ref::<ds4_types::Ds4Error>() {
                    if is_engine_pending(de) {
                        eprintln!("{de}");
                        eprintln!("{}", engine_missing_model_message());
                        return ExitCode::from(3);
                    }
                }
            }
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_constants_are_sane() {
        assert_eq!(CRATE_NAME, "ds4-cli");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn parse_backend_accepts_supported_names() {
        assert_eq!(parse_backend("cpu").unwrap(), Ds4Backend::Cpu);
        assert_eq!(parse_backend("CUDA").unwrap(), Ds4Backend::Cuda);
        assert_eq!(parse_backend("arc").unwrap(), Ds4Backend::Arc);
        assert_eq!(parse_backend("xpu").unwrap(), Ds4Backend::Arc);
        assert_eq!(parse_backend("vulkan").unwrap(), Ds4Backend::Vulkan);
        assert_eq!(parse_backend("vulcan").unwrap(), Ds4Backend::Vulkan);
        assert!(parse_backend("bogus").is_err());
    }
}
