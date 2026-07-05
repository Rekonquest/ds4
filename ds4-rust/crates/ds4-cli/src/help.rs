// DS4 (DwarfStar) — CLI usage strings.
//
// Replacement for the embedded `ds4_help.c` table. The strings here
// mirror what `ds4 --help` and `ds4 interactive` should print; they
// double as docs for the `--help` flag wired through clap.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const CRATE_NAME: &str = "ds4-cli";

pub const ONE_SHOT_DESCRIPTION: &str = "Run a single prompt against the engine and exit.";
pub const REPL_DESCRIPTION: &str = "Start an interactive REPL with persistent session state.";

pub const HELP_HEAD: &str = "\
ds4 — DwarfStar command-line interface

USAGE:
    ds4 [OPTIONS] [PROMPT ...]
    ds4 repl [OPTIONS]

ARGUMENTS:
    [PROMPT ...]   Run a one-shot completion. If omitted, start the REPL.

OPTIONS:";

pub const HELP_TAIL: &str = "\
ENVIRONMENT:
    DS4_LOG        Log filter (default: info). E.g. `DS4_LOG=debug`.
    DS4_SEED       RNG seed (u64). Overrides --seed.
    DS4_CTX_SIZE   Context size in tokens. Overrides --ctx.
    DS4_THREADS    Thread count for CPU backend. Overrides --threads.

EXAMPLES:
    ds4 --model model.gguf --ctx 8192 'Tell me a story.'
    echo 'Summarise this.' | ds4 --model model.gguf --prompt-stdin
    ds4 --model model.gguf                   # start the REPL
";

pub fn print_help() {
    println!("{HELP_HEAD}");
    println!("    -m, --model <PATH>         Path to a DwarfStar GGUF model file.");
    println!("        --mtp <PATH>           Path to optional MTP head weights.");
    println!("    -c, --ctx <N>              Context size in tokens (default: 8192).");
    println!("        --threads <N>          CPU threads (default: 1).");
    println!("        --prefill-chunk <N>    Tokens per prefill chunk (default: 512).");
    println!("        --temperature <F>      Sampling temperature (default: 1.0).");
    println!("        --top-k <N>            Top-k sampling (default: 40).");
    println!("        --top-p <F>            Top-p sampling (default: 0.9).");
    println!("        --min-p <F>            Min-p sampling (default: 0.0).");
    println!("        --seed <U64>           RNG seed.");
    println!("        --n-predict <N>        Max new tokens (default: 256).");
    println!("        --system <STR>         Optional system prompt.");
    println!("        --no-display           Suppress live token display.");
    println!("        --prompt-stdin         Read the prompt from stdin.");
    println!("    -h, --help                 Print this help.");
    println!("    -V, --version              Print version.");
    println!();
    println!("{REPL_DESCRIPTION}");
    println!("{ONE_SHOT_DESCRIPTION}");
    println!();
    println!("{HELP_TAIL}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_strings_are_non_empty() {
        assert!(!HELP_HEAD.is_empty());
        assert!(!HELP_TAIL.is_empty());
        assert!(!ONE_SHOT_DESCRIPTION.is_empty());
        assert!(!REPL_DESCRIPTION.is_empty());
    }
}
