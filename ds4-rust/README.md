# DS4 Rust — `DwarfStar` in Rust

> Development status: this Rust rewrite is still under active development. It is
> pre-1.0, has no API stability promise, and should be treated as a development
> preview until the port has more production mileage.

A Cargo workspace that rewrites the `DwarfStar` DeepSeek V4 Flash / PRO
inference engine from C / Objective-C / CUDA / HIP / Metal into Rust.

## Status

**v0.1.0 — in active development.** Pre-1.0; no API stability promise.

This workspace lives inside this fork next to the original C tree. The C tree
keeps building via its `Makefile`, and the two are kept in sync by a periodic
port of any new fields added to `ds4.h` upstream.

See `../docs/superpowers/specs/2026-07-04-ds4-rust-rewrite-design.md` for the
design spec and `../docs/superpowers/specs/REVERSE-ENGINEERING-DS4.md` for the
operator-authored reverse-engineering teardown that motivated this rewrite.

## Layout

```
ds4-rust/
├── Cargo.toml               workspace root
├── LICENSE                  addenda for ggml + candle + tract + mistralrs + TGI + Rust code
├── crates/                  14 Rust crates
├── third_party/             vendored upstream sources with their original LICENSE copies
├── tests/test-vectors/      golden vectors carried over from the C regression suite
└── tools/                   build / lint / vendor-check helpers
```

## Crates

| Crate                  | Role                                                                |
|------------------------|---------------------------------------------------------------------|
| `ds4-core`             | `Ds4Engine`, `Ds4Session`, sync state machine, KV, chat, sampler, MTP |
| `ds4-quant`            | Q8_0, Q4_K, Q2_K, IQ2_XXS, F16/F32 dot / quant kernels              |
| `ds4-tensor`           | typed wrapper over vendored candle-core                             |
| `ds4-backend-cpu`      | correctness path on top of vendored tract-linalg                    |
| `ds4-backend-cuda`     | CUDA kernel/toolchain surface; runtime model loading reports unavailable device execution |
| `ds4-backend-metal`    | MSL kernel/toolchain surface; runtime model loading reports unavailable device execution |
| `ds4-backend-rocm`     | HIP kernel/toolchain surface; runtime model loading reports unavailable device execution |
| `ds4-backend-paged`    | clean-Rust paged-attention reference crate                           |
| `ds4-kvstore`          | SHA1 prefix-hash + linear-scan disk LRU; `DSV4`/`DSVL` formats      |
| `ds4-ssd`              | SSD streaming glue + hotlist loader                                 |
| `ds4-dist`             | Distributed protocol, coordinator/worker wire frames, API hooks     |
| `ds4-cli`              | binary `ds4` (rustyline replaces linenoise)                         |
| `ds4-server`           | binary `ds4-server` (HTTP + DSML + Anthropic / OpenAI compat)      |
| `ds4-imatrix`          | imatrix collection (engine API)                                     |

## Building

```sh
# Default: CPU backend only.
cargo build --workspace

# GPU backend crates are workspace members and build with the default
# workspace command. Runtime backend selection is controlled by engine
# options, not by root workspace feature flags.
```

## Testing

```sh
cargo test --workspace                 # unit tests + official-vector regression
cargo clippy --workspace --all-targets -- -D warnings
```

## License

Dual-licensed under MIT OR Apache-2.0. See `LICENSE` for the full text
and addenda for each vendored upstream source.
