# DS4 → Rust rewrite — design spec

Date: 2026-07-04
Status: Approved by operator during brainstorming on 2026-07-04.
Source-of-truth inputs:
- `ds4/README.md`
- `ds4/AGENT.md`
- `ds4/CONTRIBUTING.md`
- `ds4/MODEL_CARD.md`
- `ds4/QA_BEFORE_RELEASES.md`
- `ds4/STRIXHALO.md`
- `../../REVERSE-ENGINEERING-DS4.md` (operator-authored teardown; copied
  into this repo as part of v1, see Section 6).

## 0. Goal

Rewrite `DwarfStar` (DeepSeek V4 Flash / PRO inference engine, currently
~113k LoC of C99 + Objective-C Metal + CUDA + ROCm/HIP + Python tooling)
into a Rust Cargo workspace that:

1. Preserves the DS4 *design identity* — `Engine + Session`, the
   `sync → rewrite_from_common → RebuildNeeded` state machine, the
   disk-native KV cache (`DSV4`/`DSVL` on-disk payloads), the
   compile-time-conscious backend split, the DeepSeek-only chat
   template, the IQ2_XXS MoE quantization.
2. Adopts the consolidation recommended in the operator's teardown:
   a `Backend` trait shape, vendored `candle-core` / `tract-linalg` /
   `mistralrs-paged-attn` (plus the TGI gRPC v3 `.proto` skeleton) under
   each upstream's original license, with attribution.
3. Delivers in v1: the `ds4` CLI binary and the `ds4-server` HTTP
   binary, on three GPU backends at parity (Metal / CUDA / ROCm) plus
   CPU correctness.
4. Deferrs to v2: ds4-agent, ds4-bench, ds4-eval, distributed
   inference, gguf-tools rewrites, full TGI gRPC v3 server.

## 1. Constraints carried in from existing DS4

These are non-negotiable invariants from the existing C code base, and
the Rust port must preserve them:

- **C99, no C++.** AGENT.md says "Do not introduce C++." The Rust port
  naturally satisfies this; we do not introduce `cc`/`autotools` to
  compile the original C files into the Rust workspace. (The original
  C tree remains buildable in-place via its `Makefile`; we are a
  sibling project, not a replacement.)
- **One model family.** DeepSeek V4 Flash + PRO with the dedicated
  GGUF layouts documented in `README.md` and `MODEL_CARD.md`. We do
  not add support for arbitrary GGUFs.
- **KV cache is a first-class disk citizen.** `DSV4` and `DSVL`
  payload magic+version+field-count are preserved byte-for-byte so
  Rust and C can interoperate on disk.
- **Beta-quality honesty.** Cargo workspace is `0.x` pre-release.
  Officially-vector regression must match within fp32 tolerance or we
  ship a known-broken label, not a silent best-effort.
- **Attribution.** Vendored source retains upstream LICENSE/NOTICE
  copies verbatim; the top-level LICENSE is a single file with
  addenda.

## 2. Cargo workspace shape

```
ds4-rust/                            # the sibling project root
├── Cargo.toml                       # workspace root
├── LICENSE                          # single file with addenda
├── README.md
├── docs/
│   └── superpowers/specs/
│       └── 2026-07-04-ds4-rust-rewrite-design.md   # this file
├── crates/
│   ├── ds4-core/                    # Engine/Session/sync/KV/chat/sampler/MTP
│   ├── ds4-quant/                   # Q8_0, Q4_K, Q2_K, IQ2_XXS, F16/F32
│   ├── ds4-tensor/                  # thin typed wrapper over candle-core
│   ├── ds4-backend-cpu/             # tract-linalg + ds4-quant
│   ├── ds4-backend-cuda/            # hand-rolled CUDA kernels
│   ├── ds4-backend-metal/           # hand-rolled MSL kernels
│   ├── ds4-backend-rocm/            # hand-rolled HIP kernels
│   ├── ds4-backend-paged/           # vendored mistralrs-paged-attn
│   ├── ds4-kvstore/                 # SHA1 + linear-scan disk LRU
│   ├── ds4-ssd/                     # SSD streaming glue + hotlist loader
│   ├── ds4-dist/                    # STUB: 4 distributed API hooks
│   ├── ds4-cli/                     # binary: ds4
│   ├── ds4-server/                  # binary: ds4-server
│   └── ds4-imatrix/                 # imatrix collection (engine API)
├── third_party/
│   ├── ggml/                        # MIT, attribution preserved
│   ├── mistralrs-paged-attn/        # MIT
│   ├── tract-linalg/                # MIT OR Apache-2.0
│   ├── candle-core/                 # MIT OR Apache-2.0
│   └── tgi-proto/                   # Apache-2.0 .proto skeleton
├── tests/
│   └── test-vectors/                # golden vectors carried over from C
└── tools/
    └── gguf-tools-rust/             # OPTIONAL in v1; only imatrix path
```

### Workspace-level Cargo features

- `cpu` (default): builds `ds4-backend-cpu`.
- `metal`: builds `ds4-backend-metal` (macOS only).
- `cuda`: builds `ds4-backend-cuda` (Linux + CUDA toolkit).
- `rocm`: builds `ds4-backend-rocm` (Linux + ROCm).
- `paged`: builds `ds4-backend-paged`.
- `imatrix`: builds `ds4-imatrix` and exposes the imatrix engine API.

Runtime backend selection via `Ds4EngineOptions::backend`. Compile-time
feature gating determines which backend crates are linked. The four
GPU backends are mutually exclusive on any given machine (you can't
have Metal and CUDA on the same box) but all four crates co-exist in
the workspace.

## 3. Public API surface

Mirrors `ds4.h` 1:1 in Rust naming.

### 3.1 Handles and enums

```rust
pub struct Ds4Engine { /* opaque */ }
pub struct Ds4Session { /* opaque */ }

pub enum Ds4Backend { Metal, Cuda, Rocm, Cpu }
pub enum Ds4ThinkMode { None, High, Max }
pub enum Ds4DistributedRole { None, Coordinator, Worker }
pub enum Ds4QuantKind { Q8_0, Q4_K, Q2_K, Iq2Xxs, F16, F32 }
```

### 3.2 Options

`Ds4EngineOptions` is a direct mirror of `ds4_engine_options` in
`ds4.h`. Every field is preserved with the same name and semantics:

| Rust field                       | C field                              |
|----------------------------------|--------------------------------------|
| `model_path`                     | `model_path`                         |
| `mtp_path`                       | `mtp_path`                           |
| `backend`                        | `backend`                            |
| `n_threads`                      | `n_threads`                          |
| `prefill_chunk`                  | `prefill_chunk`                      |
| `mtp_draft_tokens`               | `mtp_draft_tokens`                   |
| `mtp_margin`                     | `mtp_margin`                         |
| `directional_steering_file`      | `directional_steering_file`          |
| `expert_profile_path`            | `expert_profile_path`                |
| `directional_steering_attn`      | `directional_steering_attn`          |
| `directional_steering_ffn`       | `directional_steering_ffn`           |
| `power_percent`                  | `power_percent`                      |
| `ssd_streaming`                  | `ssd_streaming`                      |
| `ssd_streaming_cache_experts`    | `ssd_streaming_cache_experts`        |
| `ssd_streaming_cache_bytes`      | `ssd_streaming_cache_bytes`          |
| `ssd_streaming_preload_experts`  | `ssd_streaming_preload_experts`      |
| `ssd_streaming_cold`             | `ssd_streaming_cold`                 |
| `simulate_used_memory_bytes`     | `simulate_used_memory_bytes`         |
| `warm_weights`                   | `warm_weights`                       |
| `quality`                        | `quality`                            |
| `inspect_only`                   | `inspect_only`                       |
| `load_slice`                     | `load_slice`                         |
| `distributed`                    | `distributed`                        |

### 3.3 Engine lifecycle

```rust
Ds4Engine::open(opts: Ds4EngineOptions) -> Ds4Result<Self>
engine.close()
engine.summary()
engine.vocab_size() -> usize
engine.power() -> u8; engine.set_power(p: u8)
engine.model_name() -> &str
engine.model_id() -> &str
engine.layer_count() -> usize
engine.layer_compress_ratio(layer: usize) -> u32
engine.hidden_f32_values() -> usize
engine.routed_quant_bits() -> u8
engine.has_output_head() -> bool
engine.has_mtp() -> bool
engine.mtp_draft_tokens() -> usize
```

### 3.4 Chat + tokenization

```rust
engine.tokenize_text(text: &str) -> Ds4Result<Vec<TokenId>>
engine.tokenize_rendered_chat(text: &str) -> Ds4Result<Vec<TokenId>>
engine.chat_begin() -> Vec<TokenId>
engine.encode_chat_prompt(system: &str, prompt: &str, think: Ds4ThinkMode) -> Ds4Result<Vec<TokenId>>
engine.chat_append_max_effort_prefix(tokens: &mut Vec<TokenId>)
engine.chat_append_message(tokens: &mut Vec<TokenId>, role: Ds4Role, content: &str)
engine.chat_append_assistant_prefix(tokens: &mut Vec<TokenId>, think: Ds4ThinkMode)
engine.token_text(token: TokenId) -> Ds4Result<(&str, usize)>
engine.token_eos() -> TokenId
engine.token_user() -> TokenId
engine.token_assistant() -> TokenId
```

### 3.5 Session hot loop

```rust
Ds4Session::create(engine: &Ds4Engine, ctx_size: usize) -> Ds4Result<Self>
session.free()
session.set_progress(fn: FnMut(SessionProgress), ud: *mut c_void)
session.set_display_progress(fn: FnMut(SessionProgress), ud: *mut c_void)
session.set_cancel(fn: FnMut() -> bool, ud: *mut c_void)

session.sync(prompt: &[TokenId]) -> Ds4Result<()>     // reuse | extend | rebuild
session.rewrite_from_common(prompt: &[TokenId], common: usize) -> Ds4RewriteStatus
session.common_prefix(prompt: &[TokenId]) -> usize
session.argmax() -> TokenId
session.argmax_excluding(excluded: TokenId) -> TokenId
session.sample(temperature: f32, top_k: usize, top_p: f32, min_p: f32, rng: &mut Rng) -> TokenId
session.top_logprobs(out: &mut [f32], k: usize)
session.token_logprob(token: TokenId, out: &mut f32)
session.copy_logits(out: &mut [f32], cap: usize)
session.set_logits(logits: &[f32])
session.eval(token: TokenId) -> Ds4Result<()>
session.eval_speculative_argmax(first: TokenId, max: usize, eos: TokenId, accepted: &mut [TokenId]) -> Ds4Result<()>
session.invalidate()
session.rewind(pos: usize)
session.pos() -> usize
session.ctx() -> usize
session.prefill_cap() -> usize
session.tokens() -> &[TokenId]
```

```rust
pub enum Ds4RewriteStatus { Ok, RewriteError, RebuildNeeded }
```

### 3.6 Distributed slice-level API (stubs in v1)

```rust
session.layer_slice_reset() -> Ds4Result<()>      // returns Err(NotImplemented) in v1
session.eval_layer_slice(...) -> Ds4Result<()>    // returns Err(NotImplemented) in v1
session.eval_output_head_from_hc(...) -> Ds4Result<()>  // returns Err(NotImplemented) in v1
session.distributed_route_ready() -> Ds4Result<bool>    // returns Ok(false) in v1
session.is_distributed() -> bool                  // returns false in v1
```

The signatures match the C `ds4_session_layer_slice_reset`,
`ds4_session_eval_layer_slice`,
`ds4_session_eval_output_head_from_hc`,
`ds4_session_distributed_route_ready` so that the v2 implementation
can be a drop-in replacement for the v1 stub.

### 3.7 On-disk payload constants

```rust
pub const DS4_SESSION_PAYLOAD_MAGIC: u32 = 0x34565344;   // "DSV4"
pub const DS4_SESSION_PAYLOAD_VERSION: u32 = 2;
pub const DS4_SESSION_PAYLOAD_U32_FIELDS: u32 = 13;

pub const DS4_SESSION_LAYER_PAYLOAD_MAGIC: u32 = 0x4c565344;   // "DSVL"
pub const DS4_SESSION_LAYER_PAYLOAD_VERSION: u32 = 1;
pub const DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS: u32 = 14;
```

`ds4-kvstore::SessionPayload` and `LayerPayload` use these constants
as the wire format. Rust-written payloads must be byte-identical to
C-written payloads; the official-vector regression suite tests this.

### 3.8 Errors

```rust
pub type Ds4Result<T> = Result<T, Ds4Error>;

pub struct Ds4Error {
    pub kind: Ds4ErrorKind,
    pub message: String,
}

pub enum Ds4ErrorKind {
    InvalidArgument,
    Io,
    Model,
    Tokenizer,
    Backend,
    KvStore,
    OutOfMemory,
    NotImplemented,   // for distributed stubs in v1
    Other,
}
```

Errors are constructed via `thiserror`-derived `From` impls; the
public surface does not leak any third-party error crate.

### 3.9 Backend trait (private, behind `Ds4Engine`)

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn load_model(&mut self, opts: &Ds4EngineOptions) -> Ds4Result<Box<dyn BackendModel>>;
    fn memory_estimate(ctx_size: usize, prefill_chunk: usize) -> u64;
}

pub trait BackendModel: Send {
    fn forward_layer(&mut self, layer: usize, input: &Tensor, kv: &mut dyn KvCache) -> Ds4Result<Tensor>;
    fn forward_mtp(&mut self, ...) -> Ds4Result<Tensor>;
    fn forward_output_head(&mut self, hidden: &Tensor) -> Ds4Result<Tensor>;
    fn quant_kind(&self) -> Ds4QuantKind;
}
```

The `Backend` trait is **not** publicly exported. Users get a
`Ds4Engine`, which holds a `Box<dyn Backend>` internally and routes
calls. This matches the C design (`ds4.h` does not expose a backend
trait — callers pick at engine-open time).

The hand-rolled backends (CUDA / Metal / ROCm) and the lifted
paged-attention backend are all `Backend` implementations.
DS4's compressed-hybrid attention is *one* implementation; paged
attention is *another*. Both can co-exist behind the same Engine.

## 4. Subsystems (port-by-port mapping)

### 4.1 Tokenizer (`ds4.c:21927..22150`)

GPT-2 byte-level BPE + DeepSeek pre-tokenizer regex, hand-rolled in
Rust inside `ds4-core`. Direct port of `byte_encode`, `bpe_emit_piece`,
and the pre-tokenize regex. BPE merge table is loaded from the GGUF at
engine-open time.

### 4.2 Chat template (`ds4.c:22371..22434`)

Hand-rolled, not jinja. Direct port of `ds4_chat_begin`,
`ds4_encode_chat_prompt`, `ds4_chat_append_message`,
`ds4_chat_append_assistant_prefix`,
`ds4_chat_append_max_effort_prefix`. Special tokens hard-coded:
`bos, user, assistant, think_start, think_end, dsml`.

### 4.3 Sampler (`ds4.c:22620..22685`)

`sample_top_p_min_p` — temperature scaling + top-p + top-k + min-p
in one pass. No DRY, no repetition penalty, no Mirostat. Direct port.

### 4.4 KV cache (`ds4.c:8300..8318`)

Three-tier hierarchy preserved:
1. Raw SWA — 128-token sliding-window ring buffer.
2. Compressed — per-layer `compress_ratio` (0 = none, 2 = midpoint,
   4 = indexer-driven).
3. Indexer-compressed — for `compress_ratio == 4` layers, a separate
   `index_comp_kv` with top-k = 512 (Flash) or 1024 (Pro).

Storage:
- `Ds4KvCache` (CPU path, RAM-resident)
- Per-backend tensors in `ds4-backend-{metal,cuda,rocm}`
- `ds4-kvstore` (SHA1 prefix hashing + disk LRU; port of `ds4_kvstore.c`)
- `ds4-ssd` (expert access histograms for SSD prefetch; port of
  `ds4_ssd.c`)

### 4.5 Quantization (`ds4-quant`)

Native support, with formats ported directly from `ds4_cuda.cu` and
`ds4_quant.c`/`ds4_quant.h`:

- **Q8_0** — 32-dim blocks, `{f32 d, i8 qs[32]}`. `quantize_q8_0_f32_kernel`
  at `ds4_cuda.cu:3627`.
- **Q4_K** — 256-dim super-blocks, grouped scales, 128 nibbles.
- **Q2_K** — 2-bit super-blocks, low-memory tier.
- **IQ2_XXS** — 2-bit inter-channel for MoE gate + up experts. Per-block
  `uint16_t d`, `uint16_t qs[32]`, 32 bytes total. Grid + sign lookup
  tables from `ds4_iq2_tables_cuda.inc` are vendored as Rust constants
  in `ds4-quant/src/iq2_xxs/luts.rs`.
- **F16 / F32** — attention weights + critical paths.

### 4.6 GGUF loader

Custom, not from llama.cpp. Mmap-based, header-only parse, tensor
descriptors (name, dims, type, offsets). Direct port of the value-type
enum at `ds4.c:1536..1547` and the post-parse metadata read
(`ds4.c:1616..3074`).

### 4.7 MTP (`ds4.c:27167`)

`eval_speculative_argmax` direct port:
1. Evaluate `first_token` via the target model.
2. If MTP is ready, MTP's own transformer layer proposes up to 16
   suffix tokens against its own `raw_cache` frontier.
3. Target graph verifies the suffix layer-by-layer.
4. Commit accepted prefix; roll back speculative state on miss.
5. Fall back to single-token decode if target stream is broken.

MTP weights struct (`ds4_mtp_weights` at `ds4.c:3064`) is ported.

### 4.8 Distributed stubs (`ds4-dist`)

```rust
pub fn layer_slice_reset(s: &mut Ds4Session) -> Ds4Result<()> {
    Err(Ds4Error { kind: Ds4ErrorKind::NotImplemented,
                   message: "distributed inference is v2".into() })
}
// ... same for the other three hooks ...
pub fn distributed_route_ready(s: &Ds4Session) -> Ds4Result<bool> { Ok(false) }
pub fn is_distributed(s: &Ds4Session) -> bool { false }
```

The crate exposes the function symbols so external code (and our own
tests) can compile against the v2 API without churn.

### 4.9 CLI (`ds4-cli`)

Replaces `ds4_cli.c` (1707 LoC) and `ds4_help.c`. Uses `rustyline`
for interactive REPL (replaces `linenoise`). Persistent `Ds4Session`
across turns so the CLI's KV cache is reused.

### 4.10 Server (`ds4-server`)

Replaces `ds4_server.c` (15875 LoC). Hand-rolled sockets OR `hyper` —
TBD during implementation; recommendation: start with `hyper` because
the SSE streaming path is well-supported and saves us writing our own
HTTP/1.1 parser. DSML parser, OpenAI/Anthropic/Responses API
compatibility, UTF-8-safe streaming, DSML-entity-safe streaming — all
direct ports of `ds4_server.c:1070..1170`, `:5859..5935`, `:5604..5635`.

## 5. Build, test, verify

- **Build**: `cargo build --workspace` produces all v1 binaries.
  Feature flags `cpu` (default), `metal`, `cuda`, `rocm`, `paged`,
  `imatrix` toggle which backends/parts are linked.
- **Lint**: `cargo clippy --workspace --all-targets -- -D warnings`
  must pass before any PR.
- **Test**: `cargo test --workspace` runs:
  - Unit tests inside each crate.
  - Official-vector regression (golden vectors in `tests/test-vectors/`
    carried over from `ds4/tests/test-vectors/`; Rust harness checks
    Rust output logits match within fp32 tolerance against the
    embedded reference).
  - Smoke test: load a small model fixture and produce a fixed prompt.
- **C-vs-Rust compare** (opt-in via `cargo test --features compare-c`):
  Rust runs the same prompt on the same model and diffs the logits
  against a C reference run captured to disk. Default off.
- **CI**: gates on `cargo check`, `cargo clippy -D warnings`, `cargo test`.
- **The original C tree is not touched.** It keeps building via its
  own `Makefile` so we can A/B test at any point.

## 6. Out of scope for v1 (deferred)

These are documented as TODOs but explicitly *not* part of v1:

- ds4-agent (the seven tool primitives + CDP/Chrome web tool).
- ds4-bench (perf harness).
- ds4-eval (eval harness + fixed question bank).
- Distributed inference: full coordinator/worker + custom TCP +
  activation-bit compression + KV snapshot gather/scatter.
- gguf-tools subdir: `deepseek4-quantize.c`, `quants.c/.h`,
  `splice_mixed_expert_layers_gguf.py` — stay in C/Python until v2.
- TGI gRPC v3 backend exposure: only the vendored `.proto` skeleton;
  no actual server implementation.
- `dir-steering/`, `speed-bench/`: data and Python scripts untouched.

The `ds4-dist` crate ships the stubbed API so v2 can be a drop-in
replacement without touching call sites.

## 7. Operator-approved artefacts to copy into the repo

The user explicitly approved (during Phase 1) that the following
external file should be copied into the assigned project after the
plan is approved:

- Source: `F:\Rust DS$ engine rewrite into rust\REVERSE-ENGINEERING-DS4.md`
- Destination: `ds4/docs/superpowers/specs/REVERSE-ENGINEERING-DS4.md`
  (operator's teardown; treated as a design input, not as a Rust doc).
- Rationale: project-boundary rule says don't read outside the project
  without permission; once copied inside, future sessions can read it
  without a boundary exception.

## 8. Risks and residual items

- **Vendoring licensing overhead**: Each vendored upstream has its own
  license. The LICENSE addenda must list every retained notice; missed
  attribution is a license violation. Mitigated by a `vendor-check`
  CI step that asserts every `third_party/<proj>/LICENSE` is present
  and unmodified.
- **CUDA toolkit / ROCm install**: CI machines and developer
  workstations need CUDA and ROCm toolkits for those backends.
  Mitigated by making `cpu` the default and the others opt-in
  features.
- **DS4 parity drift**: The C tree keeps evolving. Periodic
  re-port of any new fields in `ds4.h` is required. Mitigated by a
  monthly parity review (process, not a code gate).
- **fp32 official-vector tolerance**: The C reference implementation
  uses the same float math. We expect exact parity up to associative
  reordering. Tolerance is `1e-5` per element. Out-of-tolerance
  vectors block the release.
- **Beta-quality honesty**: The README's beta status carries forward.
  We do not claim v1.0.0 quality. Cargo version starts at `0.1.0`.

## 9. Spec self-review

- Placeholders: none. Every section is concrete.
- Internal consistency: Section 3 mirrors Section 4 (subsystem ↔ API).
  v1 stubs in 3.6 match the deferred list in Section 6.
- Scope: focused on one workspace + one design doc + one operator
  approval. Implementable as a single multi-week effort.
- Ambiguity: the only remaining TBD is the HTTP server choice
  (`hyper` vs hand-rolled) in 4.10. This is a 1-line decision at
  implementation time and does not block spec approval.