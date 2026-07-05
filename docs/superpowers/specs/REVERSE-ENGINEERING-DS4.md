# DS4 (DwarfStar) -- self-teardown

Target: `C:/Projects/DS4-EXPERIMRNT/ds4`. This is the actual C-based
LLM inference engine ("DwarfStar") built around DeepSeek V4 Flash /
PRO. All four prior teardowns (mistral.rs, TGI, Candle, tract) have
been mapping back to this codebase; here we document DS4 itself,
mapping *forward* to the four Rust engines.

Convention: ASCII only, terse. Written for a reader who has read the
prior four teardowns in this folder.

---

## 0. Executive shape

From `README.md`: **DwarfStar** is a small native inference engine
optimized first for DeepSeek V4 Flash, with support for V4 PRO on
high-memory machines. Deliberately narrow: *not* a generic GGUF
runner, *not* a wrapper around another runtime. Completely
self-contained. Acknowledges the llama.cpp / GGML lineage: quant
layouts, tables, and CPU quant/dot logic are retained under MIT with
the GGML authors' copyright preserved in `LICENSE`.

Design bets:

- One model at a time; official-vector validation against reference
  logits.
- KV cache as a first-class disk citizen (SSD streaming), not just
  RAM.
- Local inference on high-end personal machines starting at ~96/128 GB
  RAM.
- Optimized graph paths on **Metal** (primary) + **CUDA / DGX Spark**
  + **Strix Halo ROCm**. CPU only for correctness checks and diagnostics.

Source tree (total ~113k lines):

    ds4.c                27,791   core engine (biggest single unit)
    ds4.h                   335   public header
    ds4_gpu.h             1,024   device-abstraction header
    ds4_cuda.cu          13,256   CUDA kernels
    ds4_rocm.cu             131   ROCm entrypoint (hipify wrapper)
    ds4_rocm.h              ...   ROCm header
    ds4_metal.m          26,819   Metal backend (Objective-C)
    ds4_distributed.c     8,414   multi-node / multi-GPU
    ds4_distributed.h       ...
    ds4_server.c         15,875   HTTP server
    ds4_web.c             1,385   web UI stub
    ds4_web.h               ...
    ds4_cli.c             1,707   REPL / interactive
    ds4_agent.c          10,244   integrated coding agent (alpha)
    ds4_kvstore.c         1,359   KV cache persistence
    ds4_kvstore.h           ...
    ds4_ssd.c               181   SSD streaming glue
    ds4_ssd.h               ...
    ds4_bench.c             683   perf benchmark harness
    ds4_eval.c            4,289   evaluation harness
    ds4_help.c/.h                 usage strings
    ds4_iq2_tables_cuda.inc       IQ2 quant tables (generated)
    ds4_streaming_hotlist.inc     streaming hot-tensor list
    linenoise.c/.h                CLI readline (embedded)
    rax.c/.h/rax_malloc.h         radix tree (embedded, from Redis)
    gguf-tools/                   offline GGUF gen + imatrix + quality
    metal/, rocm/                 backend-specific assets
    tests/                        official-vector regression
    dir-steering/                 directional steering data
    speed-bench/                  benchmark commands + charts
    misc/                         misc

Companion doc set: `README.md`, `MODEL_CARD.md`, `AGENT.md`,
`CONTRIBUTING.md`, `QA_BEFORE_RELEASES.md`, `STRIXHALO.md`. Build
system: `Makefile` with `cpu`, Metal, CUDA, ROCm targets.

Embedded third-party (source-drop):

- `linenoise` -- lightweight readline, 2-file drop.
- `rax` -- Redis's radix tree implementation.

---

## 1. Build system (Makefile)

Five binaries per build variant:

    ds4                CLI / REPL (ds4_cli.o + ds4_help.o + linenoise.o + CORE)
    ds4-server         HTTP server (ds4_server.o + ds4_kvstore.o + rax.o + CORE)
    ds4-bench          perf harness (ds4_bench.o + CORE)
    ds4-eval           eval harness (ds4_eval.o + CORE)
    ds4-agent          integrated agent (ds4_agent.o + ds4_web.o + ds4_kvstore.o + linenoise.o + CORE)

`CORE_OBJS` varies by target:

    Metal (Darwin)     ds4.o ds4_distributed.o ds4_ssd.o ds4_metal.o
    CUDA (Linux)       ds4.o ds4_distributed.o ds4_ssd.o ds4_cuda.o
    ROCm (Strix Halo)  ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
    CPU                ds4_cpu.o ds4_distributed.o ds4_ssd.o

Make targets:

    make               (Darwin) Metal build of all 5 binaries
    make cpu           CPU-only build (Metal disabled because macOS VM bug)
    make cuda-spark    CUDA for DGX Spark / GB10 (no explicit arch)
    make cuda-generic  CUDA for local GPU (CUDA_ARCH=native)
    make cuda CUDA_ARCH=sm_N       explicit compute cap
    make strix-halo    ROCm for Strix Halo (offload-arch=gfx1151)
    make rocm          alias for strix-halo
    make test          run tests
    make clean

CUDA flags: `nvcc -O3 --use_fast_math -lineinfo` + `-Xcompiler <NATIVE_CPU_FLAG>`.
Links `-lcudart -lcublas`. Search path pins CUDA to `/usr/local/cuda` with a
`sbsa-linux` extra lib path (the ARM64 CUDA layout DGX Spark uses).

ROCm flags: `hipcc -O3 -ffast-math -D__HIP_PLATFORM_AMD__
--offload-arch=gfx1151`. Links `-lhipblas -lhipblaslt`. `-D DS4_ROCM_BUILD`
selects the ROCm code paths inside `ds4.c`.

Metal flags: `-fobjc-arc` (ARC on for `ds4_metal.m`). Links `-framework
Foundation -framework Metal`.

C is `-O3 -ffast-math -Wall -Wextra -std=c99`. This is a **strict C99 pure-C
codebase**, no C++.

Regression test target: `make cuda-regression` builds
`tests/cuda_long_context_smoke` and runs it -- the reference proof for the
CUDA path.

---

## 2. Public API surface (`ds4.h`)

The header is deliberately narrow -- only what the CLI and server should
know. Tensor internals are hidden behind opaque handles.

### 2.1 Handles

Two opaque types define the entire lifecycle:

    typedef struct ds4_engine  ds4_engine;    // loaded model (immutable weights)
    typedef struct ds4_session ds4_session;   // one mutable inference timeline

Engine is the loaded model (weights, tokenizer, chat template, GPU
handles). Session is one live KV cache + one logits buffer. This is
the same shape as mistral.rs (`MistralRs` vs `Sequence`) and TGI
(engine vs Request); the C variant here uses a single-session-per-caller
model, not a multiplexer.

### 2.2 Enums

    ds4_backend     { METAL, CUDA, CPU }             // ROCm rides CUDA symbols via hip
    ds4_think_mode  { NONE, HIGH, MAX }              // DeepSeek reasoning modes
    ds4_log_type    { DEFAULT, PREFILL, GENERATION,
                      KVCACHE, TOOL, WARNING, TIMING,
                      OK, ERROR }                    // structured logging channels
    ds4_distributed_role { NONE, COORDINATOR, WORKER }

### 2.3 Engine options (`ds4_engine_options`)

The whole config passed to `ds4_engine_open`:

    model_path                     path to GGUF
    mtp_path                       path to MTP head weights (optional)
    backend                        METAL | CUDA | CPU
    n_threads                      CPU threading knob
    prefill_chunk                  tokens per prefill batch
    mtp_draft_tokens               how many draft tokens MTP proposes
    mtp_margin                     acceptance-threshold delta

    directional_steering_file      steering vectors
    expert_profile_path            per-expert routing profile
    directional_steering_attn      steering strength on attention
    directional_steering_ffn       steering strength on FFN
    power_percent                  power / thermal budget

    ssd_streaming                  enable SSD streaming of experts
    ssd_streaming_cache_experts    LRU cache size (in experts)
    ssd_streaming_cache_bytes      LRU cache size (in bytes)
    ssd_streaming_preload_experts  hot experts to pin
    ssd_streaming_cold             cold-start mode
    simulate_used_memory_bytes     debug knob to fake memory pressure

    warm_weights                   pre-touch weights to page them in
    quality                        quality / correctness mode
    inspect_only                   load + dump without inference
    load_slice, load_layer_start,
    load_layer_end, load_output    partial-load for distributed

    distributed                    ds4_distributed_options (below)

Distributed options:

    role           NONE | COORDINATOR | WORKER
    layers         { start, end, has_output, set }  // layer range this rank owns
    listen_host/port                                // worker listens
    coordinator_host/port                           // coordinator address
    prefill_chunk                                   // per-rank prefill batching
    prefill_window                                  // window across ranks
    activation_bits                                 // bit-width for activations on the wire
    replay_check                                    // deterministic replay validation
    debug

### 2.4 Public functions -- engine lifecycle

    ds4_engine_open(&out, opts)                     // load GGUF, init backend, tokenizer
    ds4_engine_close(e)
    ds4_engine_summary(e)                           // print model card
    ds4_engine_vocab_size(e)
    ds4_engine_power(e), ds4_engine_set_power(e, %)
    ds4_engine_model_name(e), ds4_engine_model_id(e)
    ds4_engine_layer_count(e)
    ds4_engine_layer_compress_ratio(e, layer)
    ds4_engine_hidden_f32_values(e)
    ds4_engine_routed_quant_bits(e)                 // e.g. 2 for IQ2
    ds4_engine_has_output_head(e)
    ds4_engine_has_mtp(e)
    ds4_engine_mtp_draft_tokens(e)

### 2.5 Chat + tokenization API

    ds4_tokenize_text(e, text, &out)
    ds4_tokenize_rendered_chat(e, text, &out)
    ds4_chat_begin(e, &tokens)
    ds4_encode_chat_prompt(e, system, prompt, think_mode, &out)
    ds4_chat_append_max_effort_prefix(e, &tokens)
    ds4_chat_append_message(e, &tokens, role, content)
    ds4_chat_append_assistant_prefix(e, &tokens, think_mode)
    ds4_token_text(e, token, &len)
    ds4_token_eos(e), ds4_token_user(e), ds4_token_assistant(e)

**Chat template rendering is inline in the C code** -- not a jinja
runtime. There is a DeepSeek-specific prompt encoder built in and the
"max effort" prefix + assistant-prefix helpers make the think-mode
control explicit at the API level.

### 2.6 Session API (the hot loop)

    ds4_session_create(&out, engine, ctx_size)
    ds4_session_free(s)
    ds4_session_set_progress(s, fn, ud)
    ds4_session_set_display_progress(s, fn, ud)
    ds4_session_set_cancel(s, fn, ud)               // cooperative cancel

    // The core: bring the live state up to a given full token prefix.
    ds4_session_sync(s, prompt, err, errlen)         // reuse | extend | rebuild

    // Sub-step of sync -- rewrite from a common prefix if possible,
    // otherwise report REBUILD_NEEDED.
    ds4_session_rewrite_from_common(s, prompt, common, err, errlen)
        -> DS4_SESSION_REWRITE_ERROR
         | DS4_SESSION_REWRITE_OK
         | DS4_SESSION_REWRITE_REBUILD_NEEDED

    ds4_session_common_prefix(s, prompt)             // longest shared prefix
    ds4_session_argmax(s)                            // greedy pick
    ds4_session_argmax_excluding(s, excluded_id)
    ds4_session_sample(s, temperature, top_k, top_p, min_p, &rng)
    ds4_session_top_logprobs(s, out, k)              // top-K logprobs
    ds4_session_token_logprob(s, token, out)
    ds4_session_copy_logits(s, out, cap)             // introspection
    ds4_session_set_logits(s, logits, n)             // test hook
    ds4_session_eval(s, token, err, errlen)          // append one token, run forward
    ds4_session_eval_speculative_argmax(s, first, max, eos, accepted[], cap, err, errlen)

    ds4_session_invalidate(s), ds4_session_rewind(s, pos)
    ds4_session_pos(s), ds4_session_ctx(s), ds4_session_prefill_cap(s)
    ds4_session_tokens(s)

The `sync -> rewrite_from_common -> REBUILD_NEEDED` state machine is
the whole prefix-cache design in three lines. Callers push a full
prompt; the session figures out whether it can reuse, extend, or must
rebuild.

### 2.7 Distributed slice-level API

Low-level entry points that the coordinator uses to route slices
across worker ranks:

    ds4_session_layer_slice_reset(s, err, errlen)
    ds4_session_eval_layer_slice(s, tokens, n, pos0,
                                 layer_start, layer_end,
                                 input_hc, output_hc,
                                 output_logits, logits,
                                 err, errlen)
    ds4_session_eval_output_head_from_hc(s, hidden_hc, n_tokens,
                                         logits, err, errlen)
    ds4_session_distributed_route_ready(s, err, errlen)
    ds4_session_is_distributed(s)

`hc` = hidden compressed. The wire carries compressed hidden states
between ranks -- not raw activations. This is the *serious* bit that
makes DS4 distributed inference cheap: activation_bits is configurable
(section 2.3) and hidden states get compressed before crossing the
network.

### 2.8 On-disk KV payload

    #define DS4_SESSION_PAYLOAD_MAGIC    UINT32_C(0x34565344)   // "DSV4"
    #define DS4_SESSION_PAYLOAD_VERSION  UINT32_C(2)
    #define DS4_SESSION_PAYLOAD_U32_FIELDS 13u

    #define DS4_SESSION_LAYER_PAYLOAD_MAGIC   UINT32_C(0x4c565344)  // "DSVL"
    #define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
    #define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

Two on-disk formats: session-level ("DSV4") and per-layer ("DSVL",
for distributed). Whole-session and layer-range save/load functions:

    ds4_session_payload_bytes(s)
    ds4_session_stage_payload(s, out, err, errlen)
    ds4_session_write_staged_payload(payload, fp, err, errlen)
    ds4_session_save_payload(s, fp, err, errlen)
    ds4_session_load_payload(s, fp, payload_bytes, err, errlen)
    ds4_session_save_snapshot / load_snapshot / snapshot_free
    ds4_session_layer_payload_bytes(s, layer_start, layer_end)
    ds4_session_save_layer_payload(s, fp, ..., err, errlen)
    ds4_session_load_layer_payload(s, fp, ..., err, errlen)

The stage/write split lets the caller compute the payload once and
write to multiple destinations without recomputing.

### 2.9 Correctness / diagnostic hooks

    ds4_engine_head_test(e, prompt)                 // just the output head
    ds4_engine_first_token_test(e, prompt)          // one forward pass
    ds4_engine_metal_graph_test(e, prompt)          // Metal graph sanity
    ds4_engine_metal_graph_full_test(e, prompt)
    ds4_engine_metal_graph_prompt_test(e, prompt, ctx)
    ds4_engine_generate_argmax(e, prompt, n_predict, ctx, emit_fn, done_fn, ...)
    ds4_engine_collect_imatrix(e, dataset, output, ctx, max_prompts, max_tokens)
    ds4_engine_dump_tokens(e, tokens)
    ds4_dump_text_tokenization(model_path, text, fp)
    ds4_context_memory_estimate(backend, ctx_size)
    ds4_context_memory_estimate_with_prefill(backend, ctx_size, prefill_chunk)
    ds4_log(fp, type, fmt, ...)

Notable: **imatrix collection is a first-class engine API** (not a
separate tool). The engine will run a dataset through itself and emit
an imatrix file, which the gguf-tools quantizer then consumes.

Also: dedicated *Metal graph* test entry points -- Metal is treated as
a first-class citizen with its own probe surface, reflecting the
README's stance that Metal is the primary target.

---

## 3. Core engine (`ds4.c`)

27,791 lines of hand-written C99. Below are the ~30 load-bearing structs
and functions with file:line refs.

### 3.1 Model shape constants (`ds4.c:140..175`)

`ds4_shape` -- static compile-time constants for the two supported
DeepSeek V4 model variants:

    Flash:  43 layers, 4096 embd, 64 heads, 256 experts (6 used + 1 shared)
    Pro:    61 layers, 7168 embd, 128 heads, 384 experts (6 used + 1 shared)

Both share: 1M vocab (129280 tokens), MLA with 64-dim latent heads,
128-token sliding window, 4-head hash compression (hc=4), indexer with
top-k = 512/1024 for sparse retrieval.

### 3.2 Model + weights (`ds4.c:1616..3074`)

- `ds4_model` (`ds4.c:1616`) -- GGUF metadata mirror: `fd`, mmap
  `uint8_t *map`, GGUF header (version, n_kv, n_tensors, alignment),
  `ds4_kv *kv` + `ds4_tensor *tensors` (descriptors only, not data).
- `ds4_layer_weights` (`ds4.c:3016`) -- per-layer tensor pointers.
  Attention: `hc_attn_fn/scale/base`, `attn_norm`, `attn_q_a/q_b`,
  `attn_kv`, `attn_output_a/b`, `compressor_{ape,kv,gate,norm}`,
  `indexer_*`. FFN: `ffn_norm`, `ffn_gate_tid2eid` (token-to-expert
  map), `ffn_gate_inp`, `ffn_gate/up/down_exps` (routed), `ffn_*_shexp`
  (shared experts).
- `ds4_weights` (`ds4.c:3054`) -- root container: `token_embd`,
  `output_hc_base/fn/scale`, `output_norm`, `output`, and
  `layer[61]` sized for the Pro variant.
- `ds4_mtp_weights` (`ds4.c:3064`) -- multi-token-prediction head: its
  own mini-transformer layer plus projections and norms.

### 3.3 Engine + session (`ds4.c:21808..23283`)

- `ds4_engine` (`ds4.c:21808`) -- top-level context: `ds4_model` +
  `mtp_model`, weights, `ds4_vocab`, backend enum, `prefill_chunk`,
  MTP draft/margin, SSD streaming config, distributed options,
  directional-steering settings.
- `ds4_vocab` (`ds4.c:21794`) -- tokenizer state: vocabulary strings,
  n_vocab, special tokens (bos, eos, user, assistant, think_start,
  think_end, dsml), open-addressed `str_i32_table token_to_id`, BPE
  `merge_rank` table.
- `ds4_session` (`ds4.c:23259`) -- per-inference state: engine pointer,
  distributed session pointer, `ds4_gpu_graph graph`, `ds4_kv_cache
  cpu_cache`, `token_vec checkpoint`, logits + mtp_logits float
  buffers, progress/cancel callbacks, ctx_size, prefill_cap,
  `checkpoint_valid` flag.

### 3.4 KV cache (`ds4.c:8300..8318`)

Three-tier hierarchy:

1. **Raw SWA** -- 128-token sliding-window ring buffer, last-N rows
   only.
2. **Compressed** -- per-layer `compress_ratio` (0 = none, 2 = midpoint,
   4 = indexer-driven). Row count grows up to live checkpoint.
3. **Indexer-compressed** -- for `compress_ratio == 4` layers, a
   separate `index_comp_kv` with top-k = 512 (Flash) or 1024 (Pro)
   rows for sparse attention.

Storage:

    cpu_cache: ds4_kv_cache                CPU path, RAM-resident
    graph.layer_*_comp / index_comp        Metal / CUDA tensors, GPU
    ds4_kvstore                            SHA1 prefix hashing + disk LRU
    ds4_streaming_hotlist.inc              expert access histograms for SSD prefetch

### 3.5 Quantization formats

Natively supported:

- **Q8_0** -- 32-dim blocks, `{f32 d, i8 qs[32]}`, quantized inline via
  `quantize_q8_0_f32_kernel` (`ds4_cuda.cu:3627`).
- **Q4_K** -- 256-dim super-blocks with grouped scales (12 bytes) +
  128 nibbles; used for down-experts and attention output.
- **Q2_K** -- 2-bit super-blocks, low-memory tier.
- **IQ2_XXS** -- 2-bit *inter-channel* format for MoE gate + up
  experts. Per-block `uint16_t d`, `uint16_t qs[32]`, 32 bytes total.
  Grid + sign lookup tables baked in as CUDA constants
  (`ds4_iq2_tables_cuda.inc`).
- **F16 / F32** -- attention weights + critical paths.

Asymmetric imatrix quantization is first-class: routed MoE experts
compress hard (IQ2_XXS), attention stays higher precision. This is the
"very asymmetrical quantization" the README boasts about.

### 3.6 GGUF loader

Custom, not from llama.cpp. Mmap-based, parses the GGUF header only;
tensors are accessed via descriptors (name, dims, type, offsets). The
value-type enum runs at `ds4.c:1536..1547`. Post-parse metadata read:
model shape, attention compression ratios, indexer top-k, expert
counts. What was "adapted from llama.cpp under MIT" is the *quant
block formats* (Q2_K, Q4_K on-disk layouts) -- not the reader itself.

### 3.7 Tokenizer

GPT-2 byte-level BPE with a DeepSeek pre-tokenizer:

- Pre-tokenize regex (`ds4.c:22133..22150`): digit groups (<=3),
  CJK/Hiragana/Katakana, letter runs, punctuation, whitespace.
- `byte_encode` (`ds4.c:21927`): raw bytes -> Unicode codepoints
  (printable + mapped unprintable).
- `bpe_emit_piece` (`ds4.c:21978`): applies merges by rank lookup.

### 3.8 Chat template (built-in, no jinja)

DeepSeek-specific rendering functions live in C:

- `ds4_chat_begin` (`ds4.c:22371`)
- `ds4_encode_chat_prompt` (`ds4.c:22375`)
- `ds4_chat_append_message` (`ds4.c:22410`)
- `ds4_chat_append_assistant_prefix` (`ds4.c:22434`)
- `ds4_chat_append_max_effort_prefix` (`ds4.c:22384`)

Special tokens hard-coded: `bos, user, assistant, think_start,
think_end, dsml`.

### 3.9 Sampler + speculative decoding

`sample_top_p_min_p` (`ds4.c:22620..22685`) does temperature scaling +
top-p + top-k + min-p in one pass. `ds4_session_sample` (`ds4.c:26990`)
is the public entry point. No DRY, no repetition penalty, no
Mirostat -- deliberately minimal.

**Speculative decoding via MTP** (`ds4_session_eval_speculative_argmax`
at `ds4.c:27167`):

1. Evaluate `first_token` via the target model (free verification of
   position 0).
2. If MTP is ready, MTP's own transformer layer proposes up to 16
   suffix tokens against its own `raw_cache` frontier.
3. Target graph verifies the suffix layer-by-layer.
4. Commit the accepted prefix; roll back speculative Metal state on
   miss.
5. If target stream is broken, fall back to single-token decode.

### 3.10 Layer forward

- `layer_forward_raw_swa_one` (`ds4.c:9679`) -- raw SWA cache update.
- `layer_forward_self_one` (`ds4.c:10077`) -- self-attention + FFN CPU
  fallback used for correctness testing.

The GPU path lives in the backend-specific files; C code drives the
sequence: prefill loop -> per-token decode loop -> optional MTP.

### 3.11 Distributed hooks

`ds4_dist_session *distributed` inside `ds4_session`. Public entry
points:

    ds4_session_layer_slice_reset       (ds4.c:26197)
    ds4_session_eval_layer_slice        (ds4.c:26314)
    ds4_session_eval_output_head_from_hc(ds4.c:26221)
    ds4_session_distributed_route_ready (ds4.c:26140)

### 3.12 What linenoise and rax are used for

- **linenoise** -- CLI readline in `ds4_cli.c` only. Not used inside
  the engine.
- **rax** (Redis's radix tree) -- **used by the agent**, not the
  engine. `ds4_agent.c` uses two rax instances: one keyed by execution
  id and one keyed by DSML block text, both for memoizing tool-call
  results across turns. The prefix cache in `ds4_kvstore.c` uses a
  SHA1-hash + linear scan design, not rax.

---

## 4. GPU backends and distributed inference

### 4.1 `ds4_gpu.h` -- device abstraction

Pure-C function-pointer dispatch with **compile-time backend
selection**. No virtual classes, no runtime dispatch table. Header
exposes ~50 functions:

- Tensor lifecycle: alloc, alloc_managed (unified memory), view, free,
  fill_f32, write, read, copy, copy_f32-to-f16.
- Command batching: `ds4_gpu_begin_commands` / `ds4_gpu_end_commands`.
- Model loading: `ds4_gpu_set_model_map*` with per-range selectivity
  (critical for SSD streaming).
- Streaming: `ds4_gpu_set_ssd_streaming`, expert cache budget
  (`_configured_count`, `_current_count`).
- Op families (quantized matmul, attention, RoPE, RMSNorm, MoE router,
  hyper-connection HC split/merge, KV compressor, argmax / top-k
  sampling).

### 4.2 CUDA backend (`ds4_cuda.cu`, 13k lines)

100+ `__global__` kernels. Highlights:

- **Quantized GEMM**: `matmul_q8_0_kernel`, `matmul_q8_0_preq_*`
  (pre-quantize activation), `grouped_q8_0_a_preq_warp8_kernel`. IQ2_XXS
  expert decoder `dev_dot_iq2_xxs_q8_K_block` (`ds4_cuda.cu:9723`).
  Fallback dense FP16 via cuBLAS.
- **MLA attention** in 6 variants:
  - `attention_prefill_raw_kernel` (`ds4_cuda.cu:4327`)
  - `attention_decode_mixed_kernel` (`:4630`)
  - `attention_indexed_mixed_kernel` (`:4798`)
  - `attention_static_mixed_heads8_online_kernel` (`:5299`)
  - Plus masked and pooled variants.
  Attention fuses raw SWA (recent tokens) + compressed (older,
  pooled) KV in one kernel.
- **RoPE + head-wise RMSNorm + FP8 KV** are fused into a single tail
  kernel (`head_rms_norm_rope_tail_kernel:4032` + `fp8_kv_quantize_kernel:4251`).
- **MoE routing**: `router_select_kernel:5963`,
  `_parallel_kernel:6015`, `_warp_topk_kernel:6077`.
- **KV compressor**: `compressor_store_kernel:5784`, `_prefill_pool_kernel:5837`,
  `_update_pool_kernel:5904`.
- **Sampling**: single-pass argmax; CUB radix sort for larger top-k.

**No third-party FlashAttention**. DS4 rolls its own online-softmax +
sliding-window + compression hybrid; cuBLAS is only used for dense
fallback.

### 4.3 DGX Spark accommodations (`ds4_cuda.cu:592..624, 1116..1121`)

DGX Spark = NVIDIA 96/128 GB UMA. Special code paths:

- Q8-to-F16 cache auto-capped at 4-16 GiB on 96/128 GB systems with
  large models, so the cuBLAS long-prefill scratch has room.
- 96 GiB default working-set cache, larger models spill to distributed
  layer loading.
- `cudaMallocManaged` + `cudaMemAdvise` read-mostly for Spark's weak
  UMA coherency.
- Search path in the Makefile pins to `sbsa-linux` (ARM64 CUDA).

### 4.4 ROCm backend (`ds4_rocm.cu` + `rocm/*.cuh`)

Thin `.cu` (131 lines) that includes 24 `.cuh` headers implementing
the whole GPU API for HIP:

    ds4_rocm_runtime.cuh          hip* memory / stream / event
    ds4_rocm_q8.cuh (67k)         quant matmul, IQ2_XXS with amd_mixed_dot
    ds4_rocm_moe.cuh (186k)       expert routing + matmul (largest file)
    ds4_rocm_attention.cuh (59k) + _launch.cuh (77k)
    ds4_rocm_matmul.cuh (39k)     dense GEMM via hipBLAS
    ds4_rocm_common.cuh, router.cuh, compressor.cuh, hc.cuh, indexer.cuh

Uses AMD ROCm precise-math libraries (`__ocml_exp_f32`, `__ocml_log1p_f32`)
for the router. Native `amd_mixed_dot` instead of CUDA's `__dp4a`.
Strix Halo (gfx1151) is the target.

### 4.5 Metal backend (`ds4_metal.m`, 26.8k lines) + `metal/*.metal`

Three-tier:

1. **Objective-C++ host** (`ds4_metal.m`): `MTLDevice` + command queue,
   pipeline cache keyed by shape-dependent function constants,
   `DS4MetalTensor` and `DS4MetalQ4ExpertTable` wrappers, encoder
   batching.
2. **MSL compute library** (`metal/` subdir, 20 files, ~12k MSL lines):
   - `dense.metal` (1600 lines) -- `kernel_mul_mm_id_map0_ne20_*`
     templates for GEMM shapes.
   - `flash_attn.metal` (1429 lines) -- `kernel_flash_attn_ext_pad`,
     `_blk`, `_vec_reduce`. Threadgroup-tile online-softmax.
   - `moe.metal` (4606 lines) -- IQ2_XXS / Q4_K expert kernels with
     Q8_K activation quantized inline.
   - `dsv4_misc.metal, dsv4_hc.metal, dsv4_kv.metal, dsv4_rope.metal,
     softmax.metal, argsort.metal, unary.metal`, plus set_rows, cpy,
     norm, repeat, sum_rows, concat, bin.
3. **Metal 4 tensor API** (`ds4_metal.m:2003..1990` region): indirect
   dispatch for dynamic prefill shapes.

Apple caveats: no managed memory (explicit async memcpy), no tensor
cores (threadgroup tiling), function constants for shape specialization.

### 4.6 Backend selection

Compile-time, not runtime. `ds4_cuda.cu` picks CUDA vs ROCm via
`#ifdef __HIP_PLATFORM_AMD__`. Metal is a separate `.m` file. CPU-only
build sets `DS4_NO_GPU` and compiles a different source object
(`ds4_cpu.o`). No dispatch table -- backends are mutually exclusive
per-binary.

### 4.7 Distributed inference (`ds4_distributed.c`, 8.4k lines)

Coordinator + Worker topology over custom TCP:

- **Wire protocol**: 4-byte magic `0x44533444` ("DS4D"), message types:

      DS4_DIST_MSG_HELLO        (1)   worker handshake
      DS4_DIST_MSG_WORK         (3)   forward request (tokens, input HC, route)
      DS4_DIST_MSG_RESULT       (4)   hidden state or logits + telemetry
      DS4_DIST_MSG_SNAPSHOT_*   (5-8) KV checkpoint gather/scatter, 8 MiB chunks

- **Sharding**: **Pipeline-parallel by layer**. Each worker owns a
  contiguous `[layer_start, layer_end)` range. Coordinator runs its
  own local layers, chains hidden state to first worker, so on down
  the route.
- **Not tensor-parallel**, not expert-parallel. MoE gates run locally
  per worker.
- **Activation bit-width configurable** (`ds4_distributed_options.activation_bits`,
  default 32). Hidden compressed states cross the wire, not raw
  activations.
- No NCCL, no MPI. Single-threaded coordinator scheduling.

### 4.8 SSD streaming interaction

- Selective mmap: `ds4_gpu_set_model_map_range` / `_spans` marks which
  parts of the model file should be pinned in device caches.
- Expert cache: pre-allocated gate/up/down pools per layer with LRU
  eviction (CUDA: `ds4_cuda.cu:1431..1925`).
- Static hotlist: `ds4_streaming_hotlist.inc` is a pre-computed
  `[layer][expert]` histogram baked into the binary -- guides prefetch
  order.
- Metal has no async SSD path: shared-memory pools instead.

### 4.9 Quantized kernel SIMD paths

    IQ2_XXS      uint16_t d, uint16_t qs[32], 32 B/block
                 nested 2x2x2 grid + 7-bit signs + 4-entry LUT
                 CUDA: LUT in shared memory for warp efficiency
                 ROCm: amd_mixed_dot for byte dot
    Q4_K / Q2_K  256-dim super-blocks, grouped scales
    Q8_0         32-dim blocks, quantized inline via kernel
    Q8_K         256-dim activation match for GEMM
    F16          cuBLAS on sm_80+ tensor cores; Metal threadgroup tiling

---

## 5. Server, CLI, agent, eval, bench

### 5.1 HTTP server (`ds4_server.c`, 16k lines)

**Hand-rolled BSD sockets loop**. No libmicrohttpd, no mongoose, no
libevent. `socket -> bind -> listen -> accept` at `ds4_server.c:11347`.

Threading model:

- One blocking thread per client connection (`client_main` at
  `:11247`).
- Client thread parses the HTTP request (headers 64 KiB max, body
  64 MiB max), enqueues a `job` struct to a work queue, then waits on
  a condvar.
- **Single Metal worker thread** (`worker_main` at `:11067`) dequeues
  jobs and processes them serially. This thread owns the live
  `ds4_session` and the KV cache.
- Job queue `enqueue` / `dequeue` at `:11039..11065` with mutex.

Route table (`ds4_server.c:11265..11301`):

    GET  /v1/models              send_models       (OpenAI compat)
    GET  /v1/models/{alias}      send_model
    POST /v1/chat/completions    parse_chat_request         -> generate_job
    POST /v1/messages            parse_anthropic_request    -> generate_job
    POST /v1/completions         parse_completion_request   -> generate_job
    POST /v1/responses           parse_responses_request    -> generate_job

Anthropic + OpenAI compat share the same job path. No `/health`, no
`/metrics`, no `/v1/embeddings`. 404 fallback.

### 5.2 Tool-call handling: DSML

DeepSeek's native tool-call format is DSML:

    <|DSML|tool_calls>
      <|DSML|invoke name="$TOOL_NAME">
        <|DSML|parameter name="$P">value</|DSML|parameter>
      </|DSML|invoke>
    </|DSML|tool_calls>

Parser at `ds4_server.c:1070..1170` (`parse_function_call`,
`parse_tool_calls_value`). Detection in stream at `:5859..5935`.
Transcoding:

- DSML -> OpenAI `function_call` / `tool_use` for client consumption.
- Anthropic `tool_use` blocks -> DSML for the model (`:1784..1791`).

**Server does not execute tools** -- it only parses / transcodes.
Execution is the client's responsibility. The *agent* binary
(`ds4_agent.c`) is the tool executor.

### 5.3 SSE streaming

Content-Type `text/event-stream`. Per-token delta streaming with
UTF-8-safe splitting (`utf8_stream_safe_len` at `:1013..1036`) and
DSML-entity-safe splitting (`:5604..5635`). Final `stream_finish`
event carries usage stats (`completion_tokens`, `prompt_tokens`).

### 5.4 `ds4_web.c` -- Chrome DevTools Protocol harness

**Not a chat UI**. `ds4_web` is a web *tool* the agent uses: it
launches a controlled Chrome process (`:1023..1128`), connects via
WebSocket to Chrome's CDP (`:385..476`), and exposes two APIs:

    ds4_web_google_search()   navigate Google, extract links
    ds4_web_visit_page()      visit URL, scrape visible content via JS

Approval required before Chrome starts (`confirm_fn` callback,
`:1112..1127`).

### 5.5 CLI (`ds4_cli.c`)

One-shot mode (single prompt -> response -> exit) or interactive REPL
with `linenoise` history. Keeps a **persistent `ds4_session` across
turns** -- KV cache reuse across the whole conversation (unlike the
stateless HTTP server, where each POST creates a fresh job).

### 5.6 Agent (`ds4_agent.c`, 10k lines)

Single process, UI thread + Metal worker thread. Seven tools:

    read           read files from workspace
    write          write / append to files
    list           list directory contents
    edit           multi-line text editing with old/new spans
    bash           shell command execution (spawns subprocess)
    bash_status    check running job output
    bash_stop      terminate running bash job

Approval flow:

- Web tool (Chrome start) requires interactive confirmation (`agent_web_confirm`
  at `:4033..4056`).
- **File I/O and bash have NO pre-approval**. They run as the invoking
  user in the workspace directory.
- No sandboxing. No seccomp, no landlock, no macOS sandbox profile.

DSML parser at `:1420..1500` (`agent_dsml_parser_step`) is separate
from the server's parser -- the agent needs to *detect* tool calls in
the streaming output and stop generation before executing.

Tool-result memoization: two `rax` (radix tree) instances --
`tool_memory.by_id` keyed by execution id and `tool_memory.by_block`
keyed by DSML block text. Reused across turns to skip re-execution.

### 5.7 Eval harness (`ds4_eval.c`)

Regression benchmark against a **fixed embedded question bank**:

- GPQA Diamond (physics / biology / chemistry, 8 questions).
- SuperGPQA (audited subset).
- AIME 2025 (math competition).
- COMPSEC (5 audited C/C++ CVE localization questions).

Answer-key correctness grading only. Tracks whether the model used
extended thinking (`think_close_kind: NONE | NATURAL | SOFT | HARD`).
Live ANSI-color two-pane terminal UI (questions on left, live status
on right). No perplexity, no ensemble, no retry.

### 5.8 Bench harness (`ds4_bench.c`)

Throughput-focused. For a fixed token sequence walked to multiple
context frontiers (e.g. 2K, 4K, ..., 32K):

- Measure incremental prefill time per frontier
  (`ds4_bench.c:601..608`).
- Snapshot session in memory, decode 128 greedy tokens for decode
  throughput, restore snapshot, advance.
- Log KV cache size per frontier (`:671`).
- Optionally dump frontier logits for correctness validation.

Not TTFT-focused. Focus: sustained prefill + decode tok/s.

### 5.9 What's genuinely unique to DS4 vs the four Rust engines

- **Integrated coding agent + Chrome DevTools web tool** shipped as a
  first-party binary.
- **Fixed embedded eval bank** (not a generic runner).
- **DSML tool-call transcoding**: OpenAI and Anthropic both translate
  to and from DeepSeek's native tool format on the wire.
- **Live KV cache reuse across CLI turns** with `linenoise` history.
- **Speculative decoding via a small MTP transformer** (not draft-model
  style; MTP is trained specifically for DeepSeek).
- **Pipeline-parallel distributed inference over custom TCP** with
  configurable activation-bit width.
- **SSD-native expert streaming** with a static prebuilt hotlist.
- **Purpose-built for one model family** -- DeepSeek V4 Flash / PRO;
  everything from the tokenizer to the quantization to the chat
  template is DeepSeek-specific.

---

## 6. Reverse mapping to the four Rust engines

DS4 sits at the intersection of everything the other four teardowns
described. This section maps DS4's features back to what the four Rust
engines chose to do, showing where DS4 aligns and where it diverges.

### 6.1 Engine architecture

| Concern                        | DS4                                  | mistral.rs                         | TGI                                | Candle                             | tract                              |
|--------------------------------|--------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| Handle type                    | `ds4_engine` + `ds4_session`         | `MistralRs` + `Sequence`           | `MistralRs` per model + `Request`  | `Tensor` + `Module`                | `TypedModel` + `SimplePlan`+`State`|
| Concurrency                    | 1 client thread + 1 worker           | 1 OS thread + tokio per engine     | Rust router + Python shards        | library (single-threaded caller)   | library                            |
| Sessions across turns          | yes (CLI reuses KV)                  | yes (agentic session store, TTL)   | no (stateless HTTP)                | n/a                                | yes (SimpleState)                  |
| Prefix-cache trigger           | `sync -> rewrite_from_common`        | radix + block-hash chain (paged)   | radix trie (v3)                    | n/a                                | plan-time flush lists              |

### 6.2 Backends

| Backend         | DS4                       | mistral.rs        | TGI          | Candle              | tract           |
|-----------------|---------------------------|-------------------|--------------|---------------------|-----------------|
| CPU             | ds4_cpu.o (correctness)   | candle CPU        | Python       | candle_core         | linalg (SIMD)   |
| CUDA            | ds4_cuda.cu (13k)         | via candle        | Python+CUDA  | cudarc              | tract-cuda      |
| ROCm            | ds4_rocm.cu (thin)        | not native        | not native   | not native          | not native      |
| Metal           | ds4_metal.m (27k)         | via candle        | not native   | candle-metal-kernels| tract-metal     |
| Backend select  | compile-time              | compile-time      | shard-time   | runtime `Device`    | runtime         |

DS4 is the *only* engine with a first-class ROCm/Strix Halo port.

### 6.3 Quantization

| Format              | DS4           | mistral.rs        | Candle             | TGI (Python)  |
|---------------------|---------------|-------------------|--------------------|---------------|
| Q4_0/Q4_1/Q5_x/Q8_0 | via candle-style layouts | full (via mistralrs-quant + candle) | full (candle-core::quantized) | GGUF via llama-loader |
| Q4_K, Q6_K, Q2_K    | Q4_K, Q2_K    | full via GGUF     | full               | full          |
| IQ2_XXS             | yes (baked LUTs) | not visible    | not in candle base | not visible   |
| GPTQ / AWQ / Marlin | no            | yes (via marlin)  | not native         | yes (Python)  |
| HQQ / AFQ           | no            | yes (mistralrs-quant) | no             | no            |

DS4's IQ2 tables for MoE gate/up projections are its defining quant
choice. mistral.rs and Candle both handle it in principle via GGUF but
neither ship the CUDA lookup-table kernel; DS4 does.

### 6.4 Distributed inference

- **DS4**: pipeline parallel across layer ranges, custom TCP protocol
  with configurable activation bit-width, hidden-compressed states on
  the wire.
- **mistral.rs**: NCCL + Ring TP via `mistralrs_quant::Comm`. TP, not
  PP. No PP path.
- **TGI**: launcher spawns N Python shards; each shard uses
  `torch.distributed` (NCCL) for TP. PP is implicit via multi-node
  worker layout, not a first-class protocol.
- **Candle**: library, no distributed layer.
- **tract**: library, no distributed layer.

DS4's pipeline-parallel wire protocol is its own thing. Nothing in the
Rust ecosystem matches the design: activation-bit compression + custom
TCP + coordinator/worker gather-scatter of KV snapshots.

### 6.5 Server surface

| API                       | DS4              | mistral.rs        | TGI              |
|---------------------------|------------------|-------------------|------------------|
| OpenAI chat completions   | yes              | yes               | yes              |
| Anthropic messages        | yes              | yes               | no               |
| KServe / SageMaker / Vertex| no              | no                | yes              |
| MCP client                | no               | yes               | no               |
| MCP server                | no               | yes               | no               |
| Tool-call auto-parse      | DSML             | 8 per-model formats + llguidance | 1 tool grammar |
| SSE streaming             | yes              | yes               | yes              |
| Web UI                    | no (chrome tool) | yes (bundled)     | no               |
| Metrics / Prometheus      | no               | via metrics crate | yes              |
| Approval broker           | agent local      | yes (server-side) | no               |

DS4's server is deliberately minimal; the *agent* binary is where the
tool-execution + approval story lives.

### 6.6 SSD / disk KV cache

Only DS4 treats SSD KV as first-class:

- **DS4**: sessions have a `DSV4` payload magic. Per-layer `DSVL`
  payload for distributed. Static hotlist for expert prefetch. LRU on
  disk (`ds4_kvstore`).
- **mistral.rs**: RAM only. Prefix cache is in-memory.
- **TGI**: RAM only. Radix trie is in-memory.
- **Candle**: none.
- **tract**: none.

This is the biggest single design bet DS4 makes: KV cache is a disk
citizen, model weights can spill to SSD via expert streaming, and the
runtime turns "does the model fit in RAM" from a hard cutoff into a
speed knob.

### 6.7 Model catalogue

DS4 supports **one model**: DeepSeek V4 Flash + PRO (with dedicated
weights, not arbitrary GGUFs of DeepSeek).

- mistral.rs supports ~34 causal + ~22 vision + speech + embedding +
  diffusion architectures.
- Candle supports ~106 model files.
- tract supports "any ONNX/NNEF/TF/TFLite model".
- TGI supports "any HF causal LM with a Python impl".

DS4 is the polar opposite of the other four. This is the readme's
"deliberately narrow" bet.

---

## 7. Verdict

DS4 (DwarfStar) is a **DeepSeek-V4-specific, disk-KV-native, C-language
inference engine with an integrated agent, one-model tokenizer + chat
template, IQ2 mixture-of-experts quantization, and pipeline-parallel
distributed inference over a custom TCP protocol**.

The three most important things to internalize about DS4:

1. **The `session_sync -> rewrite_from_common -> REBUILD_NEEDED` state
   machine.** This is the whole prefix-cache API in three enum values
   and one function. Callers push a full prompt every time; the
   session figures out reuse. mistral.rs, TGI, and Candle each
   reimplement variations of this idea more elaborately.
2. **KV cache as a first-class disk citizen.** DSV4/DSVL on-disk
   formats, per-session and per-layer, plus SSD-streamed experts with
   a static hotlist. Nothing in the four Rust engines even
   contemplates this.
3. **The compile-time backend split without runtime dispatch.** DS4's
   header is virtual-free C; the backend is chosen at Makefile time.
   This is what makes ROCm a first-class citizen (`ds4_rocm.cu`
   swaps in for `ds4_cuda.o`) without any middleware.

DS4's single biggest cost of being what it is: **no model catalogue,
no dispatch flexibility, no generic Rust ergonomics**. It is a
one-model, one-language, one-purpose engine. In exchange it gets
Metal + CUDA + ROCm all first-class, an integrated agent + eval + bench,
and disk-native KV. That trade is what the README is describing when it
says "deliberately narrow".

Mapping back to the four prior teardowns: DS4 has already, in C, most
of what the other four give you in Rust -- with the specific twist of
being DeepSeek-V4-only and disk-native. What it *doesn't* have that
the Rust ecosystem does is (a) a portable model catalogue, (b) a
production Python-worker split (TGI style), (c) a graph-IR compiler
(tract style), and (d) a broad ISQ / UQFF quantization matrix
(mistral.rs style).

If DS4 wants to broaden without losing its identity: the cheapest wins
are **lifting `mistralrs-paged-attn` kernels** into `ds4_cuda.cu` for
paged attention (DS4 currently uses sliding-window + compressed hybrid,
not paged), **lifting `tract-linalg` for a real CPU path** to replace
the correctness-only CPU fallback, and **adopting the `Backend` trait
shape** from TGI so DS4's server can drive multiple engines (not just
DwarfStar) behind the same DSML/OpenAI/Anthropic surface.

The full stack the other four engines together suggest is: keep DS4's
core (session, KV, disk payload, distributed protocol, DSML), swap the
CPU path to `tract-linalg`, add paged attention from `mistralrs-paged-attn`
next to DS4's own compressed hybrid, and expose the whole thing as a
gRPC v3-compatible backend so TGI's Rust router can drive it too. That
gives DS4 access to TGI's whole compat surface (SageMaker / Vertex /
KServe / OpenAI / Anthropic) at zero code cost. Whether that's worth
the coupling is the real design question -- but the four teardowns in
this folder now make it possible to answer.
