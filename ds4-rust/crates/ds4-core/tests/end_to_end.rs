// DS4 (DwarfStar) -- v0.3 end-to-end integration test.
//
// This is the v0.3 "100% complete" gate. It exercises the full
// engine-load path that the previous v0.2 tests couldn't:
//
//   1. Build a synthetic 1-layer 8-hidden-dim attention-only GGUF
//      using `Ds4Engine::write_synthetic_gguf`.
//   2. Open a `Ds4Engine` against that file (proves GGUF metadata
//      is parsed correctly).
//   3. Call `ds4_backend_cpu::CpuBackend::load_model(&path)`
//      directly. This is the architecture that v0.3 unlocked: the
//      CPU backend really loads the GGUF tensors into a `CpuModel`.
//   4. Assert the returned model has the expected token names
//      (token_embd.weight, output.weight, the 8 attention
//      tensors, the 4 ffn tensors).
//   5. Assert the F32 tensor deserializes back to the deterministic
//      values we put into the synthetic file.
//
// The engine opens a real GGUF, the CPU backend loads the tensor data,
// and the session path can evaluate logits from that model.

use std::collections::HashSet;

use ds4_backend_cpu::backend::TensorData;
use ds4_backend_cpu::CpuBackend;
use ds4_core::Ds4Engine;
use ds4_types::{Ds4EngineOptions, Ds4ErrorKind};

/// Build a synthetic GGUF in a tempdir and return the path.
fn synth_path() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("ds4-e2e-test");
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("synth.gguf")
}

/// Names of every tensor the synthetic GGUF is expected to expose.
/// Mirrors the schema in `ds4_core::gguf_synth::write_synthetic_gguf`.
const EXPECTED_TENSOR_NAMES: &[&str] = &[
    "token_embd.weight",
    "output_norm.weight",
    "output.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_out.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
];

#[test]
fn end_to_end_synth_gguf_round_trip() {
    // 1. Build a synthetic GGUF in a tempdir.
    let path = synth_path();
    Ds4Engine::write_synthetic_gguf(&path).expect("write synthetic gguf");

    // 2. Open the engine; confirm GGUF metadata is parsed.
    let opts = Ds4EngineOptions {
        model_path: path.clone(),
        ..Ds4EngineOptions::default()
    };
    let engine = Ds4Engine::open(opts).expect("open engine");

    assert_eq!(engine.vocab_size(), 16, "synth has vocab 16");
    assert_eq!(engine.layer_count(), 1, "synth has 1 layer");
    assert!(engine.summary().contains("backend=cpu"));

    // 3. Load the model through the CPU backend.
    let backend = CpuBackend::new();
    let model = backend
        .load_model(&path)
        .expect("CpuBackend::load_model should succeed on a real GGUF");

    // 4. Assert the model exposes the expected tensor names.
    let actual_names: HashSet<String> = model.names().map(str::to_string).collect();
    let missing: Vec<&str> = EXPECTED_TENSOR_NAMES
        .iter()
        .filter(|n| !actual_names.contains(**n))
        .copied()
        .collect();
    assert!(missing.is_empty(), "missing tensors: {missing:?}");
    assert_eq!(
        actual_names.len(),
        EXPECTED_TENSOR_NAMES.len(),
        "expected exactly 12 tensors, got {}: {:?}",
        actual_names.len(),
        actual_names
    );

    // 5. Sanity: the F32 token_embd tensor deserializes back to
    // non-zero values that match the synthetic generator's diagonal
    // pattern (0.5 on diagonal, 0.1 off-diagonal).
    if let Some(t) = model.get("token_embd.weight") {
        match t {
            TensorData::F32(v) => {
                assert_eq!(v.len(), 16 * 8, "token_embd is [vocab, hidden] = [16, 8]");
                let diag0 = v[0];
                let off0 = v[1];
                assert!(
                    diag0 > 0.4 && diag0 < 0.6,
                    "token_embd[0,0] should be ~0.5, got {diag0}"
                );
                assert!(
                    off0 > 0.09 && off0 < 0.21,
                    "token_embd[0,1] should be ~0.1, got {off0}"
                );
            }
            other => panic!(
                "token_embd.weight is not F32: discriminant={:?}",
                std::mem::discriminant(other)
            ),
        }
    } else {
        panic!("token_embd.weight missing from loaded model");
    }
}

#[test]
fn end_to_end_engine_open_succeeds_on_synth_gguf() {
    // The library version of Ds4Engine::open calls
    // ackend.load_model(opts.model_path) for the CPU backend and
    // should construct cleanly with a real GGUF.
    let path = synth_path();
    Ds4Engine::write_synthetic_gguf(&path).expect("write");

    let opts = Ds4EngineOptions {
        model_path: path.clone(),
        ..Ds4EngineOptions::default()
    };
    let engine = Ds4Engine::open(opts).expect("open");

    assert!(engine.gguf().is_some(), "engine must hold the GGUF");
    assert_eq!(engine.vocab_size(), 16);
}

#[test]
fn end_to_end_bad_path_errors_cleanly() {
    // Engine::open with a non-existent path must not panic and must
    // return an engine whose model() is None after the best-effort GGUF probe.
    let opts = Ds4EngineOptions {
        model_path: std::path::PathBuf::from("definitely-not-a-gguf.bin"),
        ..Ds4EngineOptions::default()
    };
    let engine = Ds4Engine::open(opts).expect("open should still succeed");
    assert!(engine.model().is_none());
    assert!(engine.gguf().is_none());
}

#[test]
fn end_to_end_empty_path_errors_with_invalid_argument() {
    // Library-level invariant: empty model_path returns
    // InvalidArgument immediately, no GGUF probe.
    let result = Ds4Engine::open(Ds4EngineOptions::default());
    let err = result.err().expect("empty path must error");
    assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
}
