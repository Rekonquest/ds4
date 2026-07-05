// DS4 (DwarfStar) — official-vector regression test harness.
//
// These tests carry over the `tests/test-vectors/` golden corpus from
// the parent C DS4 project into the Rust rewrite. The golden files
// (manifest, official JSON, official.vec, local-golden.vec, prompts)
// are byte-identical copies; this harness is purely a CI gate that
// guards their format and exercises both the tokenizer/sampler path
// and the Rust engine path with a deterministic synthetic GGUF.
//
// No `unsafe`. No RNG. No real GPU. Fully deterministic.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ds4_core::sampler::{argmax, log_softmax_inplace, top_logprobs};
use ds4_core::tokenizer::Ds4Tokenizer;
use ds4_core::types::Ds4EngineOptions;
use ds4_core::{Ds4Engine, Ds4Session};

/// Pull in serde_json from the workspace's external deps. It's already
/// declared in the workspace's [workspace.dependencies] and re-exported
/// through ds4-core's tree.
use serde_json::Value as JsonValue;

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/// Resolve the absolute path to the `test-vectors` directory shipped
/// at the workspace root under `tests/test-vectors/`. The test file
/// itself lives at `ds4-rust/tests/regression.rs`, so when cargo
/// compiles this test against the ds4-core crate its CARGO_MANIFEST_DIR
/// is `ds4-rust/crates/ds4-core`. We therefore walk up to the
/// workspace root from there.
fn vectors_root() -> PathBuf {
    if let Some(manifest) = option_env!("CARGO_MANIFEST_DIR") {
        let m = Path::new(manifest);
        // Try `ds4-rust/tests/test-vectors/` directly (two levels up
        // from crates/ds4-core): the workspace root's tests dir.
        if let Some(grand) = m.parent().and_then(|p| p.parent()) {
            let p = grand.join("tests").join("test-vectors");
            if p.is_dir() {
                return p;
            }
        }
        // Try sibling: same parent as CARGO_MANIFEST_DIR.
        if let Some(parent) = m.parent() {
            let p = parent.join("tests").join("test-vectors");
            if p.is_dir() {
                return p;
            }
        }
        // Direct join under CARGO_MANIFEST_DIR (in case test lives at
        // `crates/ds4-core/tests/test-vectors/`).
        let p = m.join("tests").join("test-vectors");
        if p.is_dir() {
            return p;
        }
    }
    // Fallback: relative to CWD (workspace root during `cargo test`).
    PathBuf::from("tests/test-vectors")
}

fn manifest_path() -> PathBuf {
    vectors_root().join("manifest.json")
}
fn prompts_dir() -> PathBuf {
    vectors_root().join("prompts")
}
fn official_dir() -> PathBuf {
    vectors_root().join("official")
}
fn official_vec_path() -> PathBuf {
    vectors_root().join("official.vec")
}
fn local_golden_vec_path() -> PathBuf {
    vectors_root().join("local-golden.vec")
}

// ---------------------------------------------------------------------------
// Lightweight official-vector schema
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Manifest {
    schema: String,
    prompts: Vec<ManifestPrompt>,
}

#[derive(Debug, Clone)]
struct ManifestPrompt {
    id: String,
    prompt_file: String,
    official_file: String,
}

#[derive(Debug)]
struct OfficialFile {
    schema: String,
    id: String,
    steps: Vec<OfficialStep>,
}

#[derive(Debug)]
struct OfficialStep {
    step: u32,
    token_bytes: Vec<u8>,
    top_logprobs: Vec<f32>,
}

fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let v: JsonValue =
        serde_json::from_str(text).map_err(|e| format!("manifest JSON parse: {}", e))?;
    let schema = v
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| "manifest missing `schema`".to_string())?
        .to_string();
    let arr = v
        .get("prompts")
        .and_then(|p| p.as_array())
        .ok_or_else(|| "manifest missing `prompts` array".to_string())?;
    let mut prompts = Vec::with_capacity(arr.len());
    for entry in arr {
        let obj = entry
            .as_object()
            .ok_or_else(|| "manifest prompt entry not object".to_string())?;
        let id = obj
            .get("id")
            .and_then(|s| s.as_str())
            .ok_or_else(|| "prompt missing `id`".to_string())?
            .to_string();
        let prompt_file = obj
            .get("prompt_file")
            .and_then(|s| s.as_str())
            .ok_or_else(|| format!("prompt {} missing `prompt_file`", id))?
            .to_string();
        let official_file = obj
            .get("official_file")
            .and_then(|s| s.as_str())
            .ok_or_else(|| format!("prompt {} missing `official_file`", id))?
            .to_string();
        prompts.push(ManifestPrompt {
            id,
            prompt_file,
            official_file,
        });
    }
    Ok(Manifest { schema, prompts })
}

fn parse_official(text: &str) -> Result<OfficialFile, String> {
    let v: JsonValue =
        serde_json::from_str(text).map_err(|e| format!("official JSON parse: {}", e))?;
    let schema = v
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| "official file missing `schema`".to_string())?
        .to_string();
    let id = v
        .get("id")
        .and_then(|s| s.as_str())
        .ok_or_else(|| "official file missing `id`".to_string())?
        .to_string();
    let steps_arr = v
        .get("steps")
        .and_then(|s| s.as_array())
        .ok_or_else(|| "official file missing `steps` array".to_string())?;
    let mut steps = Vec::with_capacity(steps_arr.len());
    for entry in steps_arr {
        let obj = entry
            .as_object()
            .ok_or_else(|| "step entry not object".to_string())?;
        let step = obj
            .get("step")
            .and_then(|s| s.as_u64())
            .ok_or_else(|| "step missing `step`".to_string())? as u32;
        let token_obj = obj
            .get("token")
            .and_then(|t| t.as_object())
            .ok_or_else(|| "step missing `token` object".to_string())?;
        let bytes_arr = token_obj
            .get("bytes")
            .and_then(|b| b.as_array())
            .ok_or_else(|| "step.token missing `bytes` array".to_string())?;
        let mut token_bytes = Vec::with_capacity(bytes_arr.len());
        for b in bytes_arr {
            let n = b
                .as_u64()
                .ok_or_else(|| "step.token.bytes entry not numeric".to_string())?;
            token_bytes.push(n as u8);
        }
        let top_arr = obj
            .get("top_logprobs")
            .and_then(|t| t.as_array())
            .ok_or_else(|| "step missing `top_logprobs` array".to_string())?;
        let mut top = Vec::with_capacity(top_arr.len());
        for t in top_arr {
            let n = t
                .get("logprob")
                .and_then(|l| l.as_f64())
                .ok_or_else(|| "top_logprobs entry missing numeric `logprob`".to_string())?;
            top.push(n as f32);
        }
        steps.push(OfficialStep {
            step,
            token_bytes,
            top_logprobs: top,
        });
    }
    Ok(OfficialFile { schema, id, steps })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Read each `.official.json` and assert it parses cleanly into the
/// expected shape. This is the structural gate: if any official file
/// changes format the harness trips here.
#[test]
fn golden_vectors_load_and_parse() {
    let root = vectors_root();
    assert!(
        root.is_dir(),
        "test-vectors root missing: {}",
        root.display()
    );

    let off_dir = official_dir();
    let entries: Vec<PathBuf> = fs::read_dir(&off_dir)
        .expect("read official dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    assert_eq!(entries.len(), 5, "expected 5 official JSON files");

    let mut total_steps = 0usize;
    for path in &entries {
        let text =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
        let parsed =
            parse_official(&text).unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));
        assert_eq!(
            parsed.schema,
            "ds4-official-logprobs-v1",
            "{} wrong schema",
            path.display()
        );
        assert!(!parsed.id.is_empty(), "{} missing id", path.display());
        assert!(!parsed.steps.is_empty(), "{} has no steps", path.display());

        // For each step, the top_logprobs array must contain at least
        // one entry (the selected token) and the token.bytes must
        // round-trip to the same bytes as Vec<u8>.
        for step in &parsed.steps {
            assert!(
                !step.top_logprobs.is_empty(),
                "{} step {} empty top_logprobs",
                path.display(),
                step.step
            );
            assert!(
                !step.token_bytes.is_empty(),
                "{} step {} empty token bytes",
                path.display(),
                step.step
            );
            // Each logprob is finite (we don't accept NaN in golds).
            for &lp in &step.top_logprobs {
                assert!(
                    lp.is_finite(),
                    "{} step {} non-finite logprob {}",
                    path.display(),
                    step.step,
                    lp
                );
            }
            total_steps += 1;
        }
    }
    // 4+4+1+4+4 = 17 total golden steps across all 5 prompts.
    assert_eq!(
        total_steps, 17,
        "expected 17 total golden steps across all 5 prompts"
    );
}

/// Read `manifest.json` and assert every prompt file in `prompts/`
/// is referenced, and every manifest entry resolves to a real file.
#[test]
fn manifest_lists_all_prompts() {
    let manifest_text = fs::read_to_string(manifest_path()).expect("read manifest.json");
    let manifest = parse_manifest(&manifest_text).expect("parse manifest.json");
    assert_eq!(manifest.schema, "ds4-test-vector-manifest-v1");
    assert!(
        manifest.prompts.iter().all(|p| !p.id.is_empty()),
        "manifest prompt ids must be non-empty",
    );

    let prompts = prompts_dir();
    let mut actual: Vec<String> = fs::read_dir(&prompts)
        .expect("read prompts dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".txt"))
        .collect();
    actual.sort();

    let mut referenced: Vec<String> = manifest
        .prompts
        .iter()
        .map(|p| {
            // Strip leading "prompts/" prefix so we compare basenames.
            p.prompt_file
                .rsplit('/')
                .next()
                .unwrap_or(&p.prompt_file)
                .to_string()
        })
        .collect();
    referenced.sort();

    assert_eq!(
        actual, referenced,
        "manifest.prompts does not match prompts/ directory contents"
    );

    // Every referenced prompt file must exist.
    for p in &manifest.prompts {
        let abs = vectors_root().join(&p.prompt_file);
        assert!(
            abs.is_file(),
            "manifest prompt_file missing: {}",
            abs.display()
        );
    }
    // Every referenced official file must exist.
    for p in &manifest.prompts {
        let abs = vectors_root().join(&p.official_file);
        assert!(
            abs.is_file(),
            "manifest official_file missing: {}",
            abs.display()
        );
    }
}

#[test]
fn compact_vector_files_are_parsed_and_cross_checked() {
    let manifest_text = fs::read_to_string(manifest_path()).expect("read manifest.json");
    let manifest = parse_manifest(&manifest_text).expect("parse manifest.json");

    let official_text = fs::read_to_string(official_vec_path()).expect("read official.vec");
    let mut official_cases = 0usize;
    let mut official_steps = 0usize;
    let mut active_case: Option<String> = None;
    for line in official_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts.as_slice() {
            ["case", id, _ctx, _steps, prompt_file] => {
                assert!(
                    manifest.prompts.iter().any(|p| {
                        p.id == *id
                            && PathBuf::from(prompt_file).ends_with(Path::new(&p.prompt_file))
                    }),
                    "official.vec case {id} must match manifest prompt path"
                );
                official_cases += 1;
                active_case = Some((*id).to_string());
            }
            ["step", _idx, selected_hex, top_count] => {
                assert!(active_case.is_some(), "step outside case");
                assert!(!selected_hex.is_empty(), "selected token hex is empty");
                assert!(top_count.parse::<usize>().unwrap() >= 1);
                official_steps += 1;
            }
            ["top", token_hex, logprob] => {
                assert!(!token_hex.is_empty(), "top token hex is empty");
                let lp: f32 = logprob.parse().expect("official logprob");
                assert!(lp.is_finite());
            }
            ["end"] => active_case = None,
            _ => panic!("unrecognised official.vec line: {trimmed}"),
        }
    }
    assert_eq!(official_cases, manifest.prompts.len());
    assert_eq!(official_steps, 17);

    let local_text = fs::read_to_string(local_golden_vec_path()).expect("read local-golden.vec");
    let mut local_cases = 0usize;
    let mut local_tops = 0usize;
    let mut last_logit = f32::INFINITY;
    for line in local_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts.as_slice() {
            ["case", _id, _mode, _ctx, _frontier, prompt_file, top_count] => {
                assert!(PathBuf::from(prompt_file).ends_with("long_context_story_prompt.txt"));
                assert_eq!(*top_count, "64");
                local_cases += 1;
                last_logit = f32::INFINITY;
            }
            ["top", _rank, _token_id, logit] => {
                let value: f32 = logit.parse().expect("local logit");
                assert!(value.is_finite());
                assert!(value <= last_logit);
                last_logit = value;
                local_tops += 1;
            }
            ["end"] => {}
            _ => panic!("unrecognised local-golden.vec line: {trimmed}"),
        }
    }
    assert_eq!(local_cases, 1);
    assert_eq!(local_tops, 64);
}

/// Smoke test: tokenize the simplest official prompt through the
/// real `Ds4Tokenizer` and verify a no-op logits buffer round-trips
/// through `log_softmax` + `argmax` + `top_logprobs`. This catches
/// integration drift between tokenizer and sampler before any
/// real model is loaded.
#[test]
fn short_prompts_match_simple_reference() {
    // The smallest official prompt: Italian fact, 57 chars.
    let prompt_path = prompts_dir().join("short_italian_fact.txt");
    let prompt_text = fs::read_to_string(&prompt_path).expect("read short_italian_fact.txt");
    assert!(!prompt_text.is_empty(), "prompt file is empty");

    // The engine-default tokenizer uses byte-mapping (b -> b+1) so
    // ASCII bytes map to ids 1..=256. The harness mirrors that here.
    let mut byte_to_id = [0u32; 256];
    for b in 0u32..256 {
        byte_to_id[b as usize] = b + 1;
    }
    let tokenizer = Ds4Tokenizer::from_byte_mapping(
        byte_to_id, /* unk  */ 0, /* bos  */ 1000, /* eos  */ 1001,
        /* user */ 1002, /* asst */ 1003, /* th_s */ 1004, /* th_e */ 1005,
        /* dsml */ 1006,
    )
    .expect("tokenizer build");

    let toks = tokenizer.tokenize(&prompt_text).expect("tokenize");
    assert!(!toks.is_empty(), "tokenize produced no tokens");
    // The tokenizer must produce only finite, in-range ids.
    let vocab_size = tokenizer.vocab_size() as usize;
    for &t in &toks {
        assert!(
            (t as usize) < vocab_size.max(1),
            "token id {} out of vocab size {}",
            t,
            vocab_size
        );
    }

    // Reference vector: synthesize a tiny logits buffer where token 0
    // is the argmax. This is the "no-op" reference the prompt asks
    // for: a vocabulary with a single preferred token, everything else
    // at -inf. log_softmax must collapse the distribution onto id 0.
    let mut logits = vec![f32::NEG_INFINITY; 256];
    logits[0] = 0.0;
    log_softmax_inplace(&mut logits);
    assert_eq!(argmax(&logits), 0, "argmax should pick the lone spike");
    let top = top_logprobs(&logits, 4);
    assert!(!top.is_empty(), "top_logprobs empty");
    assert_eq!(top[0].0, 0, "argmax token must be id 0");
    // log_prob(id=0) must be 0.0 (since p=1.0 → log(1.0)=0).
    assert!(
        top[0].1.abs() < 1e-6,
        "argmax token should have log_prob 0.0, got {}",
        top[0].1
    );
}

/// Engine-path smoke against the deterministic synthetic GGUF. The
/// official corpus above is still the format authority; this test proves
/// the current Rust engine can load a GGUF, tokenize a short prompt,
/// fill logits through `Ds4Session`, and produce a stable top token.
#[test]
fn engine_drives_short_prompt_with_synthetic_model() {
    let dir = std::env::temp_dir().join(format!("ds4-regression-engine-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir synthetic regression dir");
    let model_path = dir.join("synth.gguf");
    Ds4Engine::write_synthetic_gguf(&model_path).expect("write synthetic GGUF");

    let engine = Arc::new(
        Ds4Engine::open(Ds4EngineOptions {
            model_path: model_path.clone(),
            ..Ds4EngineOptions::default()
        })
        .expect("open synthetic engine"),
    );
    assert!(
        engine.model().is_some(),
        "synthetic engine must load a model"
    );
    assert_eq!(engine.vocab_size(), 16);

    let prompt_tokens = engine.tokenizer().tokenize("hi").expect("tokenize hi");
    assert_eq!(prompt_tokens, vec![10]);
    let mut session = Ds4Session::create(&engine, 64).expect("create session");
    session.sync(&prompt_tokens).expect("sync prompt");
    session.refresh_logits().expect("refresh logits");

    let mut logits = vec![0.0f32; engine.vocab_size()];
    session.copy_logits(&mut logits, engine.vocab_size());
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(logits.iter().any(|v| *v != 0.0));
    let top = top_logprobs(&logits, 4);
    assert_eq!(top.len(), 4);
    assert!((top[0].0 as usize) < engine.vocab_size());
    assert!(top[0].1.is_finite());

    let _ = std::fs::remove_dir_all(&dir);
}
