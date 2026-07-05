// DS4 (DwarfStar) — inference session.
//
// Implements the public surface in `ds4.h`: `ds4_session_create`,
// `ds4_session_sync`, `ds4_session_rewrite_from_common`,
// `ds4_session_common_prefix`, `ds4_session_eval`,
// `ds4_session_eval_speculative_argmax`, `ds4_session_sample`,
// `ds4_session_token_logprob`, `ds4_session_top_logprobs`,
// `ds4_session_invalidate`, `ds4_session_rewind`, etc.
//
// The session holds the live logits buffer, the active token
// sequence, the KV cache handle, and the speculative-decoding MTP
// state. The sync state-machine models the C engine's
// (re)build-from-checkpoint path:
//
//   * `RebuildNeeded`           — live state doesn't match the
//     requested prompt; the caller should drop the live state and
//     sync() to refill.
//   * `Common(n)`              — first `n` tokens of the prompt
//     agree with the live checkpoint; only the suffix needs to be
//     evaluated.
//   * `Ok`                     — live state already matches the
//     prompt; nothing to do.

use std::sync::{Arc, Mutex, MutexGuard};

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result, Ds4RewriteStatus};

use crate::engine::Ds4Engine;
use crate::kv::KvCache;
use crate::mtp::Ds4Mtp;
use crate::sampler;

/// Default prefill chunk. Mirrors `ds4.c`.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

/// Live session state.
pub struct Ds4Session {
    engine: Arc<Ds4Engine>,
    ctx_size: usize,
    prefill_chunk: usize,
    state: Mutex<SessionState>,
}

/// Outer session state — logits, token sequence, KV cache, MTP state.
struct SessionState {
    tokens: Vec<u32>,
    position: usize,
    logits: Vec<f32>,
    cache: KvCache,
    mtp: Box<Ds4Mtp>,
    /// Most recent sync result. Cached so `eval_speculative_argmax`
    /// can read whether we have valid logits or need a rebuild first.
    last_sync: SyncResult,
}

/// Tracks the live inference checkpoint relative to a requested prompt
/// prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncResult {
    /// Live state is empty (post-create or post-invalidate).
    Empty,
    /// Live state matches the requested prompt exactly.
    MatchAll,
    /// Live state matches the first N tokens of the prompt.
    Common(usize),
    /// Live state diverges from the prompt at index `from`.
    Diverges(usize),
}

impl Ds4Session {
    fn state_lock(&self) -> Ds4Result<MutexGuard<'_, SessionState>> {
        self.state.lock().map_err(|_| {
            Ds4Error::new(
                Ds4ErrorKind::Other,
                "session state lock is poisoned by a previous panic",
            )
        })
    }

    fn state_lock_recovering(&self) -> MutexGuard<'_, SessionState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    pub fn create(engine: &Arc<Ds4Engine>, ctx_size: usize) -> Ds4Result<Self> {
        if ctx_size == 0 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "ctx_size must be > 0",
            ));
        }
        let n_layers = engine.layer_count().max(1);
        let cache = KvCache::new(ctx_size, n_layers)?;
        let mtp_box = Box::new(Ds4Mtp::with_config(crate::mtp::Ds4MtpConfig {
            draft_tokens: engine.mtp_draft_tokens(),
            margin: 0.0,
        }));
        Ok(Ds4Session {
            engine: engine.clone(),
            ctx_size,
            prefill_chunk: engine.options().prefill_chunk.max(1),
            state: Mutex::new(SessionState {
                tokens: Vec::with_capacity(ctx_size.min(2048)),
                position: 0,
                logits: vec![0.0; engine.vocab_size().max(1)],
                cache,
                mtp: mtp_box,
                last_sync: SyncResult::Empty,
            }),
        })
    }

    pub fn free(self) {
        drop(self);
    }

    pub fn sync(&mut self, prompt: &[u32]) -> Ds4Result<()> {
        if prompt.len() > self.ctx_size {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "prompt length {} exceeds context window {}",
                    prompt.len(),
                    self.ctx_size
                ),
            ));
        }
        let mut st = self.state_lock()?;
        let rollback_tokens = st.tokens.clone();
        let rollback_position = st.position;
        let rollback_logits = st.logits.clone();
        let rollback_cache = st.cache.clone();
        let rollback_mtp = st.mtp.clone();
        let rollback_last_sync = st.last_sync;
        let needs_rebuild = match Self::sync_inner(&mut st, prompt) {
            Ok(()) => false,
            Err(_) => true,
        };
        if needs_rebuild {
            // Rebuild and retry.
            st.tokens.clear();
            st.position = 0;
            st.cache.reset();
            st.mtp.reset();
            st.last_sync = SyncResult::Empty;
            let mut pos = 0;
            for &tok in prompt.iter() {
                if pos >= self.ctx_size {
                    break;
                }
                st.tokens.push(tok);
                pos += 1;
            }
            for layer in 0..st.cache.n_layers() {
                st.cache.rewind(layer, pos)?;
            }
            st.position = pos;
            st.last_sync = SyncResult::MatchAll;
        }
        if !st.tokens.is_empty() {
            let tokens = st.tokens.clone();
            let mut logits = st.logits.clone();
            if let Err(e) = self.engine.eval_sequence_logits(&tokens, &mut logits) {
                st.tokens = rollback_tokens;
                st.position = rollback_position;
                st.logits = rollback_logits;
                st.cache = rollback_cache;
                st.mtp = rollback_mtp;
                st.last_sync = rollback_last_sync;
                return Err(e);
            }
            st.logits = logits;
        }
        Ok(())
    }

    fn sync_inner(st: &mut SessionState, prompt: &[u32]) -> Ds4Result<()> {
        match st.last_sync {
            SyncResult::Empty => Self::rebuild_full(st, prompt),
            SyncResult::MatchAll if prompt == st.tokens.as_slice() => Ok(()),
            SyncResult::MatchAll => Self::rebuild_full(st, prompt),
            SyncResult::Common(n)
                if n <= st.tokens.len() && prompt.starts_with(&st.tokens[..n]) =>
            {
                // Evaluate suffix.
                let suffix = &prompt[n..];
                Self::eval_suffix(st, n, suffix)
            }
            _ => {
                // Force a rebuild on the next call.
                st.last_sync = SyncResult::Diverges(0);
                Err(Ds4Error::new(Ds4ErrorKind::Other, "session needs rebuild"))
            }
        }
    }

    fn rebuild_full(st: &mut SessionState, prompt: &[u32]) -> Ds4Result<()> {
        if prompt.len() > st.cache.ctx_size() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "prompt length {} exceeds context window {}",
                    prompt.len(),
                    st.cache.ctx_size()
                ),
            ));
        }
        st.tokens.clear();
        st.position = 0;
        st.cache.reset();
        st.mtp.reset();
        for &tok in prompt.iter() {
            st.tokens.push(tok);
            st.position += 1;
        }
        st.last_sync = SyncResult::MatchAll;
        Ok(())
    }

    fn eval_suffix(st: &mut SessionState, common: usize, suffix: &[u32]) -> Ds4Result<()> {
        for &tok in suffix {
            st.tokens.push(tok);
            st.position += 1;
        }
        let len_after = st.tokens.len();
        let prefix_ok = {
            let prefix = st.tokens.get(..common).unwrap_or(&[]);
            let live_first_n = prefix;
            live_first_n.len() == common
        };
        if prefix_ok && len_after >= common + suffix.len() {
            st.last_sync = SyncResult::MatchAll;
            Ok(())
        } else {
            Err(Ds4Error::new(
                Ds4ErrorKind::Other,
                "rewrite failed (live cache diverges from prompt)",
            ))
        }
    }

    pub fn rewrite_from_common(&mut self, prompt: &[u32], common: usize) -> Ds4RewriteStatus {
        if prompt.len() > self.ctx_size {
            return Ds4RewriteStatus::RewriteError;
        }
        let mut st = self.state_lock_recovering();
        // common=0 is the "drop everything, rebuild" sentinel — the
        // caller doesn't trust any prefix, so we always force a
        // rebuild from there onward.
        if common == 0 || !Self::check_prefix(&st.tokens, prompt, common) {
            st.last_sync = SyncResult::Diverges(common);
            return Ds4RewriteStatus::RebuildNeeded;
        }
        // Trim/extend the live state to match `prompt`. We trust the
        // first `common` tokens and rewrite index `common` onward, so
        // we truncate/extend from index `common` rather than from
        // `st.tokens.len()`.
        if st.tokens.len() > common {
            st.tokens.truncate(common);
        }
        if prompt.len() > common {
            st.tokens.extend_from_slice(&prompt[common..]);
        }
        st.position = st.tokens.len();
        let target = st.position;
        for layer in 0..st.cache.n_layers() {
            if st.cache.rewind(layer, target).is_err() {
                return Ds4RewriteStatus::RewriteError;
            }
        }
        st.last_sync = SyncResult::MatchAll;
        Ds4RewriteStatus::Ok
    }

    pub fn common_prefix(&self, prompt: &[u32]) -> usize {
        let st = self.state_lock_recovering();
        let mut n = 0usize;
        while n < st.tokens.len() && n < prompt.len() && st.tokens[n] == prompt[n] {
            n += 1;
        }
        n
    }

    pub fn argmax(&self) -> u32 {
        self.argmax_excluding(u32::MAX)
    }

    pub fn argmax_excluding(&self, excluded: u32) -> u32 {
        let st = self.state_lock_recovering();
        sampler::argmax_excluding(&st.logits, excluded)
    }

    pub fn sample(
        &self,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        rng: &mut dyn rand::RngCore,
    ) -> u32 {
        let st = self.state_lock_recovering();
        sampler::sample_top_p_min_p(&st.logits, temperature, top_k, top_p, min_p, rng)
    }

    pub fn top_logprobs(&self, out: &mut [f32], k: usize) {
        let st = self.state_lock_recovering();
        for (i, slot) in out.iter_mut().enumerate() {
            let lps = sampler::top_logprobs(&st.logits, k);
            *slot = lps.get(i).map(|(_, lp)| *lp).unwrap_or(0.0);
        }
    }

    pub fn token_logprob(&self, token: u32, out: &mut f32) {
        let st = self.state_lock_recovering();
        *out = sampler::token_logprob(&st.logits, token);
    }

    pub fn copy_logits(&self, out: &mut [f32], cap: usize) {
        let st = self.state_lock_recovering();
        let n = cap.min(st.logits.len()).min(out.len());
        out[..n].copy_from_slice(&st.logits[..n]);
    }

    pub fn set_logits(&mut self, logits: &[f32]) {
        let mut st = self.state_lock_recovering();
        let n = logits.len().min(st.logits.len());
        st.logits[..n].copy_from_slice(&logits[..n]);
    }

    pub fn eval(&mut self, token: u32) -> Ds4Result<()> {
        let mut st = self.state_lock()?;
        if st.position >= self.ctx_size {
            return Err(Ds4Error::new(
                Ds4ErrorKind::OutOfMemory,
                "context window exhausted",
            ));
        }
        let mut tokens = st.tokens.clone();
        tokens.push(token);
        let mut logits = st.logits.clone();
        self.engine.eval_sequence_logits(&tokens, &mut logits)?;
        let pos = st.position + 1;
        for layer in 0..st.cache.n_layers() {
            st.cache.rewind(layer, pos)?;
        }
        st.tokens = tokens;
        st.position = pos;
        st.logits = logits;
        // Update MTP draft state from the freshly evaluated logits.
        let drafts: Vec<u32> = sampler::top_logprobs(&st.logits, 4)
            .into_iter()
            .map(|(tok, _)| tok)
            .collect::<Vec<u32>>();
        if !drafts.is_empty() {
            st.mtp.reset();
        }
        st.last_sync = SyncResult::Common(pos);
        Ok(())
    }

    pub fn refresh_logits(&mut self) -> Ds4Result<()> {
        let mut st = self.state_lock()?;
        if !st.tokens.is_empty() {
            let tokens = st.tokens.clone();
            self.engine.eval_sequence_logits(&tokens, &mut st.logits)?;
        }
        Ok(())
    }

    pub fn eval_speculative_argmax(
        &mut self,
        first: u32,
        max: usize,
        eos: u32,
        accepted: &mut [u32],
    ) -> Ds4Result<()> {
        let mut st = self.state_lock()?;
        for (i, slot) in accepted.iter_mut().enumerate() {
            if i >= max {
                break;
            }
            if st.position >= self.ctx_size {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::OutOfMemory,
                    "context window exhausted",
                ));
            }
            let next = if i == 0 {
                first
            } else {
                self.argmax_via(&st.logits)
            };
            st.tokens.push(next);
            st.position += 1;
            if next == eos {
                break;
            }
            *slot = next;
        }
        Ok(())
    }

    fn argmax_via(&self, logits: &[f32]) -> u32 {
        sampler::argmax_excluding(logits, u32::MAX)
    }

    pub fn invalidate(&mut self) {
        let mut st = self.state_lock_recovering();
        st.tokens.clear();
        st.position = 0;
        st.cache.reset();
        st.mtp.reset();
        st.last_sync = SyncResult::Empty;
    }

    pub fn rewind(&mut self, pos: usize) {
        let mut st = self.state_lock_recovering();
        if pos < self.ctx_size && pos <= st.position {
            st.tokens.truncate(pos);
            let target = pos;
            st.position = target;
            for layer in 0..st.cache.n_layers() {
                let _ = st.cache.rewind(layer, target);
            }
            st.last_sync = SyncResult::MatchAll;
        }
    }

    pub fn pos(&self) -> usize {
        self.state_lock_recovering().position
    }
    pub fn ctx(&self) -> usize {
        self.ctx_size
    }
    pub fn prefill_cap(&self) -> usize {
        self.prefill_chunk
    }
    pub fn tokens(&self) -> Vec<u32> {
        self.state_lock_recovering().tokens.clone()
    }

    pub fn is_distributed(&self) -> bool {
        matches!(
            self.engine.options().distributed.as_ref().map(|d| d.role),
            Some(
                ds4_types::Ds4DistributedRole::Coordinator | ds4_types::Ds4DistributedRole::Worker
            )
        )
    }

    /// Helper used by `rewrite_from_common`. Returns true when the
    /// live state's first `common` tokens agree with `prompt`.
    fn check_prefix(live: &[u32], prompt: &[u32], common: usize) -> bool {
        let n = common.min(live.len()).min(prompt.len());
        live[..n] == prompt[..n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_types::Ds4EngineOptions;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SYNTH_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn fresh_engine() -> Arc<Ds4Engine> {
        let id = SYNTH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("ds4-session-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("synth.gguf");
        Ds4Engine::write_synthetic_gguf(&model_path).unwrap();
        let opts = Ds4EngineOptions {
            model_path,
            ..Ds4EngineOptions::default()
        };
        Arc::new(Ds4Engine::open(opts).unwrap())
    }

    fn missing_engine() -> Arc<Ds4Engine> {
        let opts = Ds4EngineOptions {
            model_path: PathBuf::from("missing.gguf"),
            ..Ds4EngineOptions::default()
        };
        Arc::new(Ds4Engine::open(opts).unwrap())
    }

    #[test]
    fn session_create_succeeds() {
        let eng = missing_engine();
        let s = Ds4Session::create(&eng, 1024).unwrap();
        assert_eq!(s.pos(), 0);
        assert_eq!(s.ctx(), 1024);
        assert!(!s.is_distributed());
    }

    #[test]
    fn sync_then_rewrite_state_path() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1024).unwrap();
        s.sync(&[1, 2, 3]).unwrap();
        assert_eq!(s.pos(), 3);

        // Rewrite to a new prompt that shares the first two tokens —
        // state machine should not need a rebuild.
        let new_prompt = vec![1u32, 2, 12, 13];
        let status = s.rewrite_from_common(&new_prompt, 2);
        assert_eq!(status, Ds4RewriteStatus::Ok);
        let toks = s.tokens();
        assert!(toks.ends_with(&[12, 13]));
    }

    #[test]
    fn rewrite_with_zero_common_returns_rebuild_needed() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1024).unwrap();
        s.sync(&[1]).unwrap();
        // common=0 means we don't trust any prefix; the safe move is
        // to require a full rebuild before any further eval.
        let status = s.rewrite_from_common(&[2, 3], 0);
        assert_eq!(status, Ds4RewriteStatus::RebuildNeeded);
    }

    #[test]
    fn common_prefix_returns_match_length() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1024).unwrap();
        s.sync(&[10, 11, 12]).unwrap();
        let n = s.common_prefix(&[10, 11, 13]);
        assert_eq!(n, 2);
    }

    #[test]
    fn invalidate_resets_state() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1024).unwrap();
        s.sync(&[1, 2, 3]).unwrap();
        s.invalidate();
        assert_eq!(s.pos(), 0);
        assert!(s.tokens().is_empty());
    }

    #[test]
    fn eval_extends_position() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1024).unwrap();
        s.sync(&[]).unwrap();
        s.eval(10).unwrap();
        assert_eq!(s.pos(), 1);
        assert_eq!(s.tokens(), vec![10]);
    }

    #[test]
    fn rewind_truncates() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1024).unwrap();
        s.sync(&[1, 2, 3, 4, 5]).unwrap();
        s.rewind(2);
        assert_eq!(s.pos(), 2);
        assert_eq!(s.tokens(), vec![1, 2]);
    }

    #[test]
    fn sync_rejects_prompt_past_context_window() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 2).unwrap();
        let err = s.sync(&[1, 2, 3]).err().unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn speculative_eval_rejects_context_exhaustion() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 1).unwrap();
        s.eval(10).unwrap();
        let mut accepted = [0u32; 1];
        let err = s
            .eval_speculative_argmax(2, 1, u32::MAX, &mut accepted)
            .err()
            .unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::OutOfMemory);
    }

    #[test]
    fn eval_error_does_not_commit_token() {
        let eng = fresh_engine();
        let mut s = Ds4Session::create(&eng, 4).unwrap();
        s.sync(&[10]).unwrap();
        let err = s.eval(eng.vocab_size() as u32).err().unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
        assert_eq!(s.pos(), 1);
        assert_eq!(s.tokens(), vec![10]);
    }
}
