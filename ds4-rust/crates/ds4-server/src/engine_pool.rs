// DS4 (DwarfStar) - engine pool.
//
// Wraps `ds4_core::engine::Ds4Engine` behind a serial queue. The first
// Rust runtime path is intentionally conservative: work is drained
// synchronously on submit, preserving one-engine-at-a-time semantics
// while avoiding a background thread before GPU contexts are wired.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use ds4_core::{chat::Ds4Role, engine::Ds4Engine, session::Ds4Session};
use ds4_types::{Ds4EngineOptions, Ds4Error, Ds4ErrorKind, Ds4Result, Ds4ThinkMode};
use rand::SeedableRng;

use crate::id::new_id;

pub const MAX_CONTEXT_TOKENS: usize = 262_144;
pub const MAX_GENERATION_TOKENS: usize = 4096;

/// A unit of work for the engine worker.
pub struct Job {
    pub id: String,
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to predict.
    pub max_tokens: usize,
    /// Sampling parameters.
    pub sampling: SamplingParams,
    /// Oneshot to send the result back on. Errors are sent as
    /// `Err(Ds4Error)` so the HTTP layer can map them to status codes.
    pub callback: oneshot::Sender<Ds4Result<JobResult>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            seed: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobResult {
    pub job_id: String,
    pub completion_tokens: Vec<u32>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub model: PathBuf,
    pub mtp: Option<PathBuf>,
    pub ctx: usize,
    pub prefill_chunk: usize,
    pub n_threads: usize,
}

impl PoolConfig {
    pub fn engine_options(&self) -> Ds4EngineOptions {
        Ds4EngineOptions {
            model_path: self.model.clone(),
            mtp_path: self.mtp.clone(),
            n_threads: self.n_threads,
            prefill_chunk: self.prefill_chunk,
            ..Ds4EngineOptions::default()
        }
    }
}

#[derive(Clone)]
pub struct EnginePool {
    inner: Arc<Inner>,
}

struct Inner {
    queue: Mutex<VecDeque<Job>>,
    enqueued: Mutex<u64>,
    status: Mutex<EngineStatus>,
    engine: Option<Arc<Ds4Engine>>,
    cfg: PoolConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    Ready,
    Unavailable,
}

impl EnginePool {
    /// Open the engine. If the backend cannot provide a loaded model,
    /// requests still resolve promptly with `NotImplemented`.
    pub fn open(cfg: PoolConfig) -> Ds4Result<Self> {
        let opts = cfg.engine_options();
        let engine_result = Ds4Engine::open(opts);
        let status = match engine_result.as_ref() {
            Ok(engine) if engine.model().is_some() => EngineStatus::Ready,
            Ok(_) => EngineStatus::Unavailable,
            Err(e) if e.kind == Ds4ErrorKind::NotImplemented => EngineStatus::Unavailable,
            Err(e) => return Err(e.clone()),
        };
        let engine = engine_result.ok().map(Arc::new);
        Ok(Self {
            inner: Arc::new(Inner {
                queue: Mutex::new(VecDeque::new()),
                enqueued: Mutex::new(0),
                status: Mutex::new(status),
                engine,
                cfg,
            }),
        })
    }

    pub fn config(&self) -> &PoolConfig {
        &self.inner.cfg
    }

    pub fn is_engine_ready(&self) -> bool {
        *self.inner.status.lock() == EngineStatus::Ready
    }

    pub fn is_engine_pending(&self) -> bool {
        false
    }

    pub fn is_engine_unavailable(&self) -> bool {
        *self.inner.status.lock() == EngineStatus::Unavailable
    }

    pub fn queue_len(&self) -> usize {
        self.inner.queue.lock().len()
    }

    pub fn enqueued_count(&self) -> u64 {
        *self.inner.enqueued.lock()
    }
    pub fn tokenize_text(&self, text: &str) -> Ds4Result<Vec<u32>> {
        let engine = self.engine()?;
        engine.tokenizer().tokenize(text)
    }

    pub fn detokenize_tokens(&self, tokens: &[u32]) -> Ds4Result<String> {
        let engine = self.engine()?;
        engine.tokenizer().detokenize(tokens)
    }

    pub fn encode_chat_messages(&self, messages: &[(&str, &str)]) -> Ds4Result<Vec<u32>> {
        let engine = self.engine()?;
        let mut tokens = engine.chat_begin();
        for (role, content) in messages {
            let role = if role.eq_ignore_ascii_case("system") {
                Ds4Role::System
            } else if role.eq_ignore_ascii_case("assistant") {
                Ds4Role::Assistant
            } else if role.eq_ignore_ascii_case("user") {
                Ds4Role::User
            } else {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!("unsupported chat role `{role}`"),
                ));
            };
            engine.chat().append_message(&mut tokens, role, content);
        }
        engine
            .chat()
            .append_assistant_prefix(&mut tokens, Ds4ThinkMode::None);
        Ok(tokens)
    }

    /// Enqueue a job and return a oneshot receiver for the result.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> (String, oneshot::Receiver<Ds4Result<JobResult>>) {
        let id = new_id();
        let (tx, rx) = oneshot::channel();
        let job = Job {
            id: id.clone(),
            prompt_tokens,
            max_tokens,
            sampling,
            callback: tx,
        };
        let mut q = self.inner.queue.lock();
        q.push_back(job);
        *self.inner.enqueued.lock() += 1;
        drop(q);
        drain_ready(&self.inner);
        (id, rx)
    }

    fn engine(&self) -> Ds4Result<&Arc<Ds4Engine>> {
        self.inner.engine.as_ref().ok_or_else(|| {
            Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "engine has no loaded model for tokenization",
            )
        })
    }
}

fn drain_ready(inner: &Arc<Inner>) {
    let mut q = inner.queue.lock();
    while let Some(job) = q.pop_front() {
        let result = match inner.engine.as_ref() {
            Some(engine) => run_job(engine, &inner.cfg, &job),
            None => Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "engine has no loaded model",
            )),
        };
        let _ = job.callback.send(result);
    }
}

fn run_job(engine: &Arc<Ds4Engine>, cfg: &PoolConfig, job: &Job) -> Ds4Result<JobResult> {
    if engine.model().is_none() {
        return Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "engine has no loaded model",
        ));
    }
    if job.max_tokens > MAX_GENERATION_TOKENS {
        return Err(Ds4Error::new(
            Ds4ErrorKind::InvalidArgument,
            format!(
                "max generation tokens {} exceeds server limit {}",
                job.max_tokens, MAX_GENERATION_TOKENS
            ),
        ));
    }
    let mut session = Ds4Session::create(engine, cfg.ctx)?;
    session.sync(&job.prompt_tokens)?;
    session.refresh_logits()?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(job.sampling.seed.unwrap_or(0x4453_3453));
    let eos = engine.tokenizer().eos_id();
    let mut completion_tokens = Vec::with_capacity(job.max_tokens.min(MAX_GENERATION_TOKENS));
    for _ in 0..job.max_tokens {
        if session.pos() >= session.ctx() {
            break;
        }
        let tok = session.sample(
            job.sampling.temperature,
            job.sampling.top_k,
            job.sampling.top_p,
            job.sampling.min_p,
            &mut rng,
        );
        if tok == eos {
            break;
        }
        completion_tokens.push(tok);
        session.eval(tok)?;
    }
    Ok(JobResult {
        job_id: job.id.clone(),
        completion_tokens: completion_tokens.clone(),
        usage: TokenUsage {
            prompt_tokens: job.prompt_tokens.len(),
            completion_tokens: completion_tokens.len(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SYNTH_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn cfg() -> PoolConfig {
        PoolConfig {
            model: PathBuf::from("/tmp/model.gguf"),
            mtp: None,
            ctx: 4096,
            prefill_chunk: 512,
            n_threads: 1,
        }
    }

    fn synth_cfg() -> PoolConfig {
        let id = SYNTH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ds4-server-engine-pool-test-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("synth.gguf");
        Ds4Engine::write_synthetic_gguf(&model).unwrap();
        PoolConfig {
            model,
            mtp: None,
            ctx: 64,
            prefill_chunk: 8,
            n_threads: 1,
        }
    }

    #[test]
    fn open_returns_pool() {
        let pool = EnginePool::open(synth_cfg()).unwrap();
        assert!(pool.is_engine_ready());
        assert!(!pool.is_engine_pending());
    }

    #[test]
    fn open_without_loaded_model_is_unavailable() {
        let pool = EnginePool::open(cfg()).unwrap();
        assert!(!pool.is_engine_ready());
        assert!(!pool.is_engine_pending());
        assert!(pool.is_engine_unavailable());
    }

    #[test]
    fn submit_returns_error_via_oneshot_without_loaded_model() {
        let pool = EnginePool::open(cfg()).unwrap();
        let (_id, rx) = pool.submit(vec![1, 2, 3], 16, SamplingParams::default());
        let result = rx.blocking_recv().unwrap();
        let err = result.err().unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::NotImplemented);
    }

    #[test]
    fn submit_increments_enqueue_counter() {
        let pool = EnginePool::open(cfg()).unwrap();
        let (_id, _rx) = pool.submit(vec![1], 4, SamplingParams::default());
        let (_id, _rx) = pool.submit(vec![2], 4, SamplingParams::default());
        assert_eq!(pool.enqueued_count(), 2);
        assert_eq!(pool.queue_len(), 0);
    }

    #[test]
    fn tokenize_text_uses_model_tokenizer_metadata() {
        let pool = EnginePool::open(synth_cfg()).unwrap();
        let tokens = pool.tokenize_text("hi").unwrap();
        assert_eq!(tokens, vec![10]);
    }

    #[test]
    fn encode_chat_messages_uses_template_and_model_tokenizer() {
        let pool = EnginePool::open(synth_cfg()).unwrap();
        let tokens = pool.encode_chat_messages(&[("user", "hi")]).unwrap();
        assert_eq!(tokens[0], 1);
        assert!(tokens.contains(&3));
        assert!(tokens.contains(&4));
        assert!(tokens.contains(&10));
    }

    #[test]
    fn encode_chat_messages_rejects_unknown_roles() {
        let pool = EnginePool::open(synth_cfg()).unwrap();
        let err = pool
            .encode_chat_messages(&[("systme", "hi")])
            .err()
            .unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn submit_rejects_generation_budget_over_limit() {
        let pool = EnginePool::open(synth_cfg()).unwrap();
        let (_id, rx) = pool.submit(
            vec![1, 3, 10, 4],
            MAX_GENERATION_TOKENS + 1,
            SamplingParams::default(),
        );
        let err = rx.blocking_recv().unwrap().err().unwrap();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn submit_with_synthetic_model_returns_tokens() {
        let pool = EnginePool::open(synth_cfg()).unwrap();
        let (_id, rx) = pool.submit(
            vec![1, 3, 4],
            4,
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
        );
        let result = rx.blocking_recv().unwrap().unwrap();
        assert!(!result.completion_tokens.is_empty());
        assert_eq!(result.usage.prompt_tokens, 3);
        assert_eq!(
            result.usage.completion_tokens,
            result.completion_tokens.len()
        );
    }
}
