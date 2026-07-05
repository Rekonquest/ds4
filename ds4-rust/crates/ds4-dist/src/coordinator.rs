// DS4 (DwarfStar) — coordinator state machine.
//
// The coordinator owns worker routing and dispatches layer slices
// to connected workers through WORK/RESULT frames. The worker side
// executes the loaded model via the shared `BackendModel` hooks.

use crate::msg;
use crate::protocol::{
    decode_payload, frame_result, frame_work, Decoded, Frame, Hello, ResultMsg, WireError, Work,
};
use crate::{DEFAULT_ACTIVATION_BITS, SNAPSHOT_CHUNK_BYTES};
use ds4_tensor::Tensor;
use ds4_types::{
    Ds4DistributedOptions, Ds4DistributedRole, Ds4Error, Ds4ErrorKind, Ds4LayerSlice, Ds4Result,
};
use std::collections::HashMap;
use std::io::{Read, Write};

/// One connected worker. We hold the stream so the coordinator can
/// write WORK frames to it. `layer_start..layer_end` is the slice
/// the worker has been assigned.
pub struct WorkerConn {
    pub worker_id: u32,
    pub stream: TcpStream,
    pub layer_start: usize,
    pub layer_end: usize,
    pub has_output: bool,
    pub activation_bits: u8,
}

/// Tiny `TcpStream` shim. The real Rust standard library has
/// `std::net::TcpStream`, but we want this crate to compile on
/// platforms where the standard library's TCP stack is unavailable
/// (e.g. embedded test runners). The shim is byte-compatible with
/// `std::io::{Read, Write}` so the protocol layer doesn't care.
///
/// The wrapper accepts any `Read + Write` stream, including an OS
/// TCP stream or an in-memory duplex used by tests.
///
/// The inner stream is stored as `Box<dyn Any + Send + 'static>`.
/// `Read`/`Write` are dispatched through per-type helpers (currently
/// just `Cursor<Vec<u8>>`, which covers all in-memory test pipes).
/// Tests downcast through `Any::downcast_mut` to recover the concrete
/// type for assertions.
pub struct TcpStream {
    pub(crate) inner: Box<dyn ReadWrite + Send + 'static>,
    peer: String,
}

impl TcpStream {
    /// Wrap a concrete `Read+Write` pair as a stream (used by tests).
    pub fn new<T>(rw: T, peer: impl Into<String>) -> Self
    where
        T: Read + Write + Send + 'static,
    {
        Self {
            inner: Box::new(rw),
            peer: peer.into(),
        }
    }

    /// Wrap a boxed stream. Used by callers that already hold a
    /// `Box<dyn ReadWrite + Send>`.
    pub fn from_box(b: Box<dyn ReadWrite + Send + 'static>, peer: impl Into<String>) -> Self {
        Self {
            inner: b,
            peer: peer.into(),
        }
    }

    pub fn peer(&self) -> &str {
        &self.peer
    }

    /// Downcast the inner stream to a concrete `T`. Used by tests
    /// that drive the worker/coordinator through in-memory pipes.
    /// Returns `None` if the inner type doesn't match.
    #[cfg(test)]
    pub(crate) fn downcast_inner<T: 'static>(&mut self) -> Option<&mut T> {
        self.inner.as_mut().as_any_mut().downcast_mut::<T>()
    }
}

/// Trait alias for objects that are both `Read` and `Write`. The
/// `as_any_mut` hook preserves the old test downcast behavior while
/// allowing any full-duplex in-memory or OS stream to be wrapped.
pub trait ReadWrite: std::any::Any + Read + Write {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: std::any::Any + Read + Write> ReadWrite for T {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Read for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.inner.read_exact(buf)
    }
}

impl Write for TcpStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.inner.write_all(buf)
    }
}
/// Coordinator state.
pub struct Coordinator {
    pub workers: HashMap<u32, WorkerConn>,
    pub layers: Vec<LayerSlice>,
    pub activation_bits: u8,
    pub snapshot_chunk_bytes: usize,
    pub options: Ds4DistributedOptions,
    /// Monotonic request id; included in every WORK and used by the
    /// coordinator to match RESULTS back to the originating request.
    next_req_id: u64,
}

/// Description of a layer slice owned by a worker. The coordinator
/// keeps one of these per connected worker so dispatch can pick the
/// right peer for any given layer range.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSlice {
    pub layer_start: usize,
    pub layer_end: usize,
    pub has_output: bool,
}

impl LayerSlice {
    pub fn covers(&self, layer: usize) -> bool {
        layer >= self.layer_start && layer < self.layer_end
    }

    pub fn from_ds4(s: &Ds4LayerSlice) -> Self {
        Self {
            layer_start: s.start,
            layer_end: s.end,
            has_output: s.has_output,
        }
    }
}

impl Coordinator {
    /// Build a coordinator from the engine options. The
    /// `activation_bits` and `snapshot_chunk_bytes` are pulled from
    /// the options if set, otherwise the crate-level defaults.
    pub fn new(opts: Ds4DistributedOptions) -> Self {
        assert_eq!(
            opts.role,
            Ds4DistributedRole::Coordinator,
            "Coordinator::new called with role {:?}",
            opts.role,
        );
        let activation_bits = if opts.activation_bits == 0 {
            DEFAULT_ACTIVATION_BITS
        } else {
            opts.activation_bits
        };
        Self {
            workers: HashMap::new(),
            layers: Vec::new(),
            activation_bits,
            snapshot_chunk_bytes: SNAPSHOT_CHUNK_BYTES,
            options: opts,
            next_req_id: 1,
        }
    }

    /// Allocate the next request id. Exposed so external dispatchers
    /// can correlate logs.
    pub fn next_req_id(&mut self) -> u64 {
        let id = self.next_req_id;
        self.next_req_id = self.next_req_id.checked_add(1).unwrap_or(1);
        id
    }

    /// Register a new worker connection. Reads the HELLO frame,
    /// validates the wire version, and stores the connection in the
    /// workers map under `worker_id`.
    ///
    /// Returns `Err(WireError::BadVersion)` if the worker's wire
    /// version doesn't match ours, and `Err(WireError::BadMagic)`
    /// if the magic field is wrong.
    pub fn accept_worker(
        &mut self,
        worker_id: u32,
        mut stream: TcpStream,
    ) -> Result<Hello, WireError> {
        let frame = Frame::read_from(&mut stream)?;
        if frame.msg_type != msg::HELLO {
            return Err(WireError::BadMsgType(frame.msg_type));
        }
        let hello = match decode_payload(frame.msg_type, &frame.payload)
            .map_err(|e| WireError::Domain(e.message.clone()))?
        {
            Decoded::Hello(h) => h,
            _ => return Err(WireError::BadMsgType(frame.msg_type)),
        };
        if hello.wire_version != crate::protocol::WIRE_VERSION {
            return Err(WireError::BadVersion(hello.wire_version));
        }

        // Wire up the worker with the layer slice it should run.
        // For v1 we just use a single slice covering [0, n_layers).
        let n_layers = hello.n_layers as usize;
        let slice = LayerSlice {
            layer_start: 0,
            layer_end: n_layers,
            has_output: true, // v1: every worker holds the whole model
        };
        self.layers.push(slice.clone());

        self.workers.insert(
            worker_id,
            WorkerConn {
                worker_id,
                stream,
                layer_start: slice.layer_start,
                layer_end: slice.layer_end,
                has_output: slice.has_output,
                activation_bits: hello.activation_bits,
            },
        );
        Ok(hello)
    }

    /// Find a worker that owns the layer slice covering
    /// `[layer_start, layer_end)`. Returns the worker id.
    pub fn route_layer_slice(&self, layer_start: usize, layer_end: usize) -> Option<u32> {
        for w in self.workers.values() {
            if w.layer_start <= layer_start && w.layer_end >= layer_end {
                return Some(w.worker_id);
            }
        }
        None
    }

    /// Dispatch a layer slice by routing to a worker, writing a WORK
    /// frame, reading its RESULT frame, and returning the produced
    /// hidden-state tensor.
    pub fn dispatch_layer_slice(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        layer_start: usize,
        layer_end: usize,
        input_hc: &Tensor,
    ) -> Ds4Result<Tensor> {
        let worker_id = self
            .route_layer_slice(layer_start, layer_end)
            .ok_or_else(|| {
                Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!("no worker covers layer slice {layer_start}..{layer_end}"),
                )
            })?;
        let work = Work {
            req_id: self.next_req_id(),
            tokens: tokens.to_vec(),
            pos0,
            layer_start,
            layer_end,
            input_hc: input_hc.clone(),
        };
        let result = self.send_work_and_await_result(worker_id, &work)?;
        Ok(result.output_hc)
    }

    /// Real wire-path: build a WORK frame, write it to the chosen
    /// worker, read the RESULT frame. Returns the output hidden
    /// state tensor on success.
    pub fn send_work_and_await_result(
        &mut self,
        worker_id: u32,
        work: &Work,
    ) -> Ds4Result<ResultMsg> {
        let frame = frame_work(work)?;
        let worker = self.workers.get_mut(&worker_id).ok_or_else(|| {
            Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("no worker {worker_id}"),
            )
        })?;
        frame
            .write_to(&mut worker.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("write WORK: {e}")))?;

        let resp = Frame::read_from(&mut worker.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("read RESULT: {e}")))?;
        if resp.msg_type != msg::RESULT {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("expected RESULT, got msg_type={}", resp.msg_type),
            ));
        }
        match decode_payload(resp.msg_type, &resp.payload).map_err(|e| {
            Ds4Error::new(
                Ds4ErrorKind::Backend,
                format!("decode RESULT: {}", e.message),
            )
        })? {
            Decoded::Result(r) => {
                if r.req_id != work.req_id {
                    return Err(Ds4Error::new(
                        Ds4ErrorKind::Backend,
                        format!(
                            "RESULT req_id mismatch: sent {}, got {}",
                            work.req_id, r.req_id
                        ),
                    ));
                }
                if !r.ok {
                    let msg = r.error.unwrap_or_else(|| "unknown error".to_string());
                    return Err(Ds4Error::new(Ds4ErrorKind::Backend, msg));
                }
                Ok(r)
            }
            _ => Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                "expected RESULT payload, got something else",
            )),
        }
    }

    /// Write a RESULT frame to a coordinator-side worker stream.
    pub fn write_result(&mut self, worker_id: u32, result: &ResultMsg) -> Ds4Result<()> {
        let frame = frame_result(result)?;
        let worker = self.workers.get_mut(&worker_id).ok_or_else(|| {
            Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("no worker {worker_id}"),
            )
        })?;
        frame
            .write_to(&mut worker.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("write RESULT: {e}")))?;
        Ok(())
    }
    /// Number of registered workers.
    pub fn n_workers(&self) -> usize {
        self.workers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Frame, Hello};
    use ds4_tensor::{Shape, Tensor};
    use std::io::Cursor;

    fn opts() -> Ds4DistributedOptions {
        Ds4DistributedOptions {
            role: Ds4DistributedRole::Coordinator,
            activation_bits: 16,
            ..Ds4DistributedOptions::default()
        }
    }

    #[derive(Clone)]
    struct DuplexEnd {
        rx: std::sync::Arc<(
            std::sync::Mutex<std::collections::VecDeque<u8>>,
            std::sync::Condvar,
        )>,
        tx: std::sync::Arc<(
            std::sync::Mutex<std::collections::VecDeque<u8>>,
            std::sync::Condvar,
        )>,
    }

    impl std::io::Read for DuplexEnd {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            let (lock, cvar) = &*self.rx;
            let mut queue = lock.lock().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while queue.is_empty() {
                let now = std::time::Instant::now();
                if deadline <= now {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "duplex read timed out",
                    ));
                }
                let wait = deadline.saturating_duration_since(now);
                let (guard, timeout) = cvar.wait_timeout(queue, wait).unwrap();
                queue = guard;
                if timeout.timed_out() && queue.is_empty() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "duplex read timed out",
                    ));
                }
            }
            let n = buf.len().min(queue.len());
            for slot in &mut buf[..n] {
                *slot = queue.pop_front().unwrap();
            }
            Ok(n)
        }
    }

    impl std::io::Write for DuplexEnd {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let (lock, cvar) = &*self.tx;
            let mut queue = lock.lock().unwrap();
            queue.extend(buf.iter().copied());
            cvar.notify_all();
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn duplex_pair() -> (TcpStream, TcpStream) {
        let a_rx = std::sync::Arc::new((
            std::sync::Mutex::new(std::collections::VecDeque::new()),
            std::sync::Condvar::new(),
        ));
        let b_rx = std::sync::Arc::new((
            std::sync::Mutex::new(std::collections::VecDeque::new()),
            std::sync::Condvar::new(),
        ));
        let a = DuplexEnd {
            rx: a_rx.clone(),
            tx: b_rx.clone(),
        };
        let b = DuplexEnd { rx: b_rx, tx: a_rx };
        (TcpStream::new(a, "coord"), TcpStream::new(b, "worker"))
    }

    struct ProbeModel;
    impl ds4_types::BackendModel for ProbeModel {
        fn quant_kind(&self) -> ds4_types::Ds4QuantKind {
            ds4_types::Ds4QuantKind::F32
        }

        fn eval_layer_slice(
            &self,
            _tokens: &[u32],
            _pos0: usize,
            _layer_start: usize,
            _layer_end: usize,
            input_hc: &[f32],
            output_hc: &mut [f32],
        ) -> Ds4Result<()> {
            for (out, input) in output_hc.iter_mut().zip(input_hc.iter()) {
                *out = *input * 2.0;
            }
            Ok(())
        }

        fn eval_output_head_from_hc(
            &self,
            hidden_hc: &[f32],
            _n_tokens: usize,
            logits: &mut [f32],
        ) -> Ds4Result<()> {
            for (idx, slot) in logits.iter_mut().enumerate() {
                *slot = hidden_hc.get(idx % hidden_hc.len()).copied().unwrap_or(0.0);
            }
            Ok(())
        }
    }
    fn make_worker_stream() -> TcpStream {
        // Trivial test stream: backed by a Cursor<Vec<u8>>.
        TcpStream::new(Cursor::new(Vec::<u8>::new()), "test")
    }

    #[test]
    fn coordinator_rejects_bad_magic() {
        let mut coord = Coordinator::new(opts());
        let mut stream = make_worker_stream();
        // Write a bad-magic frame into the stream's backing buffer.
        let inner = stream.downcast_inner::<Cursor<Vec<u8>>>().unwrap();
        inner.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        inner.write_all(&msg::HELLO.to_le_bytes()).unwrap();
        inner.write_all(&0u32.to_le_bytes()).unwrap();
        inner.set_position(0);
        let err = coord.accept_worker(1, stream).unwrap_err();
        assert!(matches!(err, WireError::BadMagic { .. }));
    }

    #[test]
    fn coordinator_rejects_bad_version() {
        let mut coord = Coordinator::new(opts());
        let mut stream = make_worker_stream();
        let inner = stream.downcast_inner::<Cursor<Vec<u8>>>().unwrap();
        // Send a HELLO with wire_version=999 (wrong).
        let mut hello = Hello::worker(32, 4096);
        hello.wire_version = 999;
        let payload = hello.encode();
        let frame = Frame::new(msg::HELLO, payload).unwrap();
        let bytes = frame.encode();
        inner.write_all(&bytes).unwrap();
        inner.set_position(0);
        let err = coord.accept_worker(1, stream).unwrap_err();
        assert!(matches!(err, WireError::BadVersion(999)));
    }

    #[test]
    fn coordinator_accepts_good_hello() {
        let mut coord = Coordinator::new(opts());
        let mut stream = make_worker_stream();
        let inner = stream.downcast_inner::<Cursor<Vec<u8>>>().unwrap();
        let hello = Hello::worker(32, 4096);
        let frame = Frame::new(msg::HELLO, hello.encode()).unwrap();
        let bytes = frame.encode();
        inner.write_all(&bytes).unwrap();
        inner.set_position(0);
        let got = coord.accept_worker(1, stream).unwrap();
        assert_eq!(got.wire_version, crate::protocol::WIRE_VERSION);
        assert_eq!(got.n_layers, 32);
        assert_eq!(coord.n_workers(), 1);
    }

    #[test]
    fn dispatch_layer_slice_round_trips_through_worker() {
        let (coord_stream, worker_stream) = duplex_pair();
        let worker_opts = Ds4DistributedOptions {
            role: Ds4DistributedRole::Worker,
            layers: Ds4LayerSlice {
                start: 0,
                end: 32,
                has_output: true,
                set: true,
            },
            ..Ds4DistributedOptions::default()
        };
        let mut worker = crate::worker::Worker::new(worker_opts, worker_stream, 32, 4);
        worker.send_hello().unwrap();

        let mut coord = Coordinator::new(opts());
        coord.accept_worker(3, coord_stream).unwrap();
        let handle = std::thread::spawn(move || {
            let model = ProbeModel;
            worker.serve_one(&model).unwrap();
        });

        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], Shape::new([1, 4]));
        let out = coord.dispatch_layer_slice(&[1], 0, 0, 32, &input).unwrap();
        handle.join().unwrap();
        assert_eq!(out.as_f32(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn route_layer_slice_finds_correct_worker() {
        let mut coord = Coordinator::new(opts());
        let mut s1 = make_worker_stream();
        let inner = s1.downcast_inner::<Cursor<Vec<u8>>>().unwrap();
        let hello = Hello::worker(64, 4096);
        let frame = Frame::new(msg::HELLO, hello.encode()).unwrap();
        inner.write_all(&frame.encode()).unwrap();
        inner.set_position(0);
        coord.accept_worker(7, s1).unwrap();
        assert_eq!(coord.route_layer_slice(0, 64), Some(7));
        assert_eq!(coord.route_layer_slice(0, 32), Some(7));
        assert_eq!(coord.route_layer_slice(64, 128), None);
    }

    #[test]
    fn next_req_id_is_monotonic() {
        let mut coord = Coordinator::new(opts());
        let a = coord.next_req_id();
        let b = coord.next_req_id();
        let c = coord.next_req_id();
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn next_req_id_wraps_without_zero() {
        let mut coord = Coordinator::new(opts());
        coord.next_req_id = u64::MAX;
        assert_eq!(coord.next_req_id(), u64::MAX);
        assert_eq!(coord.next_req_id(), 1);
    }

    #[test]
    fn worker_default_activation_bits_matches_opts() {
        let mut coord = Coordinator::new(opts());
        let mut s1 = make_worker_stream();
        let inner = s1.downcast_inner::<Cursor<Vec<u8>>>().unwrap();
        let hello = Hello::worker(8, 4096);
        let frame = Frame::new(msg::HELLO, hello.encode()).unwrap();
        inner.write_all(&frame.encode()).unwrap();
        inner.set_position(0);
        coord.accept_worker(1, s1).unwrap();
        let w = &coord.workers[&1];
        // The worker HELLO sets DEFAULT_ACTIVATION_BITS_WIRE (32);
        // opts.activation_bits (16) is used by the *coordinator's*
        // outbound frames, not the inbound HELLO. Just confirm we
        // remembered the worker.
        assert_eq!(
            w.activation_bits,
            crate::protocol::DEFAULT_ACTIVATION_BITS_WIRE
        );
    }

    #[test]
    fn layer_slice_covers() {
        let s = LayerSlice {
            layer_start: 4,
            layer_end: 8,
            has_output: true,
        };
        assert!(s.covers(4));
        assert!(s.covers(7));
        assert!(!s.covers(3));
        assert!(!s.covers(8));
    }

    #[test]
    fn layer_slice_from_ds4() {
        let ds = Ds4LayerSlice {
            start: 8,
            end: 16,
            has_output: true,
            set: true,
        };
        let s = LayerSlice::from_ds4(&ds);
        assert_eq!(s.layer_start, 8);
        assert_eq!(s.layer_end, 16);
        assert!(s.has_output);
    }

    #[test]
    fn coordinator_new_asserts_role() {
        let bad = Ds4DistributedOptions {
            role: Ds4DistributedRole::Worker,
            ..Ds4DistributedOptions::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Coordinator::new(bad);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn send_work_returns_error_for_unknown_worker() {
        let mut coord = Coordinator::new(opts());
        let work = Work {
            req_id: 1,
            tokens: vec![],
            pos0: 0,
            layer_start: 0,
            layer_end: 0,
            input_hc: Tensor::from_f32(&[0.0; 4], Shape::new([4])),
        };
        let err = coord.send_work_and_await_result(42, &work).unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }
}
