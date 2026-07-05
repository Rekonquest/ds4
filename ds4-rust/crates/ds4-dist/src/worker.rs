// DS4 (DwarfStar) — worker state machine.
//
// A worker connects to a coordinator, sends its HELLO, then enters
// a serve loop reading WORK frames and dispatching them to the
// loaded model through the shared `BackendModel` layer-slice and
// output-head hooks.

use crate::coordinator::TcpStream;
use crate::msg;
use crate::protocol::{
    decode_payload, frame_hello, frame_result, Decoded, Frame, Hello, ResultMsg, Work,
};
use ds4_tensor::Tensor;
use ds4_types::BackendModel;
use ds4_types::{Ds4DistributedOptions, Ds4DistributedRole, Ds4Error, Ds4ErrorKind, Ds4Result};

/// One side of a worker connection. The worker reads WORK frames
/// from the coordinator's stream, dispatches them to the loaded
/// model, and writes RESULT frames back.
pub struct Worker {
    pub stream: TcpStream,
    pub layer_start: usize,
    pub layer_end: usize,
    pub has_output: bool,
    pub activation_bits: u8,
    pub n_layers: u32,
    pub head_elements: u32,
}

impl Worker {
    /// Build a worker from the engine options + a connected stream.
    /// The stream must already be connected to the coordinator.
    pub fn new(
        opts: Ds4DistributedOptions,
        stream: TcpStream,
        n_layers: u32,
        head_elements: u32,
    ) -> Self {
        assert_eq!(
            opts.role,
            Ds4DistributedRole::Worker,
            "Worker::new called with role {:?}",
            opts.role,
        );
        let slice = opts.layers.clone();
        Self {
            stream,
            layer_start: slice.start,
            layer_end: slice.end,
            has_output: slice.has_output,
            activation_bits: if opts.activation_bits == 0 {
                32
            } else {
                opts.activation_bits
            },
            n_layers,
            head_elements,
        }
    }

    /// Send the HELLO frame to the coordinator.
    pub fn send_hello(&mut self) -> Ds4Result<()> {
        let hello = Hello::worker(self.n_layers, self.head_elements);
        let frame = frame_hello(&hello)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("frame_hello: {}", e.message)))?;
        frame
            .write_to(&mut self.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("write HELLO: {e}")))?;
        Ok(())
    }

    /// Send a HELLO frame with the given role tag (used by tests
    /// that need to inject bad versions/magic).
    pub fn send_hello_raw(&mut self, hello: &Hello) -> Ds4Result<()> {
        let frame = frame_hello(hello)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("frame_hello: {}", e.message)))?;
        frame
            .write_to(&mut self.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("write HELLO: {e}")))?;
        Ok(())
    }

    /// Read one WORK frame from the coordinator.
    pub fn read_work(&mut self) -> Ds4Result<Work> {
        let frame = Frame::read_from(&mut self.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("read frame: {e}")))?;
        if frame.msg_type != msg::WORK {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("expected WORK, got msg_type={}", frame.msg_type),
            ));
        }
        match decode_payload(frame.msg_type, &frame.payload).map_err(|e| {
            Ds4Error::new(Ds4ErrorKind::Backend, format!("decode WORK: {}", e.message))
        })? {
            Decoded::Work(w) => Ok(w),
            _ => Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                "expected WORK payload, got something else",
            )),
        }
    }

    /// Send a RESULT frame back to the coordinator.
    pub fn send_result(&mut self, result: &ResultMsg) -> Ds4Result<()> {
        let frame = frame_result(result)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("frame_result: {}", e.message)))?;
        frame
            .write_to(&mut self.stream)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("write RESULT: {e}")))?;
        Ok(())
    }

    /// Dispatch a WORK to the loaded model and return the hidden-state
    /// slice plus optional logits.
    pub fn dispatch<M: BackendModel>(&self, model: &M, work: &Work) -> Ds4Result<(Tensor, Tensor)> {
        if work.layer_start < self.layer_start || self.layer_end < work.layer_end {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "worker slice {}..{} does not cover requested {}..{}",
                    self.layer_start, self.layer_end, work.layer_start, work.layer_end
                ),
            ));
        }
        if work.input_hc.dtype != ds4_tensor::DType::F32 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                "worker input_hc must be F32",
            ));
        }
        let input = work.input_hc.as_f32();
        let mut output = vec![0.0f32; input.len()];
        model.eval_layer_slice(
            &work.tokens,
            work.pos0,
            work.layer_start,
            work.layer_end,
            &input,
            &mut output,
        )?;
        let output_hc = Tensor::from_f32(&output, work.input_hc.shape.clone());

        let output_logits = if self.has_output {
            let n_tokens = tensor_token_count(&work.input_hc);
            let head_elements = self.head_elements as usize;
            if n_tokens != 0 && head_elements == 0 {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    "worker has output head but head_elements is zero",
                ));
            }
            let mut logits = vec![0.0f32; n_tokens * head_elements];
            model.eval_output_head_from_hc(&output, n_tokens, &mut logits)?;
            Tensor::from_f32(&logits, ds4_tensor::Shape::new([n_tokens, head_elements]))
        } else {
            Tensor::from_f32(&[], ds4_tensor::Shape::new([0]))
        };
        Ok((output_hc, output_logits))
    }

    /// Serve one request: read WORK, dispatch, send RESULT.
    /// Any dispatch error is also reported to the coordinator in a
    /// RESULT frame so request ids never hang without a response.
    pub fn serve_one<M: BackendModel>(&mut self, model: &M) -> Ds4Result<()> {
        let work = self.read_work()?;
        match self.dispatch(model, &work) {
            Ok((out_hc, out_logits)) => {
                let result = ResultMsg {
                    req_id: work.req_id,
                    ok: true,
                    error: None,
                    output_hc: out_hc,
                    output_logits: out_logits,
                };
                self.send_result(&result)
            }
            Err(e) => {
                let result = ResultMsg {
                    req_id: work.req_id,
                    ok: false,
                    error: Some(e.message.clone()),
                    output_hc: Tensor::from_f32(&[], ds4_tensor::Shape::new([0])),
                    output_logits: Tensor::from_f32(&[], ds4_tensor::Shape::new([0])),
                };
                let _ = self.send_result(&result);
                Err(e)
            }
        }
    }
    /// Run the worker loop forever, or until the stream or model
    /// returns an error.
    pub fn serve<M: BackendModel>(mut self, model: &M) -> Ds4Result<()> {
        loop {
            self.serve_one(model)?;
        }
    }
}

fn tensor_token_count(t: &Tensor) -> usize {
    if t.shape.numel() == 0 {
        return 0;
    }
    match t.shape.dims() {
        [] => 0,
        [_] => 1,
        [n, ..] => *n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Frame;
    use ds4_tensor::{Shape, Tensor};
    use std::io::{Cursor, Write};

    fn opts() -> Ds4DistributedOptions {
        Ds4DistributedOptions {
            role: Ds4DistributedRole::Worker,
            layers: ds4_types::Ds4LayerSlice {
                start: 0,
                end: 32,
                has_output: true,
                set: true,
            },
            ..Ds4DistributedOptions::default()
        }
    }

    fn hello_bad_version() -> Hello {
        let mut h = Hello::worker(32, 4096);
        h.wire_version = 999;
        h
    }

    /// Build a `Worker` whose stream is backed by a fresh empty
    /// in-memory cursor. Tests can write into the cursor directly
    /// to drive the worker.
    fn make_worker() -> Worker {
        Worker::new(
            opts(),
            TcpStream::new(Cursor::new(Vec::<u8>::new()), "test"),
            32,
            4096,
        )
    }

    struct ProbeModel;
    impl BackendModel for ProbeModel {
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
                *out = *input + 1.0;
            }
            Ok(())
        }

        fn eval_output_head_from_hc(
            &self,
            hidden_hc: &[f32],
            n_tokens: usize,
            logits: &mut [f32],
        ) -> Ds4Result<()> {
            let per_token = logits.len().checked_div(n_tokens).unwrap_or(0);
            for token_idx in 0..n_tokens {
                let base = hidden_hc.get(token_idx).copied().unwrap_or(0.0);
                for slot in &mut logits[token_idx * per_token..(token_idx + 1) * per_token] {
                    *slot = base;
                }
            }
            Ok(())
        }
    }

    /// Helper: write a HELLO frame into the worker's backing
    /// buffer so the next `Frame::read_from` returns it.
    fn seed_worker_with_hello(w: &mut Worker, hello: &Hello) {
        let frame = Frame::new(msg::HELLO, hello.encode()).unwrap();
        let bytes = frame.encode();
        // Access the inner cursor through the TcpStream shim.
        w.stream.write_all(&bytes).unwrap();
        // Reset position so read_from reads from the start.
        if let Some(cur) = w.stream.downcast_inner::<Cursor<Vec<u8>>>() {
            cur.set_position(0);
        }
    }

    /// Helper: read all bytes written into the worker's backing
    /// buffer (from position 0).
    fn drain_worker(w: &mut Worker) -> Vec<u8> {
        if let Some(cur) = w.stream.downcast_inner::<Cursor<Vec<u8>>>() {
            cur.set_position(0);
            let buf = cur.get_ref();
            let out = buf.clone();
            cur.get_mut().clear();
            cur.set_position(0);
            out
        } else {
            Vec::new()
        }
    }

    #[test]
    fn worker_sends_hello_with_correct_magic() {
        let mut w = make_worker();
        w.send_hello().unwrap();
        let bytes = drain_worker(&mut w);
        assert!(bytes.len() >= 12);
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let mtype = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(magic, 0x4453_3444);
        assert_eq!(mtype, msg::HELLO);
    }

    #[test]
    fn worker_hello_carries_injected_bad_version() {
        // `send_hello_raw` lets the test inject a HELLO with a bad
        // wire_version. The coordinator's `accept_worker` rejects
        // it via `WireError::BadVersion` (covered in coordinator.rs
        // tests). Here we just confirm the bad version survives
        // serialization.
        let mut w = make_worker();
        let bad = hello_bad_version();
        w.send_hello_raw(&bad).unwrap();
        let bytes = drain_worker(&mut w);
        let mut cur = std::io::Cursor::new(bytes);
        let frame = Frame::read_from(&mut cur).unwrap();
        let h = Hello::decode(&frame.payload).unwrap();
        assert_eq!(h.wire_version, 999);
    }

    #[test]
    fn worker_dispatch_runs_model_and_output_head() {
        let w = make_worker();
        let work = Work {
            req_id: 1,
            tokens: vec![9],
            pos0: 0,
            layer_start: 0,
            layer_end: 32,
            input_hc: Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], Shape::new([1, 4])),
        };
        let model = ProbeModel;
        let (out_hc, out_logits) = w.dispatch(&model, &work).unwrap();
        assert_eq!(out_hc.as_f32(), vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(out_logits.shape.dims(), &[1, 4096]);
        assert!(out_logits.as_f32().iter().all(|v| *v == 2.0));
    }

    #[test]
    fn worker_serve_one_sends_success_result() {
        let mut w = make_worker();
        let work = Work {
            req_id: 7,
            tokens: vec![1],
            pos0: 0,
            layer_start: 0,
            layer_end: 32,
            input_hc: Tensor::from_f32(&[1.0; 4], Shape::new([1, 4])),
        };
        let frame = Frame::new(msg::WORK, work.encode()).unwrap();
        w.stream.write_all(&frame.encode()).unwrap();
        if let Some(cur) = w.stream.downcast_inner::<Cursor<Vec<u8>>>() {
            cur.set_position(0);
        }
        let model = ProbeModel;
        w.serve_one(&model).unwrap();
        let bytes = drain_worker(&mut w);
        let mut cur = std::io::Cursor::new(bytes);
        let _work_frame = Frame::read_from(&mut cur).unwrap();
        let result_frame = Frame::read_from(&mut cur).unwrap();
        assert_eq!(result_frame.msg_type, msg::RESULT);
        let result = ResultMsg::decode(&result_frame.payload).unwrap();
        assert!(result.ok);
        assert_eq!(result.req_id, 7);
        assert_eq!(result.output_hc.as_f32(), vec![2.0; 4]);
    }
    #[test]
    fn worker_send_result_writes_frame() {
        let mut w = make_worker();
        let result = ResultMsg {
            req_id: 5,
            ok: true,
            error: None,
            output_hc: Tensor::from_f32(&[1.0, 2.0], Shape::new([2])),
            output_logits: Tensor::from_f32(&[3.0, 4.0], Shape::new([2])),
        };
        w.send_result(&result).unwrap();
        let bytes = drain_worker(&mut w);
        let mut cur = std::io::Cursor::new(bytes);
        let frame = Frame::read_from(&mut cur).unwrap();
        assert_eq!(frame.magic, 0x4453_3444);
        assert_eq!(frame.msg_type, msg::RESULT);
        let r2 = ResultMsg::decode(&frame.payload).unwrap();
        assert_eq!(r2.req_id, 5);
        assert!(r2.ok);
        assert_eq!(r2.output_hc.as_f32(), vec![1.0, 2.0]);
    }

    #[test]
    fn worker_send_result_with_error_payload() {
        let mut w = make_worker();
        let result = ResultMsg {
            req_id: 1,
            ok: false,
            error: Some("kaboom".to_string()),
            output_hc: Tensor::from_f32(&[], Shape::new([0])),
            output_logits: Tensor::from_f32(&[], Shape::new([0])),
        };
        w.send_result(&result).unwrap();
        let bytes = drain_worker(&mut w);
        let mut cur = std::io::Cursor::new(bytes);
        let frame = Frame::read_from(&mut cur).unwrap();
        assert_eq!(frame.msg_type, msg::RESULT);
        let r2 = ResultMsg::decode(&frame.payload).unwrap();
        assert!(!r2.ok);
        assert_eq!(r2.error.as_deref(), Some("kaboom"));
    }

    #[test]
    fn worker_new_asserts_role() {
        let bad = Ds4DistributedOptions {
            role: Ds4DistributedRole::Coordinator,
            ..Ds4DistributedOptions::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Worker::new(
                bad,
                TcpStream::new(Cursor::new(Vec::<u8>::new()), "x"),
                1,
                1,
            );
        }));
        assert!(result.is_err());
    }

    #[test]
    fn worker_read_work_rejects_wrong_msg_type() {
        // Seed the worker with a HELLO frame; read_work expects WORK.
        let mut w = make_worker();
        let hello = Hello::worker(1, 1);
        seed_worker_with_hello(&mut w, &hello);
        let err = w.read_work().unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn worker_new_picks_up_layer_slice_from_opts() {
        let o = Ds4DistributedOptions {
            role: Ds4DistributedRole::Worker,
            layers: ds4_types::Ds4LayerSlice {
                start: 4,
                end: 16,
                has_output: true,
                set: true,
            },
            ..Ds4DistributedOptions::default()
        };
        let w = Worker::new(
            o,
            TcpStream::new(Cursor::new(Vec::<u8>::new()), "x"),
            32,
            4096,
        );
        assert_eq!(w.layer_start, 4);
        assert_eq!(w.layer_end, 16);
        assert!(w.has_output);
    }
}
