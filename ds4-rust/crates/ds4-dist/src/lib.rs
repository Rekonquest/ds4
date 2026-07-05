// DS4 (DwarfStar) — distributed inference.
//
// Wire protocol + coordinator/worker state machines + four API
// hooks that match `ds4_distributed.h` byte-for-byte. The concrete
// coordinator and worker types execute `BackendModel` layer-slice
// and output-head hooks through real WORK/RESULT frames. The
// session-agnostic C-style helpers below fail closed unless the
// caller supplies a route through those concrete types.
//
// `ds4-dist` depends on `ds4-types` + `ds4-tensor` only, never on
// `ds4-core`. The `Ds4Session` type at the call site is
// `ds4_core::session::Ds4Session` but it isn't brought into this
// crate's deps. The API hooks are generic over any `Send` handle
// so they don't pin a concrete type either.

pub const CRATE_NAME: &str = "ds4-dist";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use ds4_tensor::Tensor;
use ds4_types::{Ds4DistributedRole, Ds4Error, Ds4ErrorKind, Ds4Result};

pub mod coordinator;
pub mod protocol;
pub mod worker;

pub const DS4_DIST_WIRE_MAGIC: u32 = 0x4453_3444; // "DS4D"

pub mod msg {
    pub const HELLO: u32 = 1;
    pub const WORK: u32 = 3;
    pub const RESULT: u32 = 4;
    pub const SNAPSHOT_BEGIN: u32 = 5;
    pub const SNAPSHOT_CHUNK: u32 = 6;
    pub const SNAPSHOT_END: u32 = 7;
    pub const SNAPSHOT_REQ: u32 = 8;
}

pub const DEFAULT_ACTIVATION_BITS: u8 = 32;
pub const SNAPSHOT_CHUNK_BYTES: usize = 8 * 1024 * 1024;

// Re-export the protocol/coord/worker public surface so callers
// don't need to know the internal module layout.
pub use coordinator::{Coordinator, LayerSlice, WorkerConn};
pub use protocol::{
    frame_hello, frame_result, frame_snapshot_begin, frame_snapshot_chunk, frame_snapshot_end,
    frame_snapshot_req, frame_work, Hello, ResultMsg, SnapshotBegin, SnapshotChunk, SnapshotEnd,
    SnapshotReq, Work, MAGIC as WIRE_MAGIC,
};
pub use worker::Worker;

/// Generic session handle so `ds4-dist` doesn't need to depend on
/// `ds4-core`. At the call site, callers pass
/// `&mut ds4_core::session::Ds4Session` which is `Send`.
pub trait SessionHandle: Send {}

impl<T: Send> SessionHandle for T {}

/// Optional session-role extension for callers that wrap a concrete
/// coordinator or worker route outside this leaf crate.
pub trait SessionDist {
    fn distributed_role(&self) -> Ds4DistributedRole;
}

fn session_route_missing() -> Ds4Error {
    Ds4Error::new(
        Ds4ErrorKind::NotImplemented,
        "distributed session route is absent for this generic handle",
    )
}

/// Reset per-request slice state. The session-agnostic helper has
/// no local bookkeeping, so reset is a successful no-op.
pub fn layer_slice_reset<S: SessionHandle>(_session: &mut S) -> Ds4Result<()> {
    Ok(())
}

/// Run the forward pass over `[layer_start, layer_end)`.
///
/// The concrete `Coordinator` and `Worker` types execute non-empty
/// layer ranges. This C-style helper can only perform the empty
/// range identity case because it does not own a route or model.
#[allow(clippy::too_many_arguments)]
pub fn eval_layer_slice<S: SessionHandle>(
    session: &mut S,
    tokens: &[u32],
    pos0: usize,
    layer_start: usize,
    layer_end: usize,
    input_hc: &Tensor,
    output_hc: &mut Tensor,
    output_logits: &mut Tensor,
) -> Ds4Result<()> {
    let _ = (session, tokens, pos0);
    if layer_start == layer_end {
        *output_hc = input_hc.clone();
        *output_logits = Tensor::from_f32(&[], ds4_tensor::Shape::new([0]));
        return Ok(());
    }
    Err(session_route_missing())
}

/// Run only the output head against the hidden state. Use
/// `Worker::serve_one` or a backend `BackendModel` directly when a
/// concrete model route is available.
pub fn eval_output_head_from_hc<S: SessionHandle>(
    _session: &mut S,
    _hidden_hc: &Tensor,
    _n_tokens: usize,
    _logits: &mut Tensor,
) -> Ds4Result<()> {
    Err(session_route_missing())
}

/// `true` if this leaf helper can see an attached distributed route.
pub fn distributed_route_ready<S: SessionHandle>(session: &S) -> Ds4Result<bool> {
    Ok(session_role(session).is_some())
}

/// Convenience: `true` if the helper can see a distributed role.
pub fn is_distributed<S: SessionHandle>(session: &S) -> bool {
    session_role(session).is_some()
}

/// Internal: this leaf crate cannot downcast arbitrary session
/// handles, so generic handles report no visible route.
fn session_role<S: SessionHandle + ?Sized>(session: &S) -> Option<Ds4DistributedRole> {
    let _ = session;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_constants_match_c() {
        assert_eq!(DS4_DIST_WIRE_MAGIC, 0x4453_3444);
        assert_eq!(SNAPSHOT_CHUNK_BYTES, 8 * 1024 * 1024);
        assert_eq!(msg::HELLO, 1);
        assert_eq!(msg::WORK, 3);
        assert_eq!(msg::RESULT, 4);
        assert_eq!(DEFAULT_ACTIVATION_BITS, 32);
    }

    #[test]
    fn route_helpers_fail_closed_without_attached_session() {
        struct ProbeSession;
        let mut s = ProbeSession;
        layer_slice_reset(&mut s).unwrap();
        assert!(!distributed_route_ready(&s).unwrap());
        assert!(!is_distributed(&s));
    }

    #[test]
    fn eval_layer_slice_zero_range_copies_input() {
        struct ProbeSession;
        let mut s = ProbeSession;
        let input = Tensor::from_f32(&[0.0; 4], ds4_tensor::Shape::new([4]));
        let mut output_hc = Tensor::from_f32(&[1.0; 4], ds4_tensor::Shape::new([4]));
        let mut output_logits = Tensor::from_f32(&[2.0; 4], ds4_tensor::Shape::new([4]));
        eval_layer_slice(
            &mut s,
            &[1, 2, 3],
            0,
            0,
            0,
            &input,
            &mut output_hc,
            &mut output_logits,
        )
        .unwrap();
        assert_eq!(output_hc.as_f32(), vec![0.0; 4]);
        assert!(output_logits.as_f32().is_empty());
    }
    #[test]
    fn eval_output_head_from_hc_fails_without_route() {
        struct ProbeSession;
        let mut s = ProbeSession;
        let hidden = Tensor::from_f32(&[0.0; 4], ds4_tensor::Shape::new([4]));
        let mut logits = Tensor::from_f32(&[0.0; 4], ds4_tensor::Shape::new([4]));
        let err = eval_output_head_from_hc(&mut s, &hidden, 1, &mut logits);
        assert!(err.is_err());
    }

    #[test]
    fn metadata_is_sane() {
        assert_eq!(CRATE_NAME, "ds4-dist");
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn magic_exported_at_crate_root() {
        assert_eq!(WIRE_MAGIC, 0x4453_3444);
    }

    #[test]
    fn is_distributed_false_for_plain_send_type() {
        struct ProbeSession;
        assert!(!is_distributed(&ProbeSession));
    }
}
