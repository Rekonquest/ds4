// DS4 (DwarfStar) — distributed wire protocol.
//
// Frame format (little-endian, no padding):
//
//   +---------+----------+---------+--------------+
//   | magic   | msg_type | length  | payload      |
//   | u32 LE  | u32 LE   | u32 LE  | length bytes |
//   +---------+----------+---------+--------------+
//
// `magic` is always `DS4_DIST_WIRE_MAGIC` (`0x4453_3444` = "DS4D").
// Receivers that see any other magic abort the connection. The
// length is the byte length of the payload that follows; the
// payload itself is message-type-specific and serialized with the
// helpers below.
//
// Message types (`crate::msg`):
//   * HELLO        — sent worker -> coordinator on connect.
//                    Payload: `Hello { role, n_layers, head_elements,
//                                       activation_bits }`.
//   * WORK         — sent coordinator -> worker to dispatch a
//                    layer slice. Payload: `Work { req_id, tokens,
//                                                 pos0, layer_start,
//                                                 layer_end,
//                                                 input_hc }`.
//   * RESULT       — sent worker -> coordinator with the output
//                    of a WORK. Payload: `Result { req_id, ok,
//                                                  output_hc,
//                                                  output_logits }`.
//   * SNAPSHOT_*   — snapshot stream (BEGIN, CHUNK, END, REQ).
//                    SNAPSHOT_CHUNK payload max size is
//                    `SNAPSHOT_CHUNK_BYTES` (8 MiB).

use crate::msg;
use crate::{DEFAULT_ACTIVATION_BITS, DS4_DIST_WIRE_MAGIC, SNAPSHOT_CHUNK_BYTES};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ds4_tensor::Tensor;
use ds4_types::{Ds4DistributedRole, Ds4Error, Ds4ErrorKind, Ds4Result};
use std::io::{Read, Write};

/// Magic constant for the wire format, exposed at the crate root.
pub const MAGIC: u32 = DS4_DIST_WIRE_MAGIC;

/// Wire-format version. Bumped whenever the frame layout changes
/// in a non-backwards-compatible way. v1 ships now; later revisions may add new
/// payload fields or compress the frame.
pub const WIRE_VERSION: u32 = 1;

/// Default activation-bits field used when a HELLO doesn't carry
/// one (defensive — every real HELLO sets it explicitly).
pub const DEFAULT_ACTIVATION_BITS_WIRE: u8 = DEFAULT_ACTIVATION_BITS;

/// HELLO payload: sent by both peers on connect. The coordinator
/// rejects HELLO packets whose `wire_version` doesn't match
/// `WIRE_VERSION`. The worker uses `n_layers` and `head_elements`
/// to size its local slice of the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Hello {
    pub wire_version: u32,
    pub role: Ds4DistributedRole,
    pub n_layers: u32,
    pub head_elements: u32,
    pub activation_bits: u8,
}

impl Hello {
    pub fn coordinator(n_layers: u32, head_elements: u32) -> Self {
        Self {
            wire_version: WIRE_VERSION,
            role: Ds4DistributedRole::Coordinator,
            n_layers,
            head_elements,
            activation_bits: DEFAULT_ACTIVATION_BITS_WIRE,
        }
    }

    pub fn worker(n_layers: u32, head_elements: u32) -> Self {
        Self {
            wire_version: WIRE_VERSION,
            role: Ds4DistributedRole::Worker,
            n_layers,
            head_elements,
            activation_bits: DEFAULT_ACTIVATION_BITS_WIRE,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + 1 + 4 + 4 + 1);
        buf.write_u32::<LittleEndian>(self.wire_version).unwrap();
        buf.write_u8(role_to_u8(self.role)).unwrap();
        buf.write_u32::<LittleEndian>(self.n_layers).unwrap();
        buf.write_u32::<LittleEndian>(self.head_elements).unwrap();
        buf.write_u8(self.activation_bits).unwrap();
        buf
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        if bytes.len() < 14 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("HELLO payload too short: {} bytes (need 14)", bytes.len()),
            ));
        }
        let mut cur = std::io::Cursor::new(bytes);
        let wire_version = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let role = role_from_u8(cur.read_u8().map_err(io_err)?)?;
        let n_layers = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let head_elements = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let activation_bits = cur.read_u8().map_err(io_err)?;
        Ok(Self {
            wire_version,
            role,
            n_layers,
            head_elements,
            activation_bits,
        })
    }
}

/// WORK payload: a request to run `tokens[pos0..]` through layers
/// `[layer_start, layer_end)`.
#[derive(Debug, Clone)]
pub struct Work {
    pub req_id: u64,
    pub tokens: Vec<u32>,
    pub pos0: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub input_hc: Tensor,
}

/// Compare two tensors element-by-element when both are F32 (using
/// the host round-trip helpers), or byte-for-byte otherwise. Used
/// by the wire-message `PartialEq` impls.
fn tensor_eq(a: &Tensor, b: &Tensor) -> bool {
    if a.dtype != b.dtype || a.shape.dims() != b.shape.dims() {
        return false;
    }
    if a.dtype == ds4_tensor::DType::F32 {
        a.as_f32() == b.as_f32()
    } else {
        a.as_bytes() == b.as_bytes()
    }
}

impl PartialEq for Work {
    fn eq(&self, other: &Self) -> bool {
        self.req_id == other.req_id
            && self.tokens == other.tokens
            && self.pos0 == other.pos0
            && self.layer_start == other.layer_start
            && self.layer_end == other.layer_end
            && tensor_eq(&self.input_hc, &other.input_hc)
    }
}

impl Work {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(self.req_id).unwrap();
        buf.write_u32::<LittleEndian>(self.tokens.len() as u32)
            .unwrap();
        for &t in &self.tokens {
            buf.write_u32::<LittleEndian>(t).unwrap();
        }
        buf.write_u64::<LittleEndian>(self.pos0 as u64).unwrap();
        buf.write_u64::<LittleEndian>(self.layer_start as u64)
            .unwrap();
        buf.write_u64::<LittleEndian>(self.layer_end as u64)
            .unwrap();
        encode_tensor(&mut buf, &self.input_hc);
        buf
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let req_id = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let n_tok = cur.read_u32::<LittleEndian>().map_err(io_err)? as usize;
        let mut tokens = Vec::with_capacity(n_tok);
        for _ in 0..n_tok {
            tokens.push(cur.read_u32::<LittleEndian>().map_err(io_err)?);
        }
        let pos0 = cur.read_u64::<LittleEndian>().map_err(io_err)? as usize;
        let layer_start = cur.read_u64::<LittleEndian>().map_err(io_err)? as usize;
        let layer_end = cur.read_u64::<LittleEndian>().map_err(io_err)? as usize;
        let input_hc = decode_tensor(&mut cur)?;
        Ok(Self {
            req_id,
            tokens,
            pos0,
            layer_start,
            layer_end,
            input_hc,
        })
    }
}

/// RESULT payload: a worker's reply to a WORK.
#[derive(Debug, Clone)]
pub struct ResultMsg {
    pub req_id: u64,
    pub ok: bool,
    pub error: Option<String>,
    pub output_hc: Tensor,
    pub output_logits: Tensor,
}

impl PartialEq for ResultMsg {
    fn eq(&self, other: &Self) -> bool {
        self.req_id == other.req_id
            && self.ok == other.ok
            && self.error == other.error
            && tensor_eq(&self.output_hc, &other.output_hc)
            && tensor_eq(&self.output_logits, &other.output_logits)
    }
}

impl ResultMsg {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(self.req_id).unwrap();
        buf.write_u8(if self.ok { 1 } else { 0 }).unwrap();
        match &self.error {
            None => {
                buf.write_u32::<LittleEndian>(0xFFFF_FFFF).unwrap();
            }
            Some(s) => {
                let bytes = s.as_bytes();
                buf.write_u32::<LittleEndian>(bytes.len() as u32).unwrap();
                buf.write_all(bytes).unwrap();
            }
        }
        encode_tensor(&mut buf, &self.output_hc);
        encode_tensor(&mut buf, &self.output_logits);
        buf
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let req_id = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let ok = cur.read_u8().map_err(io_err)? != 0;
        let err_len = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let error = if err_len == 0xFFFF_FFFF {
            None
        } else {
            let mut s = vec![0u8; err_len as usize];
            cur.read_exact(&mut s).map_err(io_err)?;
            Some(String::from_utf8_lossy(&s).into_owned())
        };
        let output_hc = decode_tensor(&mut cur)?;
        let output_logits = decode_tensor(&mut cur)?;
        Ok(Self {
            req_id,
            ok,
            error,
            output_hc,
            output_logits,
        })
    }
}

/// SNAPSHOT_BEGIN payload: announces the start of a snapshot
/// transfer. `total_bytes` is the total uncompressed payload size;
/// `chunk_count` is how many SNAPSHOT_CHUNK frames will follow.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotBegin {
    pub snapshot_id: u64,
    pub total_bytes: u64,
    pub chunk_count: u32,
    pub metadata: Vec<u8>,
}

impl SnapshotBegin {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(self.snapshot_id).unwrap();
        buf.write_u64::<LittleEndian>(self.total_bytes).unwrap();
        buf.write_u32::<LittleEndian>(self.chunk_count).unwrap();
        buf.write_u32::<LittleEndian>(self.metadata.len() as u32)
            .unwrap();
        buf.write_all(&self.metadata).unwrap();
        buf
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let snapshot_id = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let total_bytes = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let chunk_count = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let meta_len = cur.read_u32::<LittleEndian>().map_err(io_err)? as usize;
        let mut metadata = vec![0u8; meta_len];
        if meta_len > 0 {
            cur.read_exact(&mut metadata).map_err(io_err)?;
        }
        Ok(Self {
            snapshot_id,
            total_bytes,
            chunk_count,
            metadata,
        })
    }
}

/// SNAPSHOT_CHUNK payload: one slice of a snapshot's bytes.
/// `payload` must be ≤ `SNAPSHOT_CHUNK_BYTES`.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotChunk {
    pub snapshot_id: u64,
    pub chunk_index: u32,
    pub payload: Vec<u8>,
}

impl SnapshotChunk {
    pub fn encode(&self) -> Ds4Result<Vec<u8>> {
        if self.payload.len() > SNAPSHOT_CHUNK_BYTES {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "SNAPSHOT_CHUNK payload too large: {} > {}",
                    self.payload.len(),
                    SNAPSHOT_CHUNK_BYTES,
                ),
            ));
        }
        let mut buf = Vec::with_capacity(16 + self.payload.len());
        buf.write_u64::<LittleEndian>(self.snapshot_id).unwrap();
        buf.write_u32::<LittleEndian>(self.chunk_index).unwrap();
        buf.write_u32::<LittleEndian>(self.payload.len() as u32)
            .unwrap();
        buf.write_all(&self.payload).unwrap();
        Ok(buf)
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let snapshot_id = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let chunk_index = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let len = cur.read_u32::<LittleEndian>().map_err(io_err)? as usize;
        if len > SNAPSHOT_CHUNK_BYTES {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "SNAPSHOT_CHUNK payload too large on decode: {} > {}",
                    len, SNAPSHOT_CHUNK_BYTES,
                ),
            ));
        }
        let mut payload = vec![0u8; len];
        if len > 0 {
            cur.read_exact(&mut payload).map_err(io_err)?;
        }
        Ok(Self {
            snapshot_id,
            chunk_index,
            payload,
        })
    }
}

/// SNAPSHOT_END payload: marks the last chunk. The receiver
/// validates that `received_bytes == begin.total_bytes` and that
/// `received_chunks == begin.chunk_count`.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotEnd {
    pub snapshot_id: u64,
    pub received_bytes: u64,
    pub received_chunks: u32,
    pub crc32: u32,
}

impl SnapshotEnd {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(self.snapshot_id).unwrap();
        buf.write_u64::<LittleEndian>(self.received_bytes).unwrap();
        buf.write_u32::<LittleEndian>(self.received_chunks).unwrap();
        buf.write_u32::<LittleEndian>(self.crc32).unwrap();
        buf
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let snapshot_id = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let received_bytes = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let received_chunks = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        let crc32 = cur.read_u32::<LittleEndian>().map_err(io_err)?;
        Ok(Self {
            snapshot_id,
            received_bytes,
            received_chunks,
            crc32,
        })
    }
}

/// SNAPSHOT_REQ payload: a worker asks the coordinator for a
/// snapshot (used after a worker restart to rebuild its state).
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotReq {
    pub snapshot_id: u64,
    pub last_known_version: u64,
}

impl SnapshotReq {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(self.snapshot_id).unwrap();
        buf.write_u64::<LittleEndian>(self.last_known_version)
            .unwrap();
        buf
    }

    pub fn decode(bytes: &[u8]) -> Ds4Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let snapshot_id = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        let last_known_version = cur.read_u64::<LittleEndian>().map_err(io_err)?;
        Ok(Self {
            snapshot_id,
            last_known_version,
        })
    }
}

/// Errors that come out of the wire layer. Wrap `std::io::Error`
/// and a few domain-specific issues (bad magic, bad version, payload
/// too large).
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic: got {got:#x}, expected {expected:#x}")]
    BadMagic { got: u32, expected: u32 },
    #[error("frame length too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("short read: needed {needed} bytes, got {got}")]
    ShortRead { needed: usize, got: usize },
    #[error("payload too large: {got} > {max}")]
    PayloadTooLarge { got: usize, max: usize },
    #[error("unsupported wire version: {0}")]
    BadVersion(u32),
    #[error("unsupported message type: {0}")]
    BadMsgType(u32),
    #[error("domain error: {0}")]
    Domain(String),
}

/// A complete frame: `magic + msg_type + length + payload`.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub magic: u32,
    pub msg_type: u32,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Build a frame. Validates magic + length.
    pub fn new(msg_type: u32, payload: Vec<u8>) -> Ds4Result<Self> {
        if payload.len() > MAX_FRAME_PAYLOAD {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "Frame::new: payload too large: {} > {}",
                    payload.len(),
                    MAX_FRAME_PAYLOAD,
                ),
            ));
        }
        Ok(Self {
            magic: MAGIC,
            msg_type,
            payload,
        })
    }

    /// Serialize a frame to bytes (header + payload).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.payload.len());
        buf.write_u32::<LittleEndian>(self.magic).unwrap();
        buf.write_u32::<LittleEndian>(self.msg_type).unwrap();
        buf.write_u32::<LittleEndian>(self.payload.len() as u32)
            .unwrap();
        buf.write_all(&self.payload).unwrap();
        buf
    }

    /// Read a frame from a reader. Reads the 12-byte header first,
    /// then the payload. Returns the typed payload via the caller-
    /// supplied decoder.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, WireError> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != MAGIC {
            return Err(WireError::BadMagic {
                got: magic,
                expected: MAGIC,
            });
        }
        let msg_type = reader.read_u32::<LittleEndian>()?;
        let length = reader.read_u32::<LittleEndian>()? as usize;
        if length > MAX_FRAME_PAYLOAD {
            return Err(WireError::PayloadTooLarge {
                got: length,
                max: MAX_FRAME_PAYLOAD,
            });
        }
        let mut payload = vec![0u8; length];
        if length > 0 {
            reader.read_exact(&mut payload)?;
        }
        Ok(Self {
            magic,
            msg_type,
            payload,
        })
    }

    /// Write a frame to a writer. Mirrors `read_from` byte-for-byte.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), WireError> {
        if self.magic != MAGIC {
            return Err(WireError::BadMagic {
                got: self.magic,
                expected: MAGIC,
            });
        }
        writer.write_u32::<LittleEndian>(self.magic)?;
        writer.write_u32::<LittleEndian>(self.msg_type)?;
        writer.write_u32::<LittleEndian>(self.payload.len() as u32)?;
        writer.write_all(&self.payload)?;
        Ok(())
    }
}

/// The maximum payload size we accept on a single frame.
/// `SNAPSHOT_CHUNK_BYTES * 2` is a conservative ceiling — anything
/// larger is almost certainly a malformed frame and we abort the
/// connection rather than risk a memory blow-up.
pub const MAX_FRAME_PAYLOAD: usize = SNAPSHOT_CHUNK_BYTES * 2;

// ---------------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------------

fn encode_tensor<W: Write>(w: &mut W, t: &Tensor) {
    let dtype = dtype_to_u8(t.dtype);
    let rank = t.shape.dims().len() as u32;
    w.write_u8(dtype).unwrap();
    w.write_u32::<LittleEndian>(rank).unwrap();
    for &d in t.shape.dims() {
        w.write_u64::<LittleEndian>(d as u64).unwrap();
    }
    let bytes = t.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64).unwrap();
    w.write_all(bytes).unwrap();
}

fn decode_tensor<R: Read>(r: &mut R) -> Ds4Result<Tensor> {
    let dtype_u8 = r.read_u8().map_err(io_err)?;
    let dtype = dtype_from_u8(dtype_u8)?;
    let rank = r.read_u32::<LittleEndian>().map_err(io_err)? as usize;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(r.read_u64::<LittleEndian>().map_err(io_err)? as usize);
    }
    let n_bytes = r.read_u64::<LittleEndian>().map_err(io_err)? as usize;
    let mut data = vec![0u8; n_bytes];
    if n_bytes > 0 {
        r.read_exact(&mut data).map_err(io_err)?;
    }
    Ok(Tensor {
        dtype,
        shape: ds4_tensor::Shape::from(dims),
        data,
        device: ds4_tensor::Device::Cpu,
    })
}

fn dtype_to_u8(d: ds4_tensor::DType) -> u8 {
    use ds4_tensor::DType::*;
    match d {
        F32 => 0,
        F16 => 1,
        BF16 => 2,
        F64 => 3,
        I8 => 4,
        U8 => 5,
        I64 => 6,
        U32 => 7,
        U64 => 8,
    }
}

fn dtype_from_u8(b: u8) -> Ds4Result<ds4_tensor::DType> {
    use ds4_tensor::DType::*;
    Ok(match b {
        0 => F32,
        1 => F16,
        2 => BF16,
        3 => F64,
        4 => I8,
        5 => U8,
        6 => I64,
        7 => U32,
        8 => U64,
        other => {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("unknown dtype tag {other}"),
            ))
        }
    })
}

fn role_to_u8(r: Ds4DistributedRole) -> u8 {
    match r {
        Ds4DistributedRole::None => 0,
        Ds4DistributedRole::Coordinator => 1,
        Ds4DistributedRole::Worker => 2,
    }
}

fn role_from_u8(b: u8) -> Ds4Result<Ds4DistributedRole> {
    Ok(match b {
        0 => Ds4DistributedRole::None,
        1 => Ds4DistributedRole::Coordinator,
        2 => Ds4DistributedRole::Worker,
        other => {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("unknown role tag {other}"),
            ))
        }
    })
}

fn io_err(e: std::io::Error) -> Ds4Error {
    Ds4Error::new(Ds4ErrorKind::Io, format!("io: {e}"))
}

// ---------------------------------------------------------------------------
// Convenience constructors that bundle the msg_type + payload into a Frame.
// ---------------------------------------------------------------------------

/// Build a HELLO frame from a `Hello` payload.
pub fn frame_hello(h: &Hello) -> Ds4Result<Frame> {
    Frame::new(msg::HELLO, h.encode())
}

/// Build a WORK frame from a `Work` payload.
pub fn frame_work(w: &Work) -> Ds4Result<Frame> {
    Frame::new(msg::WORK, w.encode())
}

/// Build a RESULT frame from a `ResultMsg` payload.
pub fn frame_result(r: &ResultMsg) -> Ds4Result<Frame> {
    Frame::new(msg::RESULT, r.encode())
}

/// Build a SNAPSHOT_BEGIN frame.
pub fn frame_snapshot_begin(b: &SnapshotBegin) -> Ds4Result<Frame> {
    Frame::new(msg::SNAPSHOT_BEGIN, b.encode())
}

/// Build a SNAPSHOT_CHUNK frame.
pub fn frame_snapshot_chunk(c: &SnapshotChunk) -> Ds4Result<Frame> {
    Frame::new(msg::SNAPSHOT_CHUNK, c.encode()?)
}

/// Build a SNAPSHOT_END frame.
pub fn frame_snapshot_end(e: &SnapshotEnd) -> Ds4Result<Frame> {
    Frame::new(msg::SNAPSHOT_END, e.encode())
}

/// Build a SNAPSHOT_REQ frame.
pub fn frame_snapshot_req(r: &SnapshotReq) -> Ds4Result<Frame> {
    Frame::new(msg::SNAPSHOT_REQ, r.encode())
}

/// Decode a payload buffer according to the message type. Returns
/// an `Err` for unknown types.
pub fn decode_payload(msg_type: u32, payload: &[u8]) -> Ds4Result<Decoded> {
    Ok(match msg_type {
        msg::HELLO => Decoded::Hello(Hello::decode(payload)?),
        msg::WORK => Decoded::Work(Work::decode(payload)?),
        msg::RESULT => Decoded::Result(ResultMsg::decode(payload)?),
        msg::SNAPSHOT_BEGIN => Decoded::SnapshotBegin(SnapshotBegin::decode(payload)?),
        msg::SNAPSHOT_CHUNK => Decoded::SnapshotChunk(SnapshotChunk::decode(payload)?),
        msg::SNAPSHOT_END => Decoded::SnapshotEnd(SnapshotEnd::decode(payload)?),
        msg::SNAPSHOT_REQ => Decoded::SnapshotReq(SnapshotReq::decode(payload)?),
        other => {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("unknown message type {other}"),
            ))
        }
    })
}

/// Decoded payload, tagged by message type.
#[derive(Debug, Clone, PartialEq)]
pub enum Decoded {
    Hello(Hello),
    Work(Work),
    Result(ResultMsg),
    SnapshotBegin(SnapshotBegin),
    SnapshotChunk(SnapshotChunk),
    SnapshotEnd(SnapshotEnd),
    SnapshotReq(SnapshotReq),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_tensor::{Shape, Tensor};

    fn tiny_tensor() -> Tensor {
        Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]))
    }

    #[test]
    fn hello_roundtrip() {
        let h = Hello::worker(32, 4096);
        let bytes = h.encode();
        let h2 = Hello::decode(&bytes).unwrap();
        assert_eq!(h, h2);
        assert_eq!(h.wire_version, WIRE_VERSION);
    }

    #[test]
    fn hello_short_payload_rejected() {
        let err = Hello::decode(&[0u8; 5]).unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn work_roundtrip() {
        let w = Work {
            req_id: 42,
            tokens: vec![1, 2, 3, 4, 5],
            pos0: 7,
            layer_start: 4,
            layer_end: 16,
            input_hc: tiny_tensor(),
        };
        let bytes = w.encode();
        let w2 = Work::decode(&bytes).unwrap();
        assert_eq!(w.req_id, w2.req_id);
        assert_eq!(w.tokens, w2.tokens);
        assert_eq!(w.pos0, w2.pos0);
        assert_eq!(w.layer_start, w2.layer_start);
        assert_eq!(w.layer_end, w2.layer_end);
        assert_eq!(w.input_hc.as_f32(), w2.input_hc.as_f32());
    }

    #[test]
    fn result_roundtrip_ok() {
        let r = ResultMsg {
            req_id: 100,
            ok: true,
            error: None,
            output_hc: tiny_tensor(),
            output_logits: tiny_tensor(),
        };
        let bytes = r.encode();
        let r2 = ResultMsg::decode(&bytes).unwrap();
        assert_eq!(r.req_id, r2.req_id);
        assert_eq!(r.ok, r2.ok);
        assert_eq!(r.error, r2.error);
        assert_eq!(r.output_hc.as_f32(), r2.output_hc.as_f32());
        assert_eq!(r.output_logits.as_f32(), r2.output_logits.as_f32());
    }

    #[test]
    fn result_roundtrip_with_error() {
        let r = ResultMsg {
            req_id: 101,
            ok: false,
            error: Some("forward failed: NaN".to_string()),
            output_hc: tiny_tensor(),
            output_logits: tiny_tensor(),
        };
        let bytes = r.encode();
        let r2 = ResultMsg::decode(&bytes).unwrap();
        assert!(!r2.ok);
        assert_eq!(r2.error.as_deref(), Some("forward failed: NaN"));
    }

    #[test]
    fn snapshot_chunk_roundtrip() {
        let c = SnapshotChunk {
            snapshot_id: 7,
            chunk_index: 3,
            payload: vec![0xAB; 1024],
        };
        let bytes = c.encode().unwrap();
        let c2 = SnapshotChunk::decode(&bytes).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn snapshot_chunk_oversize_rejected() {
        let c = SnapshotChunk {
            snapshot_id: 1,
            chunk_index: 0,
            payload: vec![0u8; SNAPSHOT_CHUNK_BYTES + 1],
        };
        let err = c.encode().unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn snapshot_begin_roundtrip() {
        let b = SnapshotBegin {
            snapshot_id: 42,
            total_bytes: 1_000_000,
            chunk_count: 128,
            metadata: vec![1, 2, 3, 4],
        };
        let bytes = b.encode();
        let b2 = SnapshotBegin::decode(&bytes).unwrap();
        assert_eq!(b, b2);
    }

    #[test]
    fn snapshot_end_roundtrip() {
        let e = SnapshotEnd {
            snapshot_id: 42,
            received_bytes: 1_000_000,
            received_chunks: 128,
            crc32: 0xDEAD_BEEF,
        };
        let bytes = e.encode();
        let e2 = SnapshotEnd::decode(&bytes).unwrap();
        assert_eq!(e, e2);
    }

    #[test]
    fn snapshot_req_roundtrip() {
        let r = SnapshotReq {
            snapshot_id: 9,
            last_known_version: 100,
        };
        let bytes = r.encode();
        let r2 = SnapshotReq::decode(&bytes).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn frame_roundtrip() {
        let h = Hello::coordinator(40, 4096);
        let frame = frame_hello(&h).unwrap();
        let bytes = frame.encode();
        // Re-decode: parse header, then payload.
        let mut cur = std::io::Cursor::new(&bytes);
        let f2 = Frame::read_from(&mut cur).unwrap();
        assert_eq!(f2.magic, MAGIC);
        assert_eq!(f2.msg_type, msg::HELLO);
        let decoded = decode_payload(f2.msg_type, &f2.payload).unwrap();
        match decoded {
            Decoded::Hello(h2) => assert_eq!(h, h2),
            _ => panic!("expected HELLO"),
        }
    }

    #[test]
    fn frame_rejects_bad_magic() {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(0xDEADBEEF).unwrap();
        bytes.write_u32::<LittleEndian>(msg::HELLO).unwrap();
        bytes.write_u32::<LittleEndian>(0).unwrap();
        let mut cur = std::io::Cursor::new(&bytes);
        let err = Frame::read_from(&mut cur).unwrap_err();
        match err {
            WireError::BadMagic { got, expected } => {
                assert_eq!(got, 0xDEADBEEF);
                assert_eq!(expected, MAGIC);
            }
            _ => panic!("expected BadMagic, got {err:?}"),
        }
    }

    #[test]
    fn frame_rejects_oversize_payload() {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(MAGIC).unwrap();
        bytes.write_u32::<LittleEndian>(msg::HELLO).unwrap();
        bytes
            .write_u32::<LittleEndian>((MAX_FRAME_PAYLOAD + 1) as u32)
            .unwrap();
        let mut cur = std::io::Cursor::new(&bytes);
        let err = Frame::read_from(&mut cur).unwrap_err();
        assert!(matches!(err, WireError::PayloadTooLarge { .. }));
    }

    #[test]
    fn decode_payload_dispatches_by_msg_type() {
        let h = Hello::worker(1, 1);
        let frame = frame_hello(&h).unwrap();
        let decoded = decode_payload(frame.msg_type, &frame.payload).unwrap();
        assert!(matches!(decoded, Decoded::Hello(_)));
    }

    #[test]
    fn decode_payload_unknown_type_errors() {
        let err = decode_payload(9999, &[]).unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn magic_constant_matches_c() {
        assert_eq!(MAGIC, 0x4453_3444);
    }

    #[test]
    fn wire_version_is_one() {
        assert_eq!(WIRE_VERSION, 1);
    }

    #[test]
    fn dtype_roundtrip_all_variants() {
        for tag in 0u8..=8 {
            let _ = dtype_from_u8(tag).unwrap();
        }
        let err = dtype_from_u8(42).unwrap_err();
        assert_eq!(err.kind, Ds4ErrorKind::InvalidArgument);
    }

    #[test]
    fn tensor_in_frame_survives_roundtrip() {
        let t = Tensor::from_f32(&[0.5, -1.5, 2.5, 3.5], Shape::new([4]));
        let mut payload = Vec::new();
        encode_tensor(&mut payload, &t);
        let mut cur = std::io::Cursor::new(&payload);
        let t2 = decode_tensor(&mut cur).unwrap();
        assert_eq!(t.dtype, t2.dtype);
        assert_eq!(t.shape.dims(), t2.shape.dims());
        assert_eq!(t.as_f32(), t2.as_f32());
    }

    #[test]
    fn frame_write_to_writer_roundtrip() {
        let h = Hello::worker(7, 11);
        let frame = frame_hello(&h).unwrap();
        let mut sink: Vec<u8> = Vec::new();
        frame.write_to(&mut sink).unwrap();
        let mut cur = std::io::Cursor::new(&sink);
        let f2 = Frame::read_from(&mut cur).unwrap();
        assert_eq!(f2.magic, frame.magic);
        assert_eq!(f2.msg_type, frame.msg_type);
        assert_eq!(f2.payload, frame.payload);
    }
}
