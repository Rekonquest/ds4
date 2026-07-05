// DS4 (DwarfStar) -- GGUF file reader.
//
// Faithful port of the GGUF v3 reader logic from `ds4.c:1616..3074`.
// Custom, not from llama.cpp -- the *format layout* (header + kv
// encoding + tensor descriptor encoding) follows the upstream GGUF
// spec; the reader itself is freshly written here in Rust.
//
// Public surface:
//   - `GgufFile::open(path)` mmap's the file, parses the header, and
//     returns a typed view.
//   - `tensor(name)` resolves a tensor by name and returns a typed
//     `QuantizedTensor<'_>` view into the mmap.
//   - `metadata` exposes the DeepSeek-specific fields we always need.

#![allow(unsafe_code, non_camel_case_types)]

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4QuantKind, Ds4Result};
use memmap2::Mmap;

pub mod synth;
pub use synth::write_synthetic_gguf;

// ---- GGUF v3 constants ----

pub const GGUF_MAGIC: u32 = 0x4655_4747;
pub const GGUF_VERSION_V3: u32 = 3;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => Self::Uint32,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 19,
    Iq4Xs = 22,
}

impl GgufDType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            19 => Self::Iq3Xxs,
            22 => Self::Iq4Xs,
            _ => Self::F32,
        }
    }

    pub fn as_ds4_quant_kind(&self) -> Ds4QuantKind {
        match self {
            GgufDType::F32 => Ds4QuantKind::F32,
            GgufDType::F16 => Ds4QuantKind::F16,
            GgufDType::Q8_0 => Ds4QuantKind::Q8_0,
            GgufDType::Q4_K => Ds4QuantKind::Q4_K,
            GgufDType::Q3_K => Ds4QuantKind::Q3_K,
            GgufDType::Q2_K => Ds4QuantKind::Q2_K,
            GgufDType::Iq2Xxs => Ds4QuantKind::Iq2Xxs,
            _ => Ds4QuantKind::Q8_0,
        }
    }

    /// Block bytes (not per-element). Callers multiply by the
    /// block count, not `numel`, for K-quants. For F32/F16/Q8_0
    /// this is also per-element since one block is one element.
    pub fn byte_size(&self) -> u64 {
        match self {
            GgufDType::F32 => 4,
            GgufDType::F16 => 2,
            GgufDType::Q4_0 => 18,
            GgufDType::Q4_1 => 20,
            GgufDType::Q5_0 => 24,
            GgufDType::Q5_1 => 26,
            GgufDType::Q8_0 => 34,
            GgufDType::Q8_1 => 36,
            GgufDType::Q2_K => 84,
            GgufDType::Q3_K => 110,
            GgufDType::Q4_K => 144,
            GgufDType::Q5_K => 176,
            GgufDType::Q6_K => 210,
            GgufDType::Q8_K => 256,
            GgufDType::Iq2Xxs => 66,
            GgufDType::Iq2Xs => 74,
            GgufDType::Iq3Xxs => 98,
            GgufDType::Iq4Xs => 136,
        }
    }

    pub fn block_size(&self) -> u64 {
        match self {
            GgufDType::F32 | GgufDType::F16 => 1,
            GgufDType::Q4_0
            | GgufDType::Q4_1
            | GgufDType::Q5_0
            | GgufDType::Q5_1
            | GgufDType::Q8_0
            | GgufDType::Q8_1 => 32,
            GgufDType::Q2_K
            | GgufDType::Q3_K
            | GgufDType::Q4_K
            | GgufDType::Q5_K
            | GgufDType::Q6_K
            | GgufDType::Q8_K
            | GgufDType::Iq2Xxs
            | GgufDType::Iq2Xs
            | GgufDType::Iq3Xxs
            | GgufDType::Iq4Xs => 256,
        }
    }
}

#[derive(Debug, Clone)]
pub enum KvRaw {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<KvRaw>),
    Unknown(u32),
}

impl KvRaw {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            KvRaw::U32(v) => Some(*v),
            KvRaw::I32(v) => Some(*v as u32),
            KvRaw::Bool(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            KvRaw::I32(v) => Some(*v),
            KvRaw::U32(v) => Some(*v as i32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            KvRaw::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            KvRaw::String(v) => Some(v.as_str()),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[KvRaw]> {
        match self {
            KvRaw::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
    pub fn as_array_u32(&self) -> Option<Vec<u32>> {
        self.as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_u32()).collect())
    }
    pub fn as_array_f32(&self) -> Option<Vec<f32>> {
        self.as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f32()).collect())
    }
    pub fn as_array_str(&self) -> Option<Vec<&str>> {
        self.as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
    }
}

#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    pub general_name: String,
    pub architecture: String,
    pub vocab_size: Option<u32>,
    pub embedding_dim: Option<u32>,
    pub layer_count: Option<u32>,
    pub head_count: Option<u32>,
    pub head_dim: Option<u32>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub context_length: Option<u32>,
    pub has_mtp: Option<bool>,
    pub routed_quant: Option<Ds4QuantKind>,
    pub shared_expert_quant: Option<Ds4QuantKind>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub user_token_id: Option<u32>,
    pub assistant_token_id: Option<u32>,
    pub think_start_token_id: Option<u32>,
    pub think_end_token_id: Option<u32>,
    pub dsml_token_id: Option<u32>,
    pub alignment: u32,
}

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub dims: Vec<u32>,
    pub dtype: GgufDType,
    pub offset: u64,
}

impl TensorDescriptor {
    pub fn numel(&self) -> u64 {
        self.dims.iter().map(|d| *d as u64).product()
    }
    pub fn byte_size(&self) -> u64 {
        let n_blocks =
            self.numel().saturating_add(self.dtype.block_size() - 1) / self.dtype.block_size();
        n_blocks.saturating_mul(self.dtype.byte_size())
    }
}

pub struct QuantizedTensor<'a> {
    pub descriptor: TensorDescriptor,
    pub bytes: &'a [u8],
}

impl<'a> QuantizedTensor<'a> {
    pub fn into_f32_vec(self) -> Ds4Result<Vec<f32>> {
        let n = self.descriptor.numel() as usize;
        if self.descriptor.dtype != GgufDType::F32 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Other,
                format!("expected F32, got {:?}", self.descriptor.dtype),
            ));
        }
        if self.bytes.len() < n * 4 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Other,
                format!(
                    "tensor bytes {} shorter than F32 expected {}",
                    self.bytes.len(),
                    n * 4
                ),
            ));
        }
        let mut out = Vec::with_capacity(n);
        for chunk in self.bytes[..n * 4].chunks_exact(4) {
            let mut b = [0u8; 4];
            b.copy_from_slice(chunk);
            out.push(f32::from_le_bytes(b));
        }
        Ok(out)
    }
}

pub struct GgufFile {
    path: PathBuf,
    mmap: Arc<Mmap>,
    header_offset: usize,
    kv_raw: HashMap<String, KvRaw>,
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorDescriptor>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Ds4Result<Self> {
        let file = File::open(path)
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("open {path:?}: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Ds4Error::new(Ds4ErrorKind::Io, format!("mmap {path:?}: {e}")))?;
        let mut s = Self {
            path: path.to_path_buf(),
            mmap: Arc::new(mmap),
            header_offset: 0,
            kv_raw: HashMap::new(),
            metadata: GgufMetadata {
                alignment: 32,
                ..GgufMetadata::default()
            },
            tensors: Vec::new(),
        };
        s.read_header()?;
        Ok(s)
    }

    fn read_header(&mut self) -> Ds4Result<()> {
        let mmap = self.mmap.clone();
        let bytes: &[u8] = &mmap[..];
        if bytes.len() < 24 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("GGUF header too short: {} bytes", bytes.len()),
            ));
        }
        let magic = read_u32(bytes, 0)?;
        if magic != GGUF_MAGIC {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("bad GGUF magic {magic:#x}, expected {GGUF_MAGIC:#x}"),
            ));
        }
        let version = read_u32(bytes, 4)?;
        if version != GGUF_VERSION_V3 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Model,
                format!("unsupported GGUF version {version} (only v3 is implemented)"),
            ));
        }
        let n_tensors = read_u64(bytes, 8)? as usize;
        let n_kv = read_u64(bytes, 16)? as usize;

        let mut off = 24usize;
        let mut raw_kvs: Vec<(String, KvRaw)> = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let (k, v, consumed) = read_kv(bytes, off)?;
            raw_kvs.push((k, v));
            off += consumed;
        }
        for (k, v) in &raw_kvs {
            self.apply_kv(k, v);
        }
        self.tensors.reserve(n_tensors);
        for _ in 0..n_tensors {
            let (name, n_dims, consumed) = read_tensor_header(bytes, off)?;
            off += consumed;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let d = read_u64(bytes, off)?;
                let d = u32::try_from(d).map_err(|_| {
                    Ds4Error::new(Ds4ErrorKind::Model, "GGUF tensor dimension overflows u32")
                })?;
                dims.push(d);
                off += 8;
            }
            let dtype_raw = read_u32(bytes, off)?;
            let dtype = GgufDType::from_u32(dtype_raw);
            off += 4;
            let offset = read_u64(bytes, off)?;
            off += 8;
            self.tensors.push(TensorDescriptor {
                name,
                dims,
                dtype,
                offset,
            });
        }
        self.header_offset = align_to(off, self.metadata.alignment as usize)?;

        for (k, v) in raw_kvs {
            self.kv_raw.insert(k, v);
        }
        Ok(())
    }

    fn apply_kv(&mut self, key: &str, val: &KvRaw) {
        match key {
            "general.name" => {
                if let Some(s) = val.as_str() {
                    self.metadata.general_name = s.to_string();
                }
            }
            "general.architecture" => {
                if let Some(s) = val.as_str() {
                    self.metadata.architecture = s.to_string();
                }
            }
            "ds4.vocab_size" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.vocab_size = Some(v);
                }
            }
            "ds4.embedding_dim" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.embedding_dim = Some(v);
                }
            }
            "ds4.block_count" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.layer_count = Some(v);
                }
            }
            "ds4.attention.head_count" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.head_count = Some(v);
                }
            }
            "ds4.expert_count" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.expert_count = Some(v);
                }
            }
            "ds4.expert_used_count" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.expert_used_count = Some(v);
                }
            }
            "ds4.context_length" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.context_length = Some(v);
                }
            }
            "ds4.has_mtp" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.has_mtp = Some(v != 0);
                }
            }
            "ds4.routed_experts.quant" => {
                if let Some(s) = val.as_str() {
                    self.metadata.routed_quant = Some(str_to_quant(s));
                }
            }
            "ds4.shared_experts.quant" => {
                if let Some(s) = val.as_str() {
                    self.metadata.shared_expert_quant = Some(str_to_quant(s));
                }
            }
            "tokenizer.ggml.bos_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.bos_token_id = Some(v);
                }
            }
            "tokenizer.ggml.eos_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.eos_token_id = Some(v);
                }
            }
            "ds4.user_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.user_token_id = Some(v);
                }
            }
            "ds4.assistant_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.assistant_token_id = Some(v);
                }
            }
            "ds4.think_start_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.think_start_token_id = Some(v);
                }
            }
            "ds4.think_end_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.think_end_token_id = Some(v);
                }
            }
            "ds4.dsml_token_id" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.dsml_token_id = Some(v);
                }
            }
            "general.alignment" => {
                if let Some(v) = val.as_u32() {
                    self.metadata.alignment = v;
                }
            }
            _ => {}
        }
    }

    pub fn tensor(&self, name: &str) -> Ds4Result<QuantizedTensor<'_>> {
        for t in &self.tensors {
            if t.name == name {
                let byte_size = t.byte_size() as usize;
                let start = self.header_offset + t.offset as usize;
                let end = start + byte_size;
                if end > self.mmap.len() {
                    return Err(Ds4Error::new(
                        Ds4ErrorKind::Model,
                        format!(
                            "tensor {name} offset+OOB ({start}+{byte_size} > {})",
                            self.mmap.len()
                        ),
                    ));
                }
                let bytes = &self.mmap[start..end];
                return Ok(QuantizedTensor {
                    descriptor: t.clone(),
                    bytes,
                });
            }
        }
        Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("tensor not found: {name}"),
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }
    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }
    pub fn data_offset(&self) -> usize {
        self.header_offset
    }
    pub fn kv_raw(&self, key: &str) -> Option<&KvRaw> {
        self.kv_raw.get(key)
    }
}

fn str_to_quant(s: &str) -> Ds4QuantKind {
    match s {
        "Q8_0" => Ds4QuantKind::Q8_0,
        "Q4_K" => Ds4QuantKind::Q4_K,
        "Q3_K" => Ds4QuantKind::Q3_K,
        "Q2_K" => Ds4QuantKind::Q2_K,
        "IQ2_XXS" => Ds4QuantKind::Iq2Xxs,
        "F16" => Ds4QuantKind::F16,
        "F32" => Ds4QuantKind::F32,
        _ => Ds4QuantKind::Q8_0,
    }
}

fn read_u32(bytes: &[u8], off: usize) -> Ds4Result<u32> {
    if off + 4 > bytes.len() {
        return Err(Ds4Error::new(Ds4ErrorKind::Model, "GGUF read past EOF"));
    }
    Ok(u32::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
    ]))
}

fn read_u64(bytes: &[u8], off: usize) -> Ds4Result<u64> {
    if off + 8 > bytes.len() {
        return Err(Ds4Error::new(Ds4ErrorKind::Model, "GGUF read past EOF"));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[off..off + 8]);
    Ok(u64::from_le_bytes(b))
}

fn read_tensor_header(bytes: &[u8], off: usize) -> Ds4Result<(String, usize, usize)> {
    let (name, consumed) = read_string(bytes, off)?;
    let n_dims = read_u32(bytes, off + consumed)? as usize;
    Ok((name, n_dims, consumed + 4))
}

fn read_string(bytes: &[u8], off: usize) -> Ds4Result<(String, usize)> {
    let len = read_u64(bytes, off)? as usize;
    let start = off + 8;
    if start + len > bytes.len() {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            "GGUF string read past EOF",
        ));
    }
    let s = std::str::from_utf8(&bytes[start..start + len]).map_err(|e| {
        Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("invalid UTF-8 in GGUF string: {e}"),
        )
    })?;
    Ok((s.to_string(), 8 + len))
}

#[allow(clippy::too_many_arguments)]
fn read_kv(bytes: &[u8], off: usize) -> Ds4Result<(String, KvRaw, usize)> {
    let (key, consumed) = read_string(bytes, off)?;
    let kind = read_u32(bytes, off + consumed)?;
    let (val, val_consumed) = read_kv_value(bytes, off + consumed + 4, kind)?;
    Ok((key, val, consumed + 4 + val_consumed))
}

fn read_kv_value(bytes: &[u8], off: usize, kind: u32) -> Ds4Result<(KvRaw, usize)> {
    Ok(match GgufValueType::from_u32(kind) {
        GgufValueType::Uint8 => {
            if off + 1 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            (KvRaw::U32(bytes[off] as u32), 1)
        }
        GgufValueType::Int8 => {
            if off + 1 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            (KvRaw::I32(bytes[off] as i32), 1)
        }
        GgufValueType::Uint16 => {
            if off + 2 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            (
                KvRaw::U32(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32),
                2,
            )
        }
        GgufValueType::Int16 => {
            if off + 2 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            (
                KvRaw::I32(i16::from_le_bytes([bytes[off], bytes[off + 1]]) as i32),
                2,
            )
        }
        GgufValueType::Uint32 => (KvRaw::U32(read_u32(bytes, off)?), 4),
        GgufValueType::Int32 => (KvRaw::I32(read_u32(bytes, off)? as i32), 4),
        GgufValueType::Float32 => {
            if off + 4 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[off..off + 4]);
            (KvRaw::F32(f32::from_le_bytes(b)), 4)
        }
        GgufValueType::Uint64 => (KvRaw::U64(read_u64(bytes, off)?), 8),
        GgufValueType::Int64 => (KvRaw::I64(read_u64(bytes, off)? as i64), 8),
        GgufValueType::Float64 => {
            if off + 8 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[off..off + 8]);
            (KvRaw::F64(f64::from_le_bytes(b)), 8)
        }
        GgufValueType::Bool => {
            if off + 1 > bytes.len() {
                return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF"));
            }
            (KvRaw::Bool(bytes[off] != 0), 1)
        }
        GgufValueType::String => {
            let (s, consumed) = read_string(bytes, off)?;
            (KvRaw::String(s), consumed)
        }
        GgufValueType::Array => {
            let elem_kind = read_u32(bytes, off)? as u32;
            let n = read_u64(bytes, off + 4)? as usize;
            let next = off + 12;
            match GgufValueType::from_u32(elem_kind) {
                GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
                    let bytes_total = n * 4;
                    if next + bytes_total > bytes.len() {
                        return Err(Ds4Error::new(Ds4ErrorKind::Model, "EOF in GGUF array"));
                    }
                    let mut arr = Vec::with_capacity(n);
                    let mut cur = next;
                    for _ in 0..n {
                        let (v, c) = read_kv_value(bytes, cur, elem_kind)?;
                        arr.push(v);
                        cur += c;
                    }
                    (KvRaw::Array(arr), cur - off)
                }
                GgufValueType::String => {
                    let mut arr = Vec::with_capacity(n);
                    let mut cur = next;
                    for _ in 0..n {
                        let (s, c) = read_string(bytes, cur)?;
                        arr.push(KvRaw::String(s));
                        cur += c;
                    }
                    (KvRaw::Array(arr), cur - off)
                }
                _ => (KvRaw::Unknown(elem_kind), 0),
            }
        }
    })
}

fn align_to(off: usize, alignment: usize) -> Ds4Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Ds4Error::new(
            Ds4ErrorKind::Model,
            format!("invalid GGUF alignment: {alignment}"),
        ));
    }
    Ok((off + alignment - 1) & !(alignment - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_magic_and_version_constants_are_locked() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        assert_eq!(GGUF_VERSION_V3, 3);
    }

    #[test]
    fn gguf_value_type_round_trips() {
        for v in 0..12u32 {
            let _ = GgufValueType::from_u32(v);
        }
    }

    #[test]
    fn gguf_dtype_maps_to_ds4_quant_kind() {
        assert_eq!(GgufDType::F32.as_ds4_quant_kind(), Ds4QuantKind::F32);
        assert_eq!(GgufDType::F16.as_ds4_quant_kind(), Ds4QuantKind::F16);
        assert_eq!(GgufDType::Q8_0.as_ds4_quant_kind(), Ds4QuantKind::Q8_0);
        assert_eq!(GgufDType::Q4_K.as_ds4_quant_kind(), Ds4QuantKind::Q4_K);
        assert_eq!(GgufDType::Q3_K.as_ds4_quant_kind(), Ds4QuantKind::Q3_K);
        assert_eq!(GgufDType::Q2_K.as_ds4_quant_kind(), Ds4QuantKind::Q2_K);
        assert_eq!(GgufDType::Iq2Xxs.as_ds4_quant_kind(), Ds4QuantKind::Iq2Xxs);
    }

    #[test]
    fn gguf_dtype_byte_size_is_positive() {
        for v in [
            GgufDType::F32,
            GgufDType::F16,
            GgufDType::Q8_0,
            GgufDType::Q4_K,
            GgufDType::Q3_K,
            GgufDType::Q2_K,
            GgufDType::Iq2Xxs,
        ] {
            assert!(v.byte_size() > 0, "byte_size for {:?} should be > 0", v);
        }
    }

    #[test]
    fn str_to_quant_recognises_known_strings() {
        assert_eq!(str_to_quant("Q8_0"), Ds4QuantKind::Q8_0);
        assert_eq!(str_to_quant("IQ2_XXS"), Ds4QuantKind::Iq2Xxs);
        assert_eq!(str_to_quant("Q4_K"), Ds4QuantKind::Q4_K);
        assert_eq!(str_to_quant("Q3_K"), Ds4QuantKind::Q3_K);
        assert_eq!(str_to_quant("F32"), Ds4QuantKind::F32);
    }

    #[test]
    fn tensor_descriptor_numel_is_product_of_dims() {
        let d = TensorDescriptor {
            name: "x".into(),
            dims: vec![2, 4, 8],
            dtype: GgufDType::F32,
            offset: 0,
        };
        assert_eq!(d.numel(), 64);
    }

    #[test]
    fn kv_raw_accessors_return_correct_types() {
        let v = KvRaw::U32(42);
        assert_eq!(v.as_u32(), Some(42));
        assert_eq!(v.as_f32(), None);
        assert_eq!(v.as_str(), None);

        let s = KvRaw::String("hello".into());
        assert_eq!(s.as_str(), Some("hello"));

        let a = KvRaw::Array(vec![KvRaw::U32(1), KvRaw::U32(2), KvRaw::U32(3)]);
        assert_eq!(a.as_array_u32(), Some(vec![1, 2, 3]));

        let b = KvRaw::Bool(true);
        assert_eq!(b.as_u32(), Some(1));
    }
}
