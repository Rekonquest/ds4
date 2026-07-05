// DS4 (DwarfStar) — on-disk payload formats.
//
// The byte layout is a 16-byte fixed header followed by the opaque
// payload bytes:
//
//     u32  magic         // "DSV4" (whole-session) or "DSVL" (per-layer)
//     u32  version       // 2 or 1 respectively
//     u32  u32_fields    // 13 or 14 — header reservation, mirrors upstream
//     u32  payload_bytes // length of the following bytes
//     [u8; payload_bytes]
//
// Magic/version/u32_fields are pinned by `ds4.h` and MUST match
// upstream byte-for-byte so Rust-written and C-written payloads
// interoperate. See `tests::payload_magic_values_are_locked`.

use std::io::{self, Read};

use crate::store::KvError;

pub const DS4_SESSION_PAYLOAD_MAGIC: u32 = 0x3456_5344; // "DSV4"
pub const DS4_SESSION_PAYLOAD_VERSION: u32 = 2;
pub const DS4_SESSION_PAYLOAD_U32_FIELDS: u32 = 13;

pub const DS4_SESSION_LAYER_PAYLOAD_MAGIC: u32 = 0x4c56_5344; // "DSVL"
pub const DS4_SESSION_LAYER_PAYLOAD_VERSION: u32 = 1;
pub const DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS: u32 = 14;

const HEADER_BYTES: usize = 4 * 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionPayload {
    pub magic: u32,
    pub version: u32,
    pub u32_fields: u32,
    pub bytes: Vec<u8>,
}

impl SessionPayload {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            magic: DS4_SESSION_PAYLOAD_MAGIC,
            version: DS4_SESSION_PAYLOAD_VERSION,
            u32_fields: DS4_SESSION_PAYLOAD_U32_FIELDS,
            bytes,
        }
    }

    pub fn header_bytes() -> usize {
        HEADER_BYTES
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.bytes.len());
        out.extend_from_slice(&self.magic.to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.u32_fields.to_le_bytes());
        out.extend_from_slice(&(self.bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.bytes);
        out
    }

    pub fn from_bytes(input: &[u8]) -> Result<Self, KvError> {
        let mut buf = [0u8; HEADER_BYTES];
        if input.len() < HEADER_BYTES {
            return Err(KvError::Truncated);
        }
        buf.copy_from_slice(&input[..HEADER_BYTES]);
        let magic = read_u32_le(&buf, 0)?;
        let version = read_u32_le(&buf, 4)?;
        let u32_fields = read_u32_le(&buf, 8)?;
        let payload_bytes = read_u32_le(&buf, 12)? as usize;

        if magic != DS4_SESSION_PAYLOAD_MAGIC {
            return Err(KvError::BadMagic(magic));
        }
        if version != DS4_SESSION_PAYLOAD_VERSION {
            return Err(KvError::BadVersion(version));
        }
        if u32_fields != DS4_SESSION_PAYLOAD_U32_FIELDS {
            return Err(KvError::BadU32Fields(u32_fields));
        }
        if input.len() < HEADER_BYTES + payload_bytes {
            return Err(KvError::Truncated);
        }
        Ok(Self {
            magic,
            version,
            u32_fields,
            bytes: input[HEADER_BYTES..HEADER_BYTES + payload_bytes].to_vec(),
        })
    }

    /// Stream-decode the payload from any `Read`. Used when reading
    /// large payloads from disk where buffering everything in memory
    /// just to copy it back is wasteful.
    pub fn from_bytes_stream<R: Read>(mut reader: R) -> Result<Self, KvError> {
        let mut header = [0u8; HEADER_BYTES];
        reader.read_exact(&mut header).map_err(map_read)?;
        let magic = read_u32_le(&header, 0)?;
        let version = read_u32_le(&header, 4)?;
        let u32_fields = read_u32_le(&header, 8)?;
        let payload_bytes = read_u32_le(&header, 12)? as usize;

        if magic != DS4_SESSION_PAYLOAD_MAGIC {
            return Err(KvError::BadMagic(magic));
        }
        if version != DS4_SESSION_PAYLOAD_VERSION {
            return Err(KvError::BadVersion(version));
        }
        if u32_fields != DS4_SESSION_PAYLOAD_U32_FIELDS {
            return Err(KvError::BadU32Fields(u32_fields));
        }

        let mut bytes = vec![0u8; payload_bytes];
        reader.read_exact(&mut bytes).map_err(map_read)?;
        Ok(Self {
            magic,
            version,
            u32_fields,
            bytes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPayload {
    pub magic: u32,
    pub version: u32,
    pub u32_fields: u32,
    pub bytes: Vec<u8>,
}

impl LayerPayload {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            magic: DS4_SESSION_LAYER_PAYLOAD_MAGIC,
            version: DS4_SESSION_LAYER_PAYLOAD_VERSION,
            u32_fields: DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS,
            bytes,
        }
    }

    pub fn header_bytes() -> usize {
        HEADER_BYTES
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.bytes.len());
        out.extend_from_slice(&self.magic.to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.u32_fields.to_le_bytes());
        out.extend_from_slice(&(self.bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.bytes);
        out
    }

    pub fn from_bytes(input: &[u8]) -> Result<Self, KvError> {
        let mut buf = [0u8; HEADER_BYTES];
        if input.len() < HEADER_BYTES {
            return Err(KvError::Truncated);
        }
        buf.copy_from_slice(&input[..HEADER_BYTES]);
        let magic = read_u32_le(&buf, 0)?;
        let version = read_u32_le(&buf, 4)?;
        let u32_fields = read_u32_le(&buf, 8)?;
        let payload_bytes = read_u32_le(&buf, 12)? as usize;

        if magic != DS4_SESSION_LAYER_PAYLOAD_MAGIC {
            return Err(KvError::BadMagic(magic));
        }
        if version != DS4_SESSION_LAYER_PAYLOAD_VERSION {
            return Err(KvError::BadVersion(version));
        }
        if u32_fields != DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS {
            return Err(KvError::BadU32Fields(u32_fields));
        }
        if input.len() < HEADER_BYTES + payload_bytes {
            return Err(KvError::Truncated);
        }
        Ok(Self {
            magic,
            version,
            u32_fields,
            bytes: input[HEADER_BYTES..HEADER_BYTES + payload_bytes].to_vec(),
        })
    }

    /// Stream-decode the payload from any `Read`.
    pub fn from_bytes_stream<R: Read>(mut reader: R) -> Result<Self, KvError> {
        let mut header = [0u8; HEADER_BYTES];
        reader.read_exact(&mut header).map_err(map_read)?;
        let magic = read_u32_le(&header, 0)?;
        let version = read_u32_le(&header, 4)?;
        let u32_fields = read_u32_le(&header, 8)?;
        let payload_bytes = read_u32_le(&header, 12)? as usize;

        if magic != DS4_SESSION_LAYER_PAYLOAD_MAGIC {
            return Err(KvError::BadMagic(magic));
        }
        if version != DS4_SESSION_LAYER_PAYLOAD_VERSION {
            return Err(KvError::BadVersion(version));
        }
        if u32_fields != DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS {
            return Err(KvError::BadU32Fields(u32_fields));
        }

        let mut bytes = vec![0u8; payload_bytes];
        reader.read_exact(&mut bytes).map_err(map_read)?;
        Ok(Self {
            magic,
            version,
            u32_fields,
            bytes,
        })
    }
}

fn map_read(e: io::Error) -> KvError {
    if e.kind() == io::ErrorKind::UnexpectedEof {
        KvError::Truncated
    } else {
        KvError::Io(e.to_string())
    }
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, KvError> {
    let word = bytes.get(offset..offset + 4).ok_or(KvError::Truncated)?;
    Ok(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn session_header_is_16_bytes() {
        assert_eq!(SessionPayload::header_bytes(), 16);
        assert_eq!(LayerPayload::header_bytes(), 16);
    }

    #[test]
    fn session_roundtrip_via_stream() {
        let payload = SessionPayload::new(b"hello world".to_vec());
        let bytes = payload.to_bytes();
        let decoded = SessionPayload::from_bytes_stream(Cursor::new(&bytes)).expect("stream");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn layer_roundtrip_via_stream() {
        let payload = LayerPayload::new(b"layer bytes".to_vec());
        let bytes = payload.to_bytes();
        let decoded = LayerPayload::from_bytes_stream(Cursor::new(&bytes)).expect("stream");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn layer_stream_truncated_yields_kv_truncated() {
        // Construct a valid header but chop the body short.
        let payload = LayerPayload::new(vec![1, 2, 3, 4, 5]);
        let mut bytes = payload.to_bytes();
        bytes.truncate(HEADER_BYTES + 2);
        let err = LayerPayload::from_bytes_stream(Cursor::new(&bytes)).unwrap_err();
        assert!(matches!(err, KvError::Truncated));
    }
}
