// DS4 (DwarfStar) — imatrix binary file format.
//
// The upstream C code (`ds4.c:20073-20268`) writes a llama.cpp-style
// `.dat` file with the following layout (little-endian, all `i32`
// fields are native-width on the writer):
//
//   1. i32 entry_count   (== N_LAYERS * 3 in the reference writer)
//   2. For each entry:
//        i32 name_len
//        <name_len> bytes  (raw UTF-8)
//        i32 ncall         (= 1)
//        i32 nval          (= n_expert * n_col)
//        <nval> floats     (for each expert: count==0 → fill 1.0;
//                           else mean(sum2 / count) over n_col cells)
//   3. i32 chunks
//   4. i32 dataset_len
//   5. <dataset_len> bytes
//
// This module is the Rust port of that layout: `parse` reconstructs
// `Vec<ImatrixEntry>` from raw bytes; `to_bytes` produces the
// inverse.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};
use thiserror::Error;

fn write_i32<W: Write>(out: &mut W, v: i32) -> std::io::Result<()> {
    out.write_i32::<LittleEndian>(v)
}

fn write_f32<W: Write>(out: &mut W, v: f32) -> std::io::Result<()> {
    out.write_f32::<LittleEndian>(v)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImatrixEntry {
    pub tensor_name: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Error)]
pub enum ImatrixFormatError {
    #[error("imatrix: unexpected EOF while reading {what}")]
    Eof { what: &'static str },
    #[error("imatrix: name length {len} is negative")]
    NegativeNameLength { len: i32 },
    #[error("imatrix: name length {len} exceeds buffer ({remaining} bytes left)")]
    NameLengthOverflow { len: i32, remaining: usize },
    #[error("imatrix: entry count {count} is negative")]
    NegativeEntryCount { count: i32 },
    #[error("imatrix: nval {nval} not divisible by ncol ({ncol})")]
    BadNval { nval: i32, ncol: i32 },
    #[error("imatrix: dataset length {len} is negative")]
    NegativeDatasetLength { len: i32 },
    #[error("imatrix: dataset length {len} exceeds buffer ({remaining} bytes left)")]
    DatasetLengthOverflow { len: i32, remaining: usize },
    #[error("imatrix: io error: {0}")]
    Io(#[from] std::io::Error),
}

fn read_i32(c: &mut Cursor<&[u8]>, what: &'static str) -> Result<i32, ImatrixFormatError> {
    c.read_i32::<LittleEndian>()
        .map_err(|_| ImatrixFormatError::Eof { what })
}

fn read_f32(c: &mut Cursor<&[u8]>, what: &'static str) -> Result<f32, ImatrixFormatError> {
    c.read_f32::<LittleEndian>()
        .map_err(|_| ImatrixFormatError::Eof { what })
}

/// Parse an imatrix `.dat` file into its entries.
pub fn parse(bytes: &[u8]) -> Result<Vec<ImatrixEntry>, ImatrixFormatError> {
    let mut c = Cursor::new(bytes);
    let entry_count = read_i32(&mut c, "entry_count")?;
    if entry_count < 0 {
        return Err(ImatrixFormatError::NegativeEntryCount { count: entry_count });
    }
    let mut out: Vec<ImatrixEntry> = Vec::with_capacity(entry_count as usize);
    for _ in 0..entry_count {
        let name_len = read_i32(&mut c, "name_len")?;
        if name_len < 0 {
            return Err(ImatrixFormatError::NegativeNameLength { len: name_len });
        }
        let name_len = name_len as usize;
        let remaining = bytes.len().saturating_sub(c.position() as usize);
        if name_len > remaining {
            return Err(ImatrixFormatError::NameLengthOverflow {
                len: name_len as i32,
                remaining,
            });
        }
        let mut name_buf = vec![0u8; name_len];
        c.read_exact(&mut name_buf)?;
        let tensor_name = String::from_utf8_lossy(&name_buf).into_owned();
        let _ncall = read_i32(&mut c, "ncall")?;
        let nval = read_i32(&mut c, "nval")?;
        if nval < 0 {
            return Err(ImatrixFormatError::BadNval { nval, ncol: 1 });
        }
        let nval = nval as usize;
        let mut values = Vec::with_capacity(nval);
        for _ in 0..nval {
            values.push(read_f32(&mut c, "value")?);
        }
        out.push(ImatrixEntry {
            tensor_name,
            values,
        });
    }
    // Trailer is read but not returned from `parse` — the dataset
    // name and chunk count are diagnostics, not part of the
    // importance matrix itself. We still validate the trailer is
    // present and well-formed so corrupt files surface as errors.
    let _chunks = read_i32(&mut c, "chunks")?;
    let dataset_len = read_i32(&mut c, "dataset_len")?;
    if dataset_len < 0 {
        return Err(ImatrixFormatError::NegativeDatasetLength { len: dataset_len });
    }
    let dataset_len = dataset_len as usize;
    let remaining = bytes.len().saturating_sub(c.position() as usize);
    if dataset_len > remaining {
        return Err(ImatrixFormatError::DatasetLengthOverflow {
            len: dataset_len as i32,
            remaining,
        });
    }
    let mut dataset_buf = vec![0u8; dataset_len];
    c.read_exact(&mut dataset_buf)?;
    Ok(out)
}

/// Parse just the trailer (chunks + dataset path) after a successful
/// `parse`. Returns `(chunks, dataset)`.
pub fn parse_trailer(bytes: &[u8]) -> Result<(i32, String), ImatrixFormatError> {
    let mut c = Cursor::new(bytes);
    let entry_count = read_i32(&mut c, "entry_count")?;
    if entry_count < 0 {
        return Err(ImatrixFormatError::NegativeEntryCount { count: entry_count });
    }
    for _ in 0..entry_count {
        let name_len = read_i32(&mut c, "name_len")?;
        if name_len < 0 {
            return Err(ImatrixFormatError::NegativeNameLength { len: name_len });
        }
        let name_len = name_len as usize;
        let mut name_buf = vec![0u8; name_len];
        c.read_exact(&mut name_buf)
            .map_err(|_| ImatrixFormatError::Eof { what: "name" })?;
        let _ncall = read_i32(&mut c, "ncall")?;
        let nval = read_i32(&mut c, "nval")?;
        if nval < 0 {
            return Err(ImatrixFormatError::BadNval { nval, ncol: 1 });
        }
        let nval = nval as usize;
        // Skip past `nval` f32s.
        let to_skip = (nval as u64).saturating_mul(4);
        c.set_position(c.position().saturating_add(to_skip));
    }
    let chunks = read_i32(&mut c, "chunks")?;
    let dataset_len = read_i32(&mut c, "dataset_len")?;
    if dataset_len < 0 {
        return Err(ImatrixFormatError::NegativeDatasetLength { len: dataset_len });
    }
    let dataset_len = dataset_len as usize;
    let mut buf = vec![0u8; dataset_len];
    c.read_exact(&mut buf)
        .map_err(|_| ImatrixFormatError::Eof { what: "dataset" })?;
    let dataset = String::from_utf8_lossy(&buf).into_owned();
    Ok((chunks, dataset))
}

/// Serialize `entries` into the upstream `.dat` layout. Caller
/// supplies the trailer fields.
pub fn to_bytes(entries: &[ImatrixEntry], chunks: i32, dataset: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    write_i32(&mut out, entries.len() as i32).ok();
    for e in entries {
        let name_bytes = e.tensor_name.as_bytes();
        write_i32(&mut out, name_bytes.len() as i32).ok();
        out.write_all(name_bytes).ok();
        write_i32(&mut out, 1).ok(); // ncall
        write_i32(&mut out, e.values.len() as i32).ok(); // nval
        for &v in &e.values {
            write_f32(&mut out, v).ok();
        }
    }
    write_i32(&mut out, chunks).ok();
    let dataset_bytes = dataset.as_bytes();
    write_i32(&mut out, dataset_bytes.len() as i32).ok();
    out.write_all(dataset_bytes).ok();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<ImatrixEntry> {
        vec![
            ImatrixEntry {
                tensor_name: "blk.0.ffn_gate_exps.weight".to_string(),
                values: vec![0.1, 0.2, 0.3, 0.4],
            },
            ImatrixEntry {
                tensor_name: "blk.0.ffn_up_exps.weight".to_string(),
                values: vec![1.0, 1.0, 1.0, 1.0],
            },
            ImatrixEntry {
                tensor_name: "blk.0.ffn_down_exps.weight".to_string(),
                values: vec![0.5; 6],
            },
        ]
    }

    #[test]
    fn roundtrip_preserves_entries_and_trailer() {
        let entries = sample();
        let bytes = to_bytes(&entries, 7, "calib.txt");
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, entries);
        let (chunks, dataset) = parse_trailer(&bytes).unwrap();
        assert_eq!(chunks, 7);
        assert_eq!(dataset, "calib.txt");
    }

    #[test]
    fn roundtrip_empty_dataset() {
        let bytes = to_bytes(&sample(), 0, "");
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.len(), 3);
        let (_, dataset) = parse_trailer(&bytes).unwrap();
        assert!(dataset.is_empty());
    }

    #[test]
    fn parse_rejects_truncated_input() {
        let mut bytes = to_bytes(&sample(), 1, "x");
        bytes.truncate(bytes.len() - 4); // chop last f32
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn parse_rejects_negative_entry_count() {
        let mut bytes = Vec::new();
        write_i32(&mut bytes, -1).ok();
        assert!(parse(&bytes).is_err());
    }
}
