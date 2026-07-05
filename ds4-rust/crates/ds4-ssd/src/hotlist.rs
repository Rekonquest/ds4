// DS4 (DwarfStar) -- hotlist parser + histogram.
//
// Parses the upstream `ds4_streaming_hotlist.inc` format:
//   static const uint16_t foo[][2] = {
//     { 0, 1 },
//     { 2, 3 },
//   };
// where each pair is `(layer, expert)`. The histogram aggregates
// duplicate pairs and is used by the SSD streaming module to drive
// prefetch order.
//
// Both `.inc` text and raw little-endian pair bytes are supported.
// The `.inc` format is the upstream form; the bytes form is what
// the C reference pre-loads into the runtime hotlist.

use std::collections::HashMap;

use super::HotlistError;

/// One `(layer, expert)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pair {
    pub layer: u16,
    pub expert: u16,
}

/// Layer × expert histogram.
#[derive(Debug, Clone)]
pub struct Hotlist {
    layers: Vec<Vec<u32>>,
}

impl Hotlist {
    /// Allocate an empty histogram with `n_layers` rows and
    /// `n_experts` columns, all zeros.
    pub fn empty(n_layers: usize, n_experts: usize) -> Self {
        Self {
            layers: vec![vec![0u32; n_experts]; n_layers],
        }
    }

    /// Build from a precomputed pair list. The histogram dimensions
    /// are derived from the maximum layer/expert id seen.
    pub fn from_pairs(n_layers: usize, n_experts: usize, pairs: &[Pair]) -> Self {
        let mut out = Self::empty(n_layers, n_experts);
        for p in pairs {
            if (p.layer as usize) < n_layers && (p.expert as usize) < n_experts {
                out.layers[p.layer as usize][p.expert as usize] += 1;
            }
        }
        out
    }

    /// Parse the upstream `ds4_streaming_hotlist.inc` text format
    /// and return `(pairs, n_layers, n_experts)`. C-style line
    /// (`//`) and block (`/* */`) comments are stripped.
    pub fn from_inc_bytes(bytes: &[u8]) -> Result<(Vec<Pair>, usize, usize), HotlistError> {
        let cleaned =
            strip_c_comments(std::str::from_utf8(bytes).map_err(|e| HotlistError::Parse {
                line: 0,
                message: format!("not valid utf-8: {e}"),
            })?);
        let mut pairs: Vec<Pair> = Vec::new();
        let mut tuple: Vec<i64> = Vec::new();
        let mut n_layers = 1usize;
        let mut n_experts = 1usize;

        for (line_idx, raw_line) in cleaned.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let line = line.trim_end_matches(',').trim_end_matches(';').trim();
            if line == "{"
                || line == "}"
                || line.starts_with("static")
                || line.starts_with("const")
                || line.contains('[')
            {
                continue;
            }
            if line.starts_with("/*") || line.ends_with("*/") {
                continue;
            }
            // Skip C expressions (cast, sizeof, identifiers). Pair rows
            // are always `{ int, int }`; everything else is metadata
            // that the upstream `ds4_streaming_hotlist.inc` includes
            // between tables (e.g. `(uint32_t)(sizeof(...) / sizeof(...))`).
            if line.contains('(') || line.contains(')') {
                continue;
            }
            if !line.chars().any(|c| c.is_ascii_digit()) {
                continue;
            }
            if let Some(stripped) = line.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let inner = stripped.trim();
                for part in inner.split(',') {
                    push_number(&mut tuple, part, line_idx)?;
                }
                commit_pair(
                    &mut pairs,
                    &mut tuple,
                    &mut n_layers,
                    &mut n_experts,
                    line_idx,
                )?;
            } else {
                push_number(&mut tuple, line, line_idx)?;
            }
        }
        Ok((pairs, n_layers, n_experts))
    }

    /// Build a histogram from raw `(layer, expert)` bytes. Each
    /// `layer` is `u16` LE, each `expert` is `u16` LE.
    pub fn from_pairs_bytes(bytes: &[u8]) -> Result<Self, HotlistError> {
        if !bytes.len().is_multiple_of(4) {
            return Err(HotlistError::Parse {
                line: 0,
                message: "pairs bytes length must be a multiple of 4".to_string(),
            });
        }
        let mut pairs: Vec<Pair> = Vec::with_capacity(bytes.len() / 4);
        let mut max_layer = 0u16;
        let mut max_expert = 0u16;
        for chunk in bytes.chunks_exact(4) {
            let layer = u16::from_le_bytes([chunk[0], chunk[1]]);
            let expert = u16::from_le_bytes([chunk[2], chunk[3]]);
            if layer > max_layer {
                max_layer = layer;
            }
            if expert > max_expert {
                max_expert = expert;
            }
            pairs.push(Pair { layer, expert });
        }
        let n_layers = (max_layer as usize).saturating_add(1).max(1);
        let n_experts = (max_expert as usize).saturating_add(1).max(1);
        Ok(Self::from_pairs(n_layers, n_experts, &pairs))
    }

    pub fn get(&self, layer: usize, expert: usize) -> u32 {
        self.layers
            .get(layer)
            .and_then(|l| l.get(expert))
            .copied()
            .unwrap_or(0)
    }

    /// Layers × experts dimensions.
    pub fn shape(&self) -> (usize, usize) {
        let layers = self.layers.len();
        let experts = self.layers.first().map(|l| l.len()).unwrap_or(0);
        (layers, experts)
    }

    /// Top-N experts sorted by hit count, ties broken by layer first
    /// then by expert id. Returns `(layer, expert, hits)`.
    pub fn top(&self, n: usize) -> Vec<(usize, usize, u32)> {
        let mut out: Vec<(usize, usize, u32)> = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            for (e, &hits) in layer.iter().enumerate() {
                if hits > 0 {
                    out.push((l, e, hits));
                }
            }
        }
        out.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        out.truncate(n);
        out
    }

    /// Build a `[(layer, expert, hits)]` index for any caller that
    /// wants to iterate once.
    pub fn as_index(&self) -> HashMap<(u16, u16), u32> {
        let mut out = HashMap::new();
        for (l, layer) in self.layers.iter().enumerate() {
            for (e, &hits) in layer.iter().enumerate() {
                if hits > 0 {
                    out.insert((l as u16, e as u16), hits);
                }
            }
        }
        out
    }

    pub fn as_layers(&self) -> &[Vec<u32>] {
        &self.layers
    }
}

fn push_number(tuple: &mut Vec<i64>, num: &str, line: usize) -> Result<(), HotlistError> {
    let trimmed = num.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let value: i64 = trimmed.parse().map_err(|e| HotlistError::Parse {
        line,
        message: format!("cannot parse {trimmed:?}: {e}"),
    })?;
    tuple.push(value);
    Ok(())
}

fn commit_pair(
    pairs: &mut Vec<Pair>,
    tuple: &mut Vec<i64>,
    n_layers: &mut usize,
    n_experts: &mut usize,
    line: usize,
) -> Result<(), HotlistError> {
    if tuple.len() < 2 {
        tuple.clear();
        return Ok(());
    }
    let layer = tuple[0];
    let expert = tuple[1];
    if layer < 0 || expert < 0 || layer > u16::MAX as i64 || expert > u16::MAX as i64 {
        return Err(HotlistError::Parse {
            line,
            message: format!("pair out of u16 range: ({layer}, {expert})"),
        });
    }
    let lu = layer as u16;
    let eu = expert as u16;
    pairs.push(Pair {
        layer: lu,
        expert: eu,
    });
    if (lu as usize) >= *n_layers {
        *n_layers = (lu as usize) + 1;
    }
    if (eu as usize) >= *n_experts {
        *n_experts = (eu as usize) + 1;
    }
    tuple.clear();
    Ok(())
}

fn strip_c_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    out.push(b'\n');
                }
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            } else {
                i = bytes.len();
            }
        } else if bytes[i] == b'"' {
            out.push(bytes[i]);
            i += 1;
            while i < bytes.len() {
                out.push(bytes[i]);
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    out.push(bytes[i + 1]);
                    i += 2;
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_inc_array() {
        let src = b"static const uint16_t foo[][2] = {\n    {0, 1},\n    {2, 3},\n    {0, 1},\n};";
        let (pairs, n_layers, n_experts) = Hotlist::from_inc_bytes(src).unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(n_layers, 3);
        assert_eq!(n_experts, 4);
        let h = Hotlist::from_pairs(n_layers, n_experts, &pairs);
        assert_eq!(h.get(0, 1), 2);
        assert_eq!(h.get(2, 3), 1);
        assert_eq!(h.get(0, 0), 0);
    }

    #[test]
    fn strips_c_comments() {
        let src = b"/* leading */ static const uint16_t x[][2] = { // hi\n{4,5}, /* end */\n};";
        let (pairs, _, _) = Hotlist::from_inc_bytes(src).unwrap();
        assert_eq!(
            pairs,
            vec![Pair {
                layer: 4,
                expert: 5
            }]
        );
    }

    #[test]
    fn top_returns_highest_first() {
        let h = Hotlist::from_pairs(
            2,
            4,
            &[
                Pair {
                    layer: 0,
                    expert: 0,
                },
                Pair {
                    layer: 0,
                    expert: 0,
                },
                Pair {
                    layer: 0,
                    expert: 0,
                },
                Pair {
                    layer: 0,
                    expert: 1,
                },
                Pair {
                    layer: 0,
                    expert: 1,
                },
                Pair {
                    layer: 1,
                    expert: 3,
                },
            ],
        );
        let top = h.top(3);
        assert_eq!(top[0], (0, 0, 3));
        assert_eq!(top[1], (0, 1, 2));
        assert_eq!(top[2], (1, 3, 1));
    }

    #[test]
    fn parses_real_hotlist_inc() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join("ds4_streaming_hotlist.inc"));
        if let Some(path) = path {
            if let Ok(bytes) = std::fs::read(&path) {
                let (pairs, n_layers, n_experts) = Hotlist::from_inc_bytes(&bytes).unwrap();
                assert!(pairs.len() > 1000, "expected many pairs in real hotlist");
                assert!(n_layers >= 1);
                assert!(n_experts >= 1);
            }
        }
    }
}
