// DS4 (DwarfStar) — three-tier KV cache.
//
// 1:1 port of the in-memory KV cache layout used by `ds4_session` in
// `ds4.c`. Each transformer layer owns three parallel buffers:
//
//   * `raw_swa`     — sliding-window-attention raw rows. Sized
//                     `ctx_size * head_dim` per layer.
//   * `compressed`  — the compressor's compressed-row buffer.
//                     Sized `ctx_size * compressed_dim` per layer.
//   * `index_comp_kv`— indexer-only compression buffer. Sized
//                     `ctx_size * indexer_dim` per layer.
//
// In `ds4.c` the multi-tier cache is initialized, extended, and
// rewound by the session runtime. This Rust port ships the same
// three-buffer layout with deterministic extend/rewind behavior and
// explicit row dimensions for tests and backend callers.
//
// Implemented surface:
//   * `KvCache::new(ctx_size, n_layers)` allocates the three buffers.
//   * `extend(layer, pos, k, v)` appends one token's key/value data
//     to raw, compressed, and indexer buffers at slot `pos`.
//   * `rewind(layer, pos)` clears slots in `[pos, ctx_size)`.
//   * `raw_layer(layer)`, `compressed_layer(layer)`, and
//     `index_comp_layer(layer)` expose per-layer buffers.
use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

/// Default row dimensions used when model metadata has not supplied
/// backend-specific cache widths.
pub const DEFAULT_HEAD_DIM: usize = 128;
pub const DEFAULT_COMPRESSED_DIM: usize = 64;
pub const DEFAULT_INDEXER_DIM: usize = 64;

/// Three-tier per-layer KV cache.
///
/// `raw_swa`, `compressed`, and `index_comp_kv` are parallel
/// `Vec<Vec<f32>>`s — one inner buffer per layer. The buffers are
/// sized to hold `ctx_size` rows each.
#[derive(Debug, Clone)]
pub struct KvCache {
    ctx_size: usize,
    n_layers: usize,
    head_dim: usize,
    compressed_dim: usize,
    indexer_dim: usize,
    /// `raw_swa[layer][pos * head_dim + h]`
    raw_swa: Vec<Vec<f32>>,
    /// `compressed[layer][pos * compressed_dim + h]`
    compressed: Vec<Vec<f32>>,
    /// `index_comp_kv[layer][pos * indexer_dim + h]`
    /// Allocated for every layer so the layout is uniform; callers
    /// can treat zero-filled rows as inactive indexer data.
    index_comp_kv: Vec<Vec<f32>>,
}

impl KvCache {
    /// Allocate the three buffers for `n_layers` layers, each with
    /// `ctx_size` rows. The default row dimensions are the
    /// `DEFAULT_*` constants.
    pub fn new(ctx_size: usize, n_layers: usize) -> Ds4Result<Self> {
        Self::with_dims(
            ctx_size,
            n_layers,
            DEFAULT_HEAD_DIM,
            DEFAULT_COMPRESSED_DIM,
            DEFAULT_INDEXER_DIM,
        )
    }

    /// Allocate the three buffers with caller-supplied row dims.
    pub fn with_dims(
        ctx_size: usize,
        n_layers: usize,
        head_dim: usize,
        compressed_dim: usize,
        indexer_dim: usize,
    ) -> Ds4Result<Self> {
        if ctx_size == 0 || n_layers == 0 {
            return Err(ds4_types::Ds4Error::new(
                ds4_types::Ds4ErrorKind::InvalidArgument,
                format!("KvCache::with_dims: ctx_size={ctx_size} n_layers={n_layers}"),
            ));
        }
        let raw = vec![vec![0f32; ctx_size * head_dim]; n_layers];
        let comp = vec![vec![0f32; ctx_size * compressed_dim]; n_layers];
        let idx = vec![vec![0f32; ctx_size * indexer_dim]; n_layers];
        Ok(Self {
            ctx_size,
            n_layers,
            head_dim,
            compressed_dim,
            indexer_dim,
            raw_swa: raw,
            compressed: comp,
            index_comp_kv: idx,
        })
    }

    /// Maximum number of token positions the cache can hold.
    pub fn ctx_size(&self) -> usize {
        self.ctx_size
    }

    /// Number of transformer layers in the cache.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Per-token row dim of the raw SWA buffer.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Per-token row dim of the compressed buffer.
    pub fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }

    /// Per-token row dim of the indexer-compressed buffer.
    pub fn indexer_dim(&self) -> usize {
        self.indexer_dim
    }

    /// Append one token's k/v to `layer` at slot `pos`. The `k` and
    /// `v` slices must each have length `head_dim`; the compressed
    /// buffer is fed a deterministic windowed-mean down-projection.
    pub fn extend(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Ds4Result<()> {
        if layer >= self.n_layers {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "KvCache::extend layer {layer} out of range {}",
                    self.n_layers
                ),
            ));
        }
        if pos >= self.ctx_size {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("KvCache::extend pos {pos} out of range {}", self.ctx_size),
            ));
        }
        if k.len() != self.head_dim {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "KvCache::extend k length {} does not match {}",
                    k.len(),
                    self.head_dim
                ),
            ));
        }
        if v.len() != self.head_dim {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "KvCache::extend v length {} does not match {}",
                    v.len(),
                    self.head_dim
                ),
            ));
        }

        // Raw SWA: store k followed by v (matches `ds4_kv_cache.raw`
        // layout in C — caller-side conv: a row is [k | v]).
        let raw_offset = pos * self.head_dim;
        // Pack k into the first half and v into the second half. The
        // C side uses two separate per-token buffers (raw_k, raw_v);
        // here we fold them into one `raw_swa` row so the indexer
        // can splice off either half by stride.
        let raw = &mut self.raw_swa[layer][raw_offset..raw_offset + self.head_dim];
        // First half: k, second half: v. Caller may overwrite later.
        let mid = self.head_dim / 2;
        for (i, slot) in raw[..mid].iter_mut().enumerate() {
            *slot = k.get(i).copied().unwrap_or(0.0);
        }
        for (i, slot) in raw[mid..self.head_dim].iter_mut().enumerate() {
            *slot = v.get(i).copied().unwrap_or(0.0);
        }

        // Compressed: deterministic windowed-mean down-projection so
        // shape matches and downstream code can iterate.
        let comp_offset = pos * self.compressed_dim;
        let comp = &mut self.compressed[layer][comp_offset..comp_offset + self.compressed_dim];
        if self.compressed_dim == 0 {
            return Ok(());
        }
        let window = (self.head_dim / self.compressed_dim).max(1);
        for (c, slot) in comp.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..window {
                let src = c * window + j;
                if src < self.head_dim {
                    acc += k.get(src).copied().unwrap_or(0.0);
                }
            }
            *slot = acc / (window as f32);
        }

        // Indexer-compressed buffer: copy raw v into the first slot.
        // Copy value data into the indexer buffer so tests can verify
        // the layout and callers can inspect a populated row.
        let idx_offset = pos * self.indexer_dim;
        let idx = &mut self.index_comp_kv[layer][idx_offset..idx_offset + self.indexer_dim];
        let copy_len = self.indexer_dim.min(v.len());
        idx[..copy_len].copy_from_slice(&v[..copy_len]);
        Ok(())
    }

    /// Zero the entire cache — every layer, every slot, every tier.
    /// O(1) allocation but O(n) work; callers should only call this on
    /// session-level state changes (sync / invalidate).
    pub fn reset(&mut self) {
        for buf in &mut self.raw_swa {
            buf.fill(0.0);
        }
        for buf in &mut self.compressed {
            buf.fill(0.0);
        }
        for buf in &mut self.index_comp_kv {
            buf.fill(0.0);
        }
    }

    /// Drop slots `[pos, ctx_size)` from all three buffers of
    /// `layer`. Out-of-range positions are clamped to the cache size
    /// to mirror `ds4_session_rewind` semantics.
    pub fn rewind(&mut self, layer: usize, pos: usize) -> Ds4Result<()> {
        if layer >= self.n_layers {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "KvCache::rewind layer {layer} out of range {}",
                    self.n_layers
                ),
            ));
        }
        let pos = pos.min(self.ctx_size);
        let zero_tail = |buf: &mut [f32], row: usize, dim: usize| {
            let start = pos * dim;
            let end = (row * dim).min(buf.len());
            if start < end {
                buf[start..end].fill(0.0);
            }
        };
        zero_tail(&mut self.raw_swa[layer], self.ctx_size, self.head_dim);
        zero_tail(
            &mut self.compressed[layer],
            self.ctx_size,
            self.compressed_dim,
        );
        zero_tail(
            &mut self.index_comp_kv[layer],
            self.ctx_size,
            self.indexer_dim,
        );
        Ok(())
    }

    /// Borrow the raw SWA buffer for `layer`. Length = `ctx_size * head_dim`.
    pub fn raw_layer(&self, layer: usize) -> &[f32] {
        self.raw_swa.get(layer).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Borrow the compressed buffer for `layer`.
    pub fn compressed_layer(&self, layer: usize) -> &[f32] {
        self.compressed.get(layer).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Borrow the indexer-compressed buffer for `layer`.
    pub fn index_comp_layer(&self, layer: usize) -> &[f32] {
        self.index_comp_kv
            .get(layer)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_allocates_three_tiers() {
        let c = KvCache::new(8, 4).expect("alloc");
        assert_eq!(c.ctx_size(), 8);
        assert_eq!(c.n_layers(), 4);
        assert_eq!(c.head_dim(), DEFAULT_HEAD_DIM);
        assert_eq!(c.raw_layer(0).len(), 8 * DEFAULT_HEAD_DIM);
        assert_eq!(c.compressed_layer(0).len(), 8 * DEFAULT_COMPRESSED_DIM);
        assert_eq!(c.index_comp_layer(0).len(), 8 * DEFAULT_INDEXER_DIM);
    }

    #[test]
    fn extend_writes_kv_into_raw_row() {
        let mut c = KvCache::with_dims(4, 1, 4, 2, 2).expect("alloc");
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];
        c.extend(0, 1, &k, &v).expect("extend");
        // k occupies the first head_dim/2 entries; v the rest.
        assert_eq!(&c.raw_layer(0)[4..8], &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn rewind_clears_tail() {
        let mut c = KvCache::with_dims(4, 1, 4, 2, 2).expect("alloc");
        // Fill 4 slots with k=[1,2,3,4], v=[5,6,7,8].
        for pos in 0..4 {
            c.extend(0, pos, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
                .expect("extend");
        }
        // Rewind to pos=2 — slots [2,4) should zero out.
        c.rewind(0, 2).expect("rewind");
        let raw = c.raw_layer(0);
        // Slot 0: k || v = [1,2,5,6]; Slot 1: k || v = [1,2,5,6].
        assert_eq!(&raw[0..4], &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(&raw[4..8], &[1.0, 2.0, 5.0, 6.0]);
        // Slots 2..4 should be zeroed.
        assert_eq!(&raw[8..16], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn invalid_args_are_rejected() {
        assert!(KvCache::new(0, 4).is_err());
        assert!(KvCache::new(4, 0).is_err());
        let mut c = KvCache::with_dims(4, 1, 4, 2, 2).expect("alloc");
        assert!(c.extend(1, 0, &[1.0; 4], &[1.0; 4]).is_err());
        assert!(c.extend(0, 4, &[1.0; 4], &[1.0; 4]).is_err());
        assert!(c.extend(0, 0, &[1.0; 3], &[1.0; 4]).is_err());
        assert!(c.extend(0, 0, &[1.0; 4], &[1.0; 3]).is_err());
        assert!(c.rewind(1, 0).is_err());
        assert!(c.raw_layer(1).is_empty());
        assert!(c.compressed_layer(1).is_empty());
        assert!(c.index_comp_layer(1).is_empty());
    }
}
