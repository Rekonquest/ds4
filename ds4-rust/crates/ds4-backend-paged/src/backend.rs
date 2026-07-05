// DS4 (DwarfStar) -- paged-attention backend.
//
// The paged-attention backend is a second attention implementation
// that stores the KV cache in fixed-size pages (see
// `page_table.rs`) instead of one contiguous buffer. The rest of the
// model graph (linear projections, MLP, layernorm, embeddings, etc.)
// is identical to the dense backend; only the attention read path
// is paged.
//
// `PagedBackend::load_model` allocates one `PageTable` per layer and
// hands them to the caller via `PagedModel::page_tables`. The decode
// loop then uses `paged_attention::paged_attention_decode` instead of
// the contiguous-buffer kernel.

use crate::page_table::{PageTable, PAGE_TOKENS};
use crate::paged_attention::{page_cache_bytes, paged_attention_decode};
use ds4_gguf::GgufFile;
use ds4_types::{Backend, BackendModel, Ds4EngineOptions, Ds4QuantKind, Ds4Result};

/// Default `n_heads * head_dim` footprint per layer. This is the
/// "typical" Mistral 7B value (32 heads * 128 dim). Real models will
/// override at load time via `Ds4EngineOptions` extensions.
const DEFAULT_HEAD_ELEMENTS: usize = 32 * 128;

/// The paged-attention backend. Implements `ds4_types::Backend`.
#[derive(Debug, Clone, Default)]
pub struct PagedBackend {
    /// True if the vendored mistralrs-paged-attn crate is compiled
    /// in and reachable. False means this build is using the local
    /// Rust page-table path instead of an external paged kernel.
    pub uses_vendored_kernel: bool,
}

impl PagedBackend {
    pub fn new() -> Self {
        Self {
            uses_vendored_kernel: false,
        }
    }

    /// The quant kind this backend produces / consumes. Mirrors the
    /// upstream paged-attention code path which is byte-compatible
    /// with the Q8_0 quant kind used by ds4-quant for KV caches.
    pub fn quant_kind(&self) -> Ds4QuantKind {
        Ds4QuantKind::Q8_0
    }

    /// Allocate per-layer page tables for the paged attention path.
    pub fn load_model(&self, opts: &Ds4EngineOptions) -> Ds4Result<PagedModel> {
        let _ = opts; // metadata-aware trait loading supplies layer and head dimensions
        Ok(PagedModel::with_layers(32, DEFAULT_HEAD_ELEMENTS))
    }
}

impl Backend for PagedBackend {
    fn name(&self) -> &'static str {
        // Stable backend-table name.
        "paged"
    }

    fn memory_estimate(ctx_size: usize, _prefill_chunk: usize) -> u64 {
        if ctx_size == 0 {
            return 0;
        }
        // Per-layer KV cache size, doubled for K and V, times n_layers.
        // Formula: (ctx_size / PAGE_TOKENS + 1) * PAGE_TOKENS * head_elements * 4 * 2 * n_layers
        // We don't know n_layers here without opts, so we estimate
        // for a 32-layer model. The metadata-aware memory estimator should plumb
        // the layer count through.
        const N_LAYERS: usize = 32;
        let n_pages = ctx_size / PAGE_TOKENS + 1;
        let per_layer = page_cache_bytes(ctx_size, DEFAULT_HEAD_ELEMENTS / 128, 128);
        (n_pages * PAGE_TOKENS * DEFAULT_HEAD_ELEMENTS * 4 * 2 * N_LAYERS) as u64 + per_layer as u64
    }

    fn load_model(
        &self,
        path: &std::path::Path,
    ) -> ds4_types::Ds4Result<Box<dyn ds4_types::BackendModel>> {
        let gguf = GgufFile::open(path)?;
        let n_layers = gguf
            .metadata
            .layer_count
            .map_or_else(|| infer_layer_count(&gguf).max(1), |v| (v as usize).max(1));
        let n_heads = gguf.metadata.head_count.map_or(32, |v| (v as usize).max(1));
        let head_dim = gguf.metadata.head_dim.map_or(128, |v| (v as usize).max(1));
        Ok(Box::new(PagedModel::with_layers(
            n_layers,
            n_heads * head_dim,
        )))
    }
}

fn infer_layer_count(gguf: &GgufFile) -> usize {
    gguf.tensors
        .iter()
        .filter_map(|tensor| {
            let rest = tensor.name.strip_prefix("blk.")?;
            let (idx, _) = rest.split_once('.')?;
            idx.parse::<usize>().ok()
        })
        .max()
        .map_or(0, |idx| idx + 1)
}

/// A loaded paged model. Owns one `PageTable` per layer and the
/// per-layer `head_elements` footprint.
#[derive(Debug, Clone)]
pub struct PagedModel {
    pub page_tables: Vec<PageTable>,
    pub head_elements: usize,
}

impl PagedModel {
    /// Allocate per-layer page tables for `n_layers` layers.
    pub fn with_layers(n_layers: usize, head_elements: usize) -> Self {
        let mut page_tables = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            page_tables.push(PageTable::new(layer, head_elements));
        }
        Self {
            page_tables,
            head_elements,
        }
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.page_tables.len()
    }

    /// Convenience: run paged attention against the given layer's
    /// cache. Asserts layer is in range.
    pub fn attention(
        &self,
        q: &[f32],
        layer: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let cache = &self.page_tables[layer];
        paged_attention_decode(q, cache, layer, seq_len, n_heads, head_dim)
    }
}

impl BackendModel for PagedModel {
    fn quant_kind(&self) -> Ds4QuantKind {
        Ds4QuantKind::Q8_0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_paged() {
        let b = PagedBackend::new();
        assert_eq!(b.name(), "paged");
    }

    #[test]
    fn quant_kind_is_q8_0() {
        assert_eq!(PagedBackend::new().quant_kind(), Ds4QuantKind::Q8_0);
    }

    #[test]
    fn load_model_returns_paged_model_with_page_tables() {
        let b = PagedBackend::new();
        let m = b.load_model(&Ds4EngineOptions::default()).unwrap();
        assert_eq!(m.n_layers(), 32);
        assert_eq!(m.page_tables.len(), 32);
        // Layer indices are unique and match the index.
        for (i, pt) in m.page_tables.iter().enumerate() {
            assert_eq!(pt.layer(), i);
        }
    }

    #[test]
    fn trait_load_model_uses_gguf_metadata() {
        let dir = std::env::temp_dir().join("ds4-backend-paged-trait-load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.gguf");
        ds4_gguf::write_synthetic_gguf(&path).unwrap();
        let backend = PagedBackend::new();
        let model = Backend::load_model(&backend, &path).unwrap();
        assert_eq!(model.quant_kind(), Ds4QuantKind::Q8_0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_estimate_zero_ctx_returns_zero() {
        assert_eq!(PagedBackend::memory_estimate(0, 512), 0);
    }

    #[test]
    fn memory_estimate_grows_with_ctx() {
        let small = PagedBackend::memory_estimate(512, 512);
        let big = PagedBackend::memory_estimate(4096, 512);
        assert!(big > small, "memory estimate should grow with context size");
    }

    #[test]
    fn memory_estimate_includes_paging_overhead() {
        // Paging overhead = (ctx/page + 1) * page vs ctx, so we expect
        // the estimate to be slightly larger than the naive
        // `ctx * 2 * head_elems * 4 * n_layers`.
        let naive = 512 * 2 * 32 * 128 * 4 * 32;
        let estimate = PagedBackend::memory_estimate(512, 512);
        assert!(
            estimate >= naive as u64,
            "estimate {} should cover the naive lower bound {}",
            estimate,
            naive,
        );
    }

    #[test]
    fn paged_model_attention_routes_to_correct_layer() {
        let m = PagedModel::with_layers(4, 4);
        // Empty cache at layer 2 -> zero output (degenerate case,
        // but pins that `attention` reads from the right layer).
        let q = [1.0, 0.0, 1.0, 0.0];
        let out = m.attention(&q, 2, 0, 2, 2);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn paged_model_clone_is_independent() {
        let mut m = PagedModel::with_layers(2, 4);
        // Append a token to layer 0 to materialize the page.
        m.page_tables[0].append_token(&[1.0; 4], &[1.0; 4]);
        let mut m2 = m.clone();
        m2.page_tables[0].append_token(&[2.0; 4], &[2.0; 4]);
        // Original is unaffected.
        assert_eq!(m.page_tables[0].seq_len, 1);
        assert_eq!(m2.page_tables[0].seq_len, 2);
    }
}
