// DS4 (DwarfStar) -- paged-attention page table.
//
// The KV cache for a single transformer layer is split into fixed-size
// "pages" instead of one contiguous buffer. Each page holds
// `PAGE_TOKENS` tokens of K and V vectors for a single layer.
//
// Why paged?
//   * Pages can be on GPU, CPU, or disk — memory is granular.
//   * Two sequences that share a system-prompt prefix can share the
//     same pages (no copy required).
//   * Sparse retrieval -- only the pages inside the current query
//     window are touched at decode time.
//
// The layout of a `Page` matches what the attention kernel needs:
// a contiguous `[PAGE_TOKENS, n_heads * head_dim]` buffer (allocated
// lazily when the page is first written to, so empty pages cost zero
// memory until they are touched).

use std::collections::HashMap;

/// Default number of tokens per page. The v1 reference implementation
/// uses 16; the vendored `mistralrs-paged-attn` exposes this as a
/// build-time constant and the C source uses 16 as well.
pub const PAGE_TOKENS: usize = 16;

/// A single paged-attention page.
///
/// Holds up to `PAGE_TOKENS` K vectors and `PAGE_TOKENS` V vectors
/// for one transformer layer. Both buffers are stored row-major as
/// `[PAGE_TOKENS, n_heads * head_dim]` -- that is the layout the
/// attention kernel expects.
///
/// `n_tokens` is the number of *valid* entries; the unused tail
/// slots are left as zeros and must be masked by the attention
/// kernel. `layer` identifies which transformer layer this page
/// belongs to (the page table is shared across layers but every
/// page is tagged with its layer for sanity checking).
#[derive(Debug, Clone)]
pub struct Page {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub layer: usize,
    pub n_tokens: usize,
    /// Per-page token stride in elements. The K/V buffers have
    /// `n_heads * head_dim` elements per token.
    pub head_elements: usize,
}

impl Page {
    /// Allocate an empty page for the given layer. The K/V buffers
    /// are left at zero length until `set_head_elements` is called,
    /// which keeps `PageTable::new` cheap.
    pub fn empty(layer: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            layer,
            n_tokens: 0,
            head_elements: 0,
        }
    }

    /// Lazily size the K/V buffers once we know the layer's
    /// `n_heads * head_dim` footprint. Idempotent.
    pub fn ensure_capacity(&mut self, head_elements: usize) {
        let needed = PAGE_TOKENS * head_elements;
        if self.k.len() != needed {
            self.k.resize(needed, 0.0);
        }
        if self.v.len() != needed {
            self.v.resize(needed, 0.0);
        }
        if self.head_elements == 0 {
            self.head_elements = head_elements;
        }
    }

    /// Write a single (k, v) pair at `token_offset_in_page`. Returns
    /// `true` if the page was full *before* the write (so the caller
    /// knows to allocate the next page), `false` otherwise.
    pub fn write_token(&mut self, token_offset_in_page: usize, k: &[f32], v: &[f32]) -> bool {
        assert!(
            token_offset_in_page < PAGE_TOKENS,
            "paged_attention: token_offset_in_page {} out of bounds (PAGE_TOKENS = {})",
            token_offset_in_page,
            PAGE_TOKENS,
        );
        assert_eq!(
            k.len(),
            self.head_elements,
            "paged_attention: k.len() ({}) != page.head_elements ({})",
            k.len(),
            self.head_elements,
        );
        assert_eq!(v.len(), self.head_elements);
        let was_full = self.n_tokens == PAGE_TOKENS;
        let base = token_offset_in_page * self.head_elements;
        self.k[base..base + self.head_elements].copy_from_slice(k);
        self.v[base..base + self.head_elements].copy_from_slice(v);
        if token_offset_in_page >= self.n_tokens {
            self.n_tokens = token_offset_in_page + 1;
        }
        was_full
    }
}

/// The page table for one (logical) sequence. The struct owns a
/// `Vec<Option<Page>>` indexed by page id and a `free_list` of
/// page indices that have been freed and are ready for re-use.
///
/// `seq_len` is the *logical* sequence length (number of tokens
/// stored across all pages for this sequence), maintained alongside
/// the page table so `lookup` can be answered in O(1) amortized.
#[derive(Debug, Clone, Default)]
pub struct PageTable {
    pub pages: Vec<Option<Page>>,
    pub free_list: Vec<usize>,
    pub seq_len: usize,
    /// Optional `head_elements` for the layer this page table is
    /// attached to. Set by the backend when the model is loaded.
    head_elements: usize,
    /// Which transformer layer this page table belongs to.
    layer: usize,
    /// `pages -> Vec<Option<Page>>` reverse map for prefix sharing:
    /// when a caller wants to attach an existing page id to a new
    /// sequence at offset 0, it can clone the `Page` rather than
    /// allocating a new one. Tracked here only to support
    /// `share_prefix`; lookups go straight through `pages`.
    share_count: HashMap<usize, usize>,
}

impl PageTable {
    /// Construct an empty page table for a single layer.
    pub fn new(layer: usize, head_elements: usize) -> Self {
        Self {
            pages: Vec::new(),
            free_list: Vec::new(),
            seq_len: 0,
            head_elements,
            layer,
            share_count: HashMap::new(),
        }
    }

    /// The layer this page table is attached to.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// `n_heads * head_dim` -- the per-token K/V footprint.
    pub fn head_elements(&self) -> usize {
        self.head_elements
    }

    /// Set the per-token footprint after construction. Any already-
    /// allocated pages are resized on their next write.
    pub fn set_head_elements(&mut self, head_elements: usize) {
        self.head_elements = head_elements;
    }

    /// Allocate a fresh page for `layer` and return its index. The
    /// returned index is always the *next sequential* slot
    /// (`pages.len()` before the push) -- `append_token` relies on
    /// `seq_len / PAGE_TOKENS` matching the returned index, so the
    /// free list is *not* consulted here. The free list is kept
    /// for the explicit `free` API and compaction passes
    /// that want to reclaim memory without changing the sequence
    /// layout. Newly allocated pages are materialized with
    /// `Some(Page::empty(layer))` sized to `head_elements`.
    pub fn alloc(&mut self) -> usize {
        let idx = self.pages.len();
        let mut page = Page::empty(self.layer);
        page.ensure_capacity(self.head_elements);
        self.pages.push(Some(page));
        idx
    }

    /// Free a page index, returning it to the free list. The slot
    /// in `pages` is replaced with `None` and the page's K/V buffers
    /// are dropped.
    pub fn free(&mut self, idx: usize) {
        if idx >= self.pages.len() {
            return;
        }
        if self.pages[idx].is_some() {
            self.pages[idx] = None;
            self.free_list.push(idx);
            self.share_count.remove(&idx);
        }
    }

    /// Returns the page index and the offset-within-page for the
    /// token at logical `token_pos` (0-based). `None` if the token
    /// is past `seq_len` or the slot is not materialized.
    pub fn lookup(&self, token_pos: usize) -> Option<(usize, usize)> {
        if token_pos >= self.seq_len {
            return None;
        }
        let page_idx = token_pos / PAGE_TOKENS;
        let offset = token_pos % PAGE_TOKENS;
        let slot = self.pages.get(page_idx)?;
        slot.as_ref().map(|_| (page_idx, offset))
    }

    /// Append a token to the end of the sequence, allocating a new
    /// page if the page slot at `page_idx` is empty. Returns the
    /// page idx and offset where the token was written.
    pub fn append_token(&mut self, k: &[f32], v: &[f32]) -> (usize, usize) {
        if self.head_elements == 0 && !k.is_empty() {
            self.head_elements = k.len();
        }
        assert_eq!(
            k.len(),
            self.head_elements,
            "paged_attention: k.len() ({}) != page_table.head_elements ({})",
            k.len(),
            self.head_elements,
        );
        assert_eq!(v.len(), self.head_elements);

        let page_idx = self.seq_len / PAGE_TOKENS;
        let offset = self.seq_len % PAGE_TOKENS;

        // Allocate the page slot only if it isn't already
        // materialized. `share_prefix_from` may have populated the
        // slot out-of-band; in that case we just reuse it.
        let needs_alloc = !matches!(self.pages.get(page_idx), Some(Some(_)));
        if needs_alloc {
            let new_idx = self.alloc();
            debug_assert_eq!(
                new_idx, page_idx,
                "paged_attention: alloc returned {} but expected sequential {} -- append_token must follow the page slot ordering",
                new_idx, page_idx,
            );
        }

        let page = self.pages[page_idx]
            .as_mut()
            .expect("paged_attention: page slot unexpectedly empty after alloc");
        page.ensure_capacity(self.head_elements);
        let was_full = page.write_token(offset, k, v);
        self.seq_len += 1;
        let _ = was_full;
        (page_idx, offset)
    }

    /// Share an existing page at the current tail of the sequence.
    /// Used to mirror a prefix from one sequence onto another
    /// without copying K/V. The shared page must be logically
    /// "before" the tail -- i.e. it must be the next sequential slot
    /// (`seq_len / PAGE_TOKENS`) and it must be `n_tokens`-full (so
    /// `seq_len` advances to exactly `(idx + 1) * PAGE_TOKENS`).
    ///
    /// Returns the new page index in this table. Updates `seq_len`
    /// to reflect the appended prefix; the caller must then drive
    /// the rest of the sequence with `append_token`.
    pub fn share_prefix_from(&mut self, src_page: Page) -> usize {
        assert_eq!(
            src_page.layer, self.layer,
            "paged_attention: share_prefix_from layer mismatch ({} vs {})",
            src_page.layer, self.layer,
        );
        let expected_idx = self.seq_len / PAGE_TOKENS;
        let offset_within = self.seq_len % PAGE_TOKENS;
        assert_eq!(
            offset_within, 0,
            "paged_attention: share_prefix_from requires seq_len aligned to a page boundary (got offset {})",
            offset_within,
        );

        let idx = self.alloc();
        assert_eq!(
            idx, expected_idx,
            "paged_attention: share_prefix_from returned {} but expected sequential {}",
            idx, expected_idx,
        );

        let n_tokens = src_page.n_tokens;
        assert_eq!(
            n_tokens, PAGE_TOKENS,
            "paged_attention: share_prefix_from requires a full page ({} tokens), got {}",
            PAGE_TOKENS, n_tokens,
        );
        let head_elements = src_page.head_elements;
        if let Some(slot) = self.pages.get_mut(idx) {
            *slot = Some(src_page);
            *self.share_count.entry(idx).or_insert(0) += 1;
        }
        if self.head_elements == 0 {
            self.head_elements = head_elements;
        }
        self.seq_len += n_tokens;
        idx
    }

    /// Number of materialized pages (i.e. pages whose `Option` is
    /// `Some`). Equal to `pages.len() - free_list.len()` once
    /// bookkeeping is consistent (the free list tracks freed slots
    /// so this is exact).
    pub fn n_materialized(&self) -> usize {
        self.pages.iter().filter(|p| p.is_some()).count()
    }

    /// Total number of pages allocated in the underlying storage
    /// (materialized + freed).
    pub fn n_slots(&self) -> usize {
        self.pages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head_elements() -> usize {
        // 2 heads * 2 dim = 4 elements per token.
        4
    }

    #[test]
    fn alloc_returns_distinct_indices_until_free() {
        let mut t = PageTable::new(0, head_elements());
        let a = t.alloc();
        let b = t.alloc();
        assert_ne!(a, b);
        assert_eq!(t.n_materialized(), 2);
        assert_eq!(t.n_slots(), 2);
    }

    #[test]
    fn free_then_alloc_does_not_recycle_index_in_v1() {
        let mut t = PageTable::new(0, head_elements());
        let a = t.alloc();
        let b = t.alloc();
        t.free(a);
        let c = t.alloc();
        assert_ne!(c, a, "v1 alloc does not recycle freed indices");
        assert_eq!(c, b + 1, "next sequential slot");
        assert!(c != b);
        assert_eq!(t.n_materialized(), 2);
        assert_eq!(t.n_slots(), 3);
    }

    #[test]
    fn free_of_unknown_index_is_a_noop() {
        let mut t = PageTable::new(0, head_elements());
        t.alloc();
        t.free(999);
        assert_eq!(t.n_materialized(), 1);
    }

    #[test]
    fn lookup_returns_none_past_seq_len() {
        let mut t = PageTable::new(0, head_elements());
        assert!(
            t.lookup(0).is_none(),
            "no tokens appended yet -> seq_len = 0"
        );
        t.append_token(&[0.0; 4], &[0.0; 4]);
        assert_eq!(t.lookup(0), Some((0, 0)));
        assert!(t.lookup(1).is_none(), "only one token stored");
    }

    #[test]
    fn append_token_spans_pages() {
        let mut t = PageTable::new(0, head_elements());
        for i in 0..(PAGE_TOKENS + 3) {
            let k = [i as f32, 0.0, 0.0, 0.0];
            let v = [0.0, i as f32, 0.0, 0.0];
            t.append_token(&k, &v);
        }
        assert_eq!(t.seq_len, PAGE_TOKENS + 3);
        assert_eq!(t.lookup(0), Some((0, 0)));
        assert_eq!(t.lookup(PAGE_TOKENS - 1), Some((0, PAGE_TOKENS - 1)));
        assert_eq!(t.lookup(PAGE_TOKENS), Some((1, 0)));
        assert_eq!(t.lookup(PAGE_TOKENS + 2), Some((1, 2)));
        assert!(t.lookup(PAGE_TOKENS + 3).is_none());
        let (page_idx, offset) = t.lookup(2).unwrap();
        let page = t.pages[page_idx].as_ref().unwrap();
        assert_eq!(page.k[offset * page.head_elements], 2.0);
    }

    #[test]
    fn share_prefix_does_not_reallocate_kv() {
        let mut a = PageTable::new(0, head_elements());
        for i in 0..PAGE_TOKENS {
            a.append_token(&[i as f32, 0.0, 0.0, 0.0], &[0.0; 4]);
        }
        assert_eq!(a.seq_len, PAGE_TOKENS);
        assert_eq!(a.n_materialized(), 1);

        let mut b = PageTable::new(0, head_elements());
        let shared_idx = {
            let p0 = a.pages[0].as_ref().unwrap().clone();
            b.share_prefix_from(p0)
        };
        assert_eq!(shared_idx, 0);
        assert_eq!(b.seq_len, PAGE_TOKENS);
        assert_eq!(b.n_materialized(), 1);

        b.append_token(&[99.0; 4], &[99.0; 4]);
        assert_eq!(b.seq_len, PAGE_TOKENS + 1);
        assert_eq!(b.n_materialized(), 2);
        assert_eq!(a.seq_len, PAGE_TOKENS);
        assert_eq!(a.n_materialized(), 1);
        let shared = b.pages[shared_idx].as_ref().unwrap();
        assert_eq!(shared.n_tokens, PAGE_TOKENS);
        assert_eq!(shared.k[0], 0.0);
        let fresh = b.pages[1].as_ref().unwrap();
        assert_eq!(fresh.n_tokens, 1);
        assert_eq!(fresh.k[0], 99.0);
    }

    #[test]
    fn share_prefix_must_be_full_page() {
        let mut src = PageTable::new(0, head_elements());
        src.append_token(&[1.0; 4], &[1.0; 4]);
        let mut dst = PageTable::new(0, head_elements());
        let p = src.pages[0].as_ref().unwrap().clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dst.share_prefix_from(p);
        }));
        assert!(
            result.is_err(),
            "share_prefix_from must reject partial pages"
        );
    }

    #[test]
    fn page_ensure_capacity_is_idempotent() {
        let mut p = Page::empty(0);
        p.ensure_capacity(8);
        let first = p.k.len();
        p.ensure_capacity(8);
        assert_eq!(
            p.k.len(),
            first,
            "ensure_capacity must not reallocate on repeat calls"
        );
    }

    #[test]
    fn freed_page_yields_none_on_lookup() {
        let mut t = PageTable::new(0, head_elements());
        t.append_token(&[1.0; 4], &[1.0; 4]);
        assert_eq!(t.seq_len, 1);
        let idx = 0;
        t.free(idx);
        let r = t.lookup(0);
        assert!(r.is_none(), "freed slot must not return data: got {:?}", r);
        t.free(idx);
        assert!(t.lookup(0).is_none());
    }
}
