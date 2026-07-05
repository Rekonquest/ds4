// DS4 (DwarfStar) -- CUDA backend typed buffers.
//
// In the real CUDA runtime, `Buffer*` would wrap `cudaMalloc`-allocated
// device memory. Here we wrap plain byte buffers with a free-list
// allocator because the actual kernel compilation lives behind the
// `compile()` function in `kernels.rs`, which gates on `nvcc` being
// on PATH. If `nvcc` is absent, the buffer allocator still works
// correctly for host-side unit tests.

use parking_lot::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Q4_K,
    Q3_K,
    Q2_K,
    Iq2Xxs,
}

#[derive(Debug)]
pub struct Buffer {
    pub dtype: DType,
    pub bytes: Vec<u8>,
    pub len: usize,
    pub capacity: usize,
}

impl Buffer {
    pub fn new(dtype: DType, len: usize) -> Self {
        let elem = dtype_elem(dtype);
        let bytes = vec![0u8; len * elem];
        Self {
            dtype,
            bytes,
            len,
            capacity: len,
        }
    }

    /// Read this buffer as a `Vec<f32>` (safe copy via `from_le_bytes`).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DType::F32);
        self.bytes
            .chunks_exact(4)
            .map(|c| {
                let mut b = [0u8; 4];
                b.copy_from_slice(c);
                f32::from_le_bytes(b)
            })
            .collect()
    }

    pub fn from_f32_slice(data: &[f32], len: usize) -> Self {
        let mut b = Self::new(DType::F32, len);
        for (i, v) in data.iter().enumerate().take(len) {
            let bytes = v.to_le_bytes();
            let off = i * 4;
            b.bytes[off..off + 4].copy_from_slice(&bytes);
        }
        b
    }
}

/// Free-list allocator. Buffers are kept around for reuse to mimic
/// the CUDA stream-ordered allocation pattern.
#[derive(Default)]
pub struct BufferPool {
    free: Mutex<Vec<Buffer>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&self, dtype: DType, len: usize) -> Buffer {
        let mut free = self.free.lock();
        if let Some(idx) = free
            .iter()
            .position(|b| b.dtype == dtype && b.capacity >= len)
        {
            let mut b = free.swap_remove(idx);
            b.len = len;
            for byte in b.bytes[..len * dtype_elem(dtype)].iter_mut() {
                *byte = 0;
            }
            return b;
        }
        Buffer::new(dtype, len)
    }

    pub fn release(&self, b: Buffer) {
        self.free.lock().push(b);
    }
}

pub fn dtype_elem(d: DType) -> usize {
    match d {
        DType::F32 => 4,
        DType::F16 => 2,
        DType::Q8_0 => 36, // f32 d + i8 qs[32]
        DType::Q4_K => 144,
        DType::Q3_K => 110,
        DType::Q2_K => 84,
        DType::Iq2Xxs => 66,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_allocates_and_reuses() {
        let pool = BufferPool::new();
        let b1 = pool.alloc(DType::F32, 16);
        assert_eq!(b1.len, 16);
        let b2 = pool.alloc(DType::F32, 32);
        assert_eq!(b2.len, 32);
        pool.release(b1);
        let b3 = pool.alloc(DType::F32, 8);
        assert_eq!(b3.len, 8);
        assert_eq!(b3.capacity, 16, "should reuse the released buffer");
        pool.release(b2);
        pool.release(b3);
    }

    #[test]
    fn pool_distinguishes_dtypes() {
        let pool = BufferPool::new();
        let a = pool.alloc(DType::F32, 4);
        let b = pool.alloc(DType::F16, 4);
        assert_ne!(dtype_elem(a.dtype), dtype_elem(b.dtype));
    }

    #[test]
    fn buffer_new_zeros_capacity() {
        let b = Buffer::new(DType::F32, 100);
        assert_eq!(b.bytes.len(), 400);
        assert!(b.bytes.iter().all(|&x| x == 0));
    }

    #[test]
    fn f32_roundtrip_safe() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let b = Buffer::from_f32_slice(&data, 4);
        let out = b.to_f32_vec();
        assert_eq!(out, data);
    }

    #[test]
    fn dtype_elem_matches_block_sizes() {
        assert_eq!(dtype_elem(DType::Q8_0), 36);
        assert_eq!(dtype_elem(DType::Q4_K), 144);
        assert_eq!(dtype_elem(DType::Q3_K), 110);
        assert_eq!(dtype_elem(DType::Q2_K), 84);
        assert_eq!(dtype_elem(DType::Iq2Xxs), 66);
    }
}
