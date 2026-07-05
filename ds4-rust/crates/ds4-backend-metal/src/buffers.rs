// DS4 (DwarfStar) -- Metal backend typed buffers.
//
// Buffers here model `MTLBuffer`s. In production, allocation goes
// through `MTLDevice::newBufferWithLength:options:`. In this
// host-side implementation, we use the same byte-vector representation as the
// CUDA backend.

use parking_lot::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Q4_K,
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
}

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
        DType::Q8_0 => 36,
        DType::Q4_K => 144,
        DType::Q2_K => 84,
        DType::Iq2Xxs => 66,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_basic() {
        let pool = BufferPool::new();
        let b = pool.alloc(DType::F32, 16);
        assert_eq!(b.len, 16);
        pool.release(b);
    }

    #[test]
    fn dtype_elem_matches() {
        assert_eq!(dtype_elem(DType::F32), 4);
        assert_eq!(dtype_elem(DType::F16), 2);
    }
}
