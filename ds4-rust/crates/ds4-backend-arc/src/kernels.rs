pub const ARC_KERNEL_SRC: &str = r#"
#define DS4_DTYPE_F32 0
#define DS4_DTYPE_F16 1
#define DS4_DTYPE_Q8_0 2
#define DS4_DTYPE_Q4_K 3
#define DS4_DTYPE_Q3_K 4
#define DS4_DTYPE_Q2_K 5

ushort ds4_u16(__global const uchar* p) {
    return ((ushort)p[0]) | ((ushort)p[1] << 8);
}

uint ds4_u32(__global const uchar* p) {
    return ((uint)p[0]) | ((uint)p[1] << 8) |
           ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

float ds4_f32(__global const uchar* p) {
    return as_float(ds4_u32(p));
}

float ds4_f16(ushort h) {
    uint sign = ((uint)h & 0x8000U) << 16;
    int exp = (int)((h >> 10) & 0x1fU);
    uint mant = (uint)h & 0x03ffU;
    if (exp == 0) {
        if (mant == 0) {
            return as_float(sign);
        }
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            exp -= 1;
        }
        exp += 1;
        mant &= ~0x0400U;
    } else if (exp == 31) {
        return as_float(sign | 0x7f800000U | (mant << 13));
    }
    exp = exp + (127 - 15);
    return as_float(sign | ((uint)exp << 23) | (mant << 13));
}

void ds4_scale_min_k4(int j, __global const uchar* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

int ds4_q3_scale(__global const uchar* scales, int j) {
    int low = (j < 8) ? (scales[j] & 0x0f) : ((scales[j - 8] >> 4) & 0x0f);
    int high = (scales[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
    return (low | (high << 4)) - 32;
}

float ds4_load_q8_0(__global const uchar* data, ulong index) {
    ulong block = index >> 5;
    int r = (int)(index & 31UL);
    __global const uchar* b = data + block * 34UL;
    float d = ds4_f16(ds4_u16(b));
    char q = (char)b[2 + r];
    return d * (float)q;
}

float ds4_load_q4_k(__global const uchar* data, ulong index) {
    ulong block = index >> 8;
    int r = (int)(index & 255UL);
    __global const uchar* b = data + block * 144UL;
    float d = ds4_f16(ds4_u16(b));
    float dmin = ds4_f16(ds4_u16(b + 2));
    __global const uchar* scales = b + 4;
    __global const uchar* qs = b + 16;
    int group = r >> 6;
    int lane = r & 63;
    int sc = 0;
    int mn = 0;
    ds4_scale_min_k4(group * 2 + (lane >= 32), scales, &sc, &mn);
    uchar packed = qs[group * 32 + (lane & 31)];
    int q = (lane < 32) ? (packed & 0x0f) : (packed >> 4);
    return d * (float)sc * (float)q - dmin * (float)mn;
}

float ds4_load_q3_k(__global const uchar* data, ulong index) {
    ulong block = index >> 8;
    int r = (int)(index & 255UL);
    __global const uchar* b = data + block * 110UL;
    __global const uchar* hmask = b;
    __global const uchar* qs = b + 32;
    __global const uchar* scales = b + 96;
    float d = ds4_f16(ds4_u16(b + 108));
    int part = r >> 7;
    int rem = r & 127;
    int j = rem >> 5;
    int within = rem & 31;
    int scale_idx = part * 8 + j * 2 + (within >= 16);
    int q_off = part * 32 + within;
    int shift = j * 2;
    int q = (qs[q_off] >> shift) & 3;
    int mask = 1 << (part * 4 + j);
    int bias = (hmask[within] & mask) ? 0 : 4;
    return d * (float)ds4_q3_scale(scales, scale_idx) * (float)(q - bias);
}

float ds4_load_q2_k(__global const uchar* data, ulong index) {
    ulong block = index >> 8;
    int r = (int)(index & 255UL);
    __global const uchar* b = data + block * 84UL;
    __global const uchar* scales = b;
    __global const uchar* qs = b + 16;
    float d = ds4_f16(ds4_u16(b + 80));
    float dmin = ds4_f16(ds4_u16(b + 82));
    int part = r >> 7;
    int rem = r & 127;
    int pair = rem >> 5;
    int within = rem & 31;
    int scale_idx = part * 8 + pair * 2 + (within >= 16);
    int shift = pair * 2;
    int q = (qs[part * 32 + within] >> shift) & 3;
    int sc = scales[scale_idx];
    return d * (float)(sc & 0x0f) * (float)q - dmin * (float)(sc >> 4);
}

float ds4_load_weight(__global const uchar* data, int dtype, ulong index) {
    if (dtype == DS4_DTYPE_F32) {
        return ds4_f32(data + index * 4UL);
    }
    if (dtype == DS4_DTYPE_F16) {
        return ds4_f16(ds4_u16(data + index * 2UL));
    }
    if (dtype == DS4_DTYPE_Q8_0) {
        return ds4_load_q8_0(data, index);
    }
    if (dtype == DS4_DTYPE_Q4_K) {
        return ds4_load_q4_k(data, index);
    }
    if (dtype == DS4_DTYPE_Q3_K) {
        return ds4_load_q3_k(data, index);
    }
    if (dtype == DS4_DTYPE_Q2_K) {
        return ds4_load_q2_k(data, index);
    }
    return 0.0f;
}

__kernel void ds4_embedding_weight(
    __global const uchar* token_embd,
    const int dtype,
    const uint token,
    const int hidden,
    __global float* out
) {
    int i = get_global_id(0);
    if (i < hidden) {
        out[i] = ds4_load_weight(token_embd, dtype, (ulong)token * (ulong)hidden + (ulong)i);
    }
}

__kernel void ds4_rmsnorm_weight(
    __global const float* input,
    __global const uchar* weight,
    const int dtype,
    const int n,
    const float eps,
    __global float* out
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = input[i];
        sum_sq += v * v;
    }
    float inv_rms = rsqrt(sum_sq / (float)n + eps);
    int i = get_global_id(0);
    if (i < n) {
        out[i] = input[i] * inv_rms * ds4_load_weight(weight, dtype, (ulong)i);
    }
}

__kernel void ds4_matvec_weight(
    __global const float* input,
    __global const uchar* weight,
    const int dtype,
    const int in_dim,
    const int out_dim,
    __global float* out
) {
    int j = get_global_id(0);
    if (j >= out_dim) {
        return;
    }
    float acc = 0.0f;
    for (int p = 0; p < in_dim; ++p) {
        acc += input[p] * ds4_load_weight(weight, dtype, (ulong)p * (ulong)out_dim + (ulong)j);
    }
    out[j] = acc;
}

__kernel void ds4_embedding_f32(
    __global const float* token_embd,
    const uint token,
    const int hidden,
    __global float* out
) {
    int i = get_global_id(0);
    if (i < hidden) {
        out[i] = token_embd[(ulong)token * (ulong)hidden + (ulong)i];
    }
}

__kernel void ds4_rmsnorm_f32(
    __global const float* input,
    __global const float* weight,
    const int n,
    const float eps,
    __global float* out
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = input[i];
        sum_sq += v * v;
    }
    float inv_rms = rsqrt(sum_sq / (float)n + eps);
    int i = get_global_id(0);
    if (i < n) {
        out[i] = input[i] * inv_rms * weight[i];
    }
}

__kernel void ds4_matvec_f32(
    __global const float* input,
    __global const float* weight,
    const int in_dim,
    const int out_dim,
    __global float* out
) {
    int j = get_global_id(0);
    if (j >= out_dim) {
        return;
    }
    float acc = 0.0f;
    for (int p = 0; p < in_dim; ++p) {
        acc += input[p] * weight[(ulong)p * (ulong)out_dim + (ulong)j];
    }
    out[j] = acc;
}

__kernel void ds4_add_inplace_f32(
    __global float* dst,
    __global const float* src,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        dst[i] += src[i];
    }
}

__kernel void ds4_add_scaled_inplace_f32(
    __global float* dst,
    __global const float* src,
    const float scale,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        dst[i] += src[i] * scale;
    }
}

__kernel void ds4_silu_product_f32(
    __global float* gate,
    __global const float* up,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + exp(-g))) * up[i];
    }
}

__kernel void ds4_rope_f32(
    __global float* x,
    const int pos,
    const int n_heads,
    const int head_dim,
    const float freq_base
) {
    int pair = get_global_id(0);
    int rotary_pairs = head_dim / 2;
    int total = n_heads * rotary_pairs;
    if (pair >= total) {
        return;
    }
    int h = pair / rotary_pairs;
    int i = pair - h * rotary_pairs;
    int off = h * head_dim + i * 2;
    float exponent = (2.0f * (float)i) / (float)head_dim;
    float theta = (float)pos / pow(freq_base, exponent);
    float s = sin(theta);
    float c = cos(theta);
    float a = x[off];
    float b = x[off + 1];
    x[off] = a * c - b * s;
    x[off + 1] = a * s + b * c;
}

__kernel void ds4_store_cache_f32(
    __global float* cache,
    __global const float* src,
    const int offset,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        cache[offset + i] = src[i];
    }
}

__kernel void ds4_attention_decode_f32(
    __global const float* q,
    __global const float* k_cache,
    __global const float* v_cache,
    const int prefix_len,
    const int n_heads,
    const int head_dim,
    __global float* out
) {
    int idx = get_global_id(0);
    int total = n_heads * head_dim;
    if (idx >= total) {
        return;
    }
    int h = idx / head_dim;
    int d_out = idx - h * head_dim;
    float scale = rsqrt((float)head_dim);
    float max_score = -3.402823466e+38f;
    for (int t = 0; t < prefix_len; ++t) {
        float score = 0.0f;
        ulong base = ((ulong)t * (ulong)n_heads + (ulong)h) * (ulong)head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q[(ulong)h * (ulong)head_dim + (ulong)d] * k_cache[base + (ulong)d];
        }
        score *= scale;
        max_score = fmax(max_score, score);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int t = 0; t < prefix_len; ++t) {
        float score = 0.0f;
        ulong base = ((ulong)t * (ulong)n_heads + (ulong)h) * (ulong)head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q[(ulong)h * (ulong)head_dim + (ulong)d] * k_cache[base + (ulong)d];
        }
        float w = exp(score * scale - max_score);
        denom += w;
        acc += w * v_cache[base + (ulong)d_out];
    }
    out[idx] = acc / denom;
}
"#;
