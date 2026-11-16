#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include "../util.h"
#include "../util.cuh"
#include "../triton_kernel.h"
#include "rdna3_attn.h"

// Flash-decoding split kernel for AMD RDNA3 (gfx11, wave32), a drop-in for the Triton
// _paged_attn_decode_split_kernel on the graphed decode/verify path (bc_attn.py): same launch
// arguments (plus Triton's two hidden scratch pointers), same program/split decomposition and
// the same partial_o / partial_ml layout, so the Triton combine kernel reduces its output.
//
// Per (kv head, split) block of 4 waves; each wave walks 32-token blocks of the split:
// - scores: lane <-> token. The lane reads its token's key row, fp16 or 8-bit (midpoint grid,
//   fp16 scale per 32 values, rotated domain); q rows are LDS broadcasts; v_dot2 per 32-group
// - online softmax with a wave-uniform running max per row
// - values: lane <-> dims (8 per lane at head_dim 256), p broadcast through LDS, fp32 FMAs
// - the four waves merge through LDS, then one partial per (row, split) is stored
//
// Scope: causal, no window / softcap / sinks, q_len <= 16, head_dim 128 or 256, cache fp16 or
// quantized (KB-bit K, VB-bit V, 2..8 each, see rdna3_attn_split_kernel for the instantiated pairs),
// softmax scale 1/sqrt(head_dim). Other configurations keep Triton.
//
// Quantized cache layout (cache/q_cache_kernels.cuh, read by triton_paged._qc_load_kt): a group of 32
// values owns B uint32 words holding the power-of-two bit planes B is made of (8, 4, 2, 1; widest and most
// significant first), plane W storing value j in word j * W / 32 at bit (j * W) % 32. A value is
// (raw - (2^(B-1) - 0.5)) * scale / 2^(B-1) in the Hadamard-rotated domain. B = 8 is one byte per value.
//
// Adapted from exllamav3-rocm (github.com/phoenixhaxor/exllamav3-rocm, MIT).

#ifdef __HIP_PLATFORM_AMD__

namespace rdna3_attn {

constexpr int PAGE = 256;
constexpr int TB = 32;              // tokens per wave block

typedef _Float16 hv2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false);
}

// Two unsigned bytes of w (selected by v_perm; bytes 4-7 of the pair are 0x64) as a half2 of (b - MID):
// 0x64bb is 1024 + b
template <int MID>
__device__ __forceinline__ uint32_t bytes_to_h2c(uint32_t w, uint32_t sel)
{
    uint32_t h = __builtin_amdgcn_perm(0x64646464u, w, sel);
    half2 v = *reinterpret_cast<half2*>(&h);
    v = __hsub2(v, __float2half2_rn((float) (1024 + MID)));
    return *reinterpret_cast<uint32_t*>(&v);
}

// The B words of one 32-value group, from a pointer that need not be 16-byte aligned (B * 4 bytes per group)
template <int B>
__device__ __forceinline__ void qc_load_words(const uint32_t* p, uint32_t* w)
{
    if constexpr (B % 4 == 0)
    {
        #pragma unroll
        for (int i = 0; i < B / 4; ++i) { const uint4 v = ((const uint4*) p)[i]; w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w; }
    }
    else if constexpr (B % 2 == 0)
    {
        #pragma unroll
        for (int i = 0; i < B / 2; ++i) { const uint2 v = ((const uint2*) p)[i]; w[2 * i] = v.x; w[2 * i + 1] = v.y; }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < B; ++i) w[i] = p[i];
    }
}

// The 32 raw values (0 .. 2^B - 1) of a group as 8 words of 4 bytes, value 4i + k in byte k of word i
template <int B>
__device__ __forceinline__ void qc_unpack32(const uint32_t* w, uint32_t* out)
{
    #pragma unroll
    for (int o = 0; o < 8; ++o)
    {
        uint32_t word = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const int j = o * 4 + k;
            uint32_t raw = 0;
            int wb = 0;
            #pragma unroll
            for (int W = 8; W >= 1; W >>= 1)
                if (B & W)
                {
                    raw = (raw << W) | ((w[wb + (j * W) / 32] >> ((j * W) % 32)) & ((1u << W) - 1u));
                    wb += W;
                }
            word |= raw << (8 * k);
        }
        out[o] = word;
    }
}

// The DPL consecutive raw values o0 .. o0 + DPL - 1 of a group (o0 a multiple of DPL), V's lane <-> dims layout:
// accumulates one plane at a time, widest first
template <int B, int DPL, int W>
__device__ __forceinline__ void qc_v_plane(const uint32_t* vw, int o0, int& wb, uint32_t* raw)
{
    if constexpr (B & W)
    {
        constexpr int NW = (DPL * W + 31) / 32;
        const int wi = wb + (o0 * W) / 32;
        uint64_t ww = vw[wi];
        if constexpr (NW > 1) ww |= (uint64_t) vw[wi + 1] << 32;
        const int sh = (o0 * W) % 32;
        #pragma unroll
        for (int k = 0; k < DPL; ++k)
            raw[k] = (raw[k] << W) | (uint32_t) ((ww >> (sh + k * W)) & ((1u << W) - 1u));
        wb += W;
    }
}

template <int B, int DPL>
__device__ __forceinline__ void qc_v_raw(const uint32_t* vw, int o0, uint32_t* raw)
{
    #pragma unroll
    for (int k = 0; k < DPL; ++k) raw[k] = 0;
    int wb = 0;
    qc_v_plane<B, DPL, 8>(vw, o0, wb, raw);
    qc_v_plane<B, DPL, 4>(vw, o0, wb, raw);
    qc_v_plane<B, DPL, 2>(vw, o0, wb, raw);
    qc_v_plane<B, DPL, 1>(vw, o0, wb, raw);
}

__device__ __forceinline__ float wave_max(float v)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v = fmaxf(v, __shfl_xor(v, i));
    return v;
}

__device__ __forceinline__ float wave_sum(float v)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v += __shfl_xor(v, i);
    return v;
}

}  // namespace rdna3_attn

// Block layout: QL == 1 (decode): TG = 4 waves split the tokens, each computes all GROUP rows,
// merged through LDS. QL > 1 (draft verification): one wave per query position (GROUP rows each)
// over all tokens of the split, no merge; the program for h_block 0 of each kv head computes every
// row and stores the partials of all its h_blocks, the other h_block programs exit (K/V read once)
template <int QL> struct AttnGeom
{
    static constexpr int TG = QL == 1 ? 4 : 1;       // token groups (waves) per row group
    static constexpr int RG = QL;                    // row groups (waves): one per query position
    static constexpr int WAVES = TG * RG;
};

template <int KB, int VB, int HD, int QL, int GROUP>
__global__ __launch_bounds__(AttnGeom<QL>::WAVES * 32)
void rdna3_attn_decode_split_kernel
(
    const half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,
    void* scratch0,
    void* scratch1
)
{
    using namespace rdna3_attn;
    constexpr int TG = AttnGeom<QL>::TG;
    constexpr int WAVES = AttnGeom<QL>::WAVES;
    constexpr int NTH = WAVES * 32;
    constexpr int BM = QL <= 1 ? 1 : QL <= 2 ? 2 : QL <= 4 ? 4 : QL <= 8 ? 8 : 16;
    constexpr int BH = 16 / BM;
    constexpr int BLOCK_ROWS = QL * GROUP <= 16 ? 16 : 32;   // partial layout rows (decode_row_layout)
    constexpr int RW = GROUP;                                // rows per wave (one query position)
    constexpr int NROWS = QL * GROUP;                        // rows of this kv head, query-major
    constexpr int G = HD / 32;                               // 32-value groups per head
    constexpr int DPL = HD / 32;                             // value dims per lane
    constexpr int H_BLOCKS = (QL * GROUP + BLOCK_ROWS - 1) / BLOCK_ROWS;
    constexpr int KMID = KB > 0 ? 1 << (KB - 1) : 1;         // the quantized grid's half range: raw - KMID + 0.5 is a unit-step value
    constexpr int VMID = VB > 0 ? 1 << (VB - 1) : 1;

    __shared__ uint32_t qs[NROWS][HD / 2];       // q rows as half2 (rotated for quantized keys)
    __shared__ float qsum[NROWS][G];             // per-group sums of qs (midpoint correction)
    __shared__ float pbuf[WAVES][TB][RW];        // softmax weights of the current block
    __shared__ float accm[TG > 1 ? RW : 1][TG > 1 ? HD : 1];    // cross-wave merge (decode)
    __shared__ float mw[WAVES][RW], lw[WAVES][RW];

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int h_block = pid % H_BLOCKS;
    if (h_block != 0) return;
    const int bh = pid / H_BLOCKS;
    const int NKV = (int) (intptr_t) scratch0;   // kv heads, bound at kernel creation (rdna3_attn_split_kernel)
    const int NQH = NKV * GROUP;
    const int batch = bh / NKV;
    const int kvh = bh % NKV;

    const int total_k_len = cache_seqlens[batch] + QL;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Stage q: one thread per (row, 32-group); 32-point Hadamard for rotated (8-bit) keys
    for (int task = threadIdx.x; task < NROWS * G; task += NTH)
    {
        const int gr = task / G, g = task % G;
        const int qi = gr / GROUP, hl = gr % GROUP;
        float x[32];
        const half* qp = q + ((size_t) (batch * QL + qi) * NQH + kvh * GROUP + hl) * HD + g * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) x[i] = __half2float(qp[i]);
        if constexpr (KB > 0)
        {
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1)
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] *= 0.17677669529663687f;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            half2 v = __floats2half2_rn(x[2 * i], x[2 * i + 1]);
            s += __low2float(v) + __high2float(v);
            qs[gr][g * 16 + i] = *reinterpret_cast<uint32_t*>(&v);
        }
        qsum[gr][g] = s;
    }
    if constexpr (TG > 1)
        for (int i = threadIdx.x; i < RW * HD; i += NTH) (&accm[0][0])[i] = 0.0f;
    __syncthreads();

    const float sm_scale = rsqrtf((float) HD) / (float) (KB > 0 ? KMID : 1);

    const int rg = warp / TG;                      // row group = query position
    const int tg = warp % TG;                      // token group
    const int row0 = rg * RW;
    const int q_abs = total_k_len - QL + rg;       // causal bound of this wave's rows

    float m[RW], l[RW], acc[RW][DPL], psum[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        m[r] = -INFINITY; l[r] = 0.0f; psum[r] = 0.0f;
        #pragma unroll
        for (int j = 0; j < DPL; ++j) acc[r][j] = 0.0f;
    }

    const size_t row_elems = (size_t) NKV * HD;      // per token row: all kv heads
    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const int t_last = min(n_end, q_abs + 1);        // causal: no keys past this wave's position

    for (int b0 = n_start + tg * TB; b0 < t_last; b0 += TG * TB)
    {
        const int t = b0 + lane;
        const bool tvalid = t < t_last;
        const int tc = tvalid ? t : t_last - 1;
        const int tok_row = bt[tc / PAGE] * PAGE + (tc % PAGE);
        const size_t kbase = (size_t) tok_row * row_elems + (size_t) kvh * HD;
        [[maybe_unused]] const size_t kword = (size_t) tok_row * (NKV * G * (KB > 0 ? KB : 1)) + (size_t) kvh * (G * (KB > 0 ? KB : 1));

        // Scores for this lane's token
        float s[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) s[r] = 0.0f;

        #pragma unroll 1
        for (int g = 0; g < G; ++g)
        {
            uint32_t kw[16];
            float gs = 1.0f;
            if constexpr (KB > 0)
            {
                uint32_t pk[KB > 0 ? KB : 1], w8[8];
                rdna3_attn::qc_load_words<KB>((const uint32_t*) k_cache + kword + g * KB, pk);
                rdna3_attn::qc_unpack32<KB>(pk, w8);
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    kw[2 * i] = rdna3_attn::bytes_to_h2c<KMID>(w8[i], 0x04010400u);
                    kw[2 * i + 1] = rdna3_attn::bytes_to_h2c<KMID>(w8[i], 0x04030402u);
                }
                gs = __half2float(k_scales[(size_t) tok_row * (NKV * G) + kvh * G + g]);
            }
            else
            {
                const uint4* kp = (const uint4*) ((const half*) k_cache + kbase + g * 32);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 v = kp[i];
                    kw[4 * i] = v.x; kw[4 * i + 1] = v.y; kw[4 * i + 2] = v.z; kw[4 * i + 3] = v.w;
                }
            }
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const uint4* qp = (const uint4*) &qs[row0 + r][g * 16];
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 qv = qp[i];
                    d = rdna3_attn::dot2(kw[4 * i], qv.x, d);
                    d = rdna3_attn::dot2(kw[4 * i + 1], qv.y, d);
                    d = rdna3_attn::dot2(kw[4 * i + 2], qv.z, d);
                    d = rdna3_attn::dot2(kw[4 * i + 3], qv.w, d);
                }
                if constexpr (KB > 0) d = (d + 0.5f * qsum[row0 + r][g]) * gs;   // (raw - KMID) + 0.5
                s[r] += d;
            }
        }

        // Online softmax, wave-uniform running max per row
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            const float sc = tvalid ? s[r] * sm_scale : -INFINITY;
            const float m_new = fmaxf(m[r], rdna3_attn::wave_max(sc));
            const float alpha = m[r] == -INFINITY ? 0.0f : __expf(m[r] - m_new);
            const float p = tvalid ? __expf(sc - m_new) : 0.0f;
            l[r] = l[r] * alpha + p;
            psum[r] *= alpha;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] *= alpha;
            m[r] = m_new;
            pbuf[warp][lane][r] = p;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // Values: lane <-> dims
        const int nt = min(TB, t_last - b0);
        for (int tt = 0; tt < nt; ++tt)
        {
            const int vrow = __shfl(tok_row, tt);
            const size_t vbase = (size_t) vrow * row_elems + (size_t) kvh * HD + lane * DPL;
            float vf[DPL];
            float vs = 1.0f;
            if constexpr (VB > 0)
            {
                const uint32_t* vw = (const uint32_t*) v_cache + (size_t) vrow * (NKV * G * VB) + (size_t) kvh * (G * VB) + (size_t) ((lane * DPL) / 32) * VB;
                uint32_t vr[DPL];
                rdna3_attn::qc_v_raw<VB, DPL>(vw, (lane * DPL) % 32, vr);
                #pragma unroll
                for (int j = 0; j < DPL; ++j) vf[j] = (float) vr[j];
                vs = __half2float(v_scales[(size_t) vrow * (NKV * G) + kvh * G + (lane * DPL) / 32]) * (1.0f / (float) VMID);
            }
            else
            {
                const half2* vp = (const half2*) ((const half*) v_cache + vbase);
                #pragma unroll
                for (int j = 0; j < DPL / 2; ++j)
                {
                    half2 v = vp[j];
                    vf[2 * j] = __low2float(v);
                    vf[2 * j + 1] = __high2float(v);
                }
            }
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const float pp = pbuf[warp][tt][r] * vs;
                psum[r] += pp;
                #pragma unroll
                for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(pp, vf[j], acc[r][j]);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }

    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        l[r] = rdna3_attn::wave_sum(l[r]);
        if constexpr (VB > 0)
        {
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] -= ((float) VMID - 0.5f) * psum[r];
        }
    }

    // Row (qi, hl) -> partial slot: rows of a kv head are packed head-major (hl * q_len + qi) and cut
    // into BLOCK_ROWS-row programs, as in triton_paged.decode_row_layout
    auto slot = [&] (int qi, int hl, float*& po, float*& pm)
    {
        const int row = hl * QL + qi;
        const int hb = row / BLOCK_ROWS;
        const int r16 = row % BLOCK_ROWS;
        const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
        po = partial_o + (pbase * BLOCK_ROWS + r16) * HD;
        pm = partial_ml + (pbase * BLOCK_ROWS + r16) * 2;
    };

    if constexpr (TG == 1)
    {
        if (split >= num_splits) return;
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            float* po; float* pm;
            slot(rg, r, po, pm);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) po[lane * DPL + j] = acc[r][j];
            if (lane == 0) { pm[0] = m[r]; pm[1] = l[r]; }
        }
        // Padded / out-of-group rows of the partial tiles are never read back into valid outputs
    }
    else
    {
        // Merge the token-group waves through LDS
        #pragma unroll
        for (int r = 0; r < RW; ++r)
            if (lane == 0) { mw[warp][r] = m[r]; lw[warp][r] = l[r]; }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            float mt = -INFINITY;
            #pragma unroll
            for (int w = 0; w < TG; ++w) mt = fmaxf(mt, mw[w][r]);
            const float f = m[r] == -INFINITY ? 0.0f : __expf(m[r] - mt);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) atomicAdd(&accm[r][lane * DPL + j], acc[r][j] * f);
        }
        __syncthreads();
        if (split >= num_splits) return;
        for (int i = threadIdx.x; i < BLOCK_ROWS * HD; i += NTH)
        {
            const int r16 = i / HD;            // QL == 1: r16 = head within h_block 0
            float* po = partial_o + ((size_t) (bh * H_BLOCKS) * num_splits + split) * BLOCK_ROWS * HD;
            po[i] = r16 < RW ? accm[r16][i % HD] : 0.0f;
        }
        if (threadIdx.x < BLOCK_ROWS)
        {
            const int r = threadIdx.x;
            float mt = -INFINITY, lt = 0.0f;
            if (r < RW)
            {
                #pragma unroll
                for (int w = 0; w < TG; ++w) mt = fmaxf(mt, mw[w][r]);
                #pragma unroll
                for (int w = 0; w < TG; ++w)
                    if (mw[w][r] != -INFINITY) lt += lw[w][r] * __expf(mw[w][r] - mt);
            }
            float* pm = partial_ml + ((size_t) (bh * H_BLOCKS) * num_splits + split) * BLOCK_ROWS * 2;
            pm[r * 2] = mt;
            pm[r * 2 + 1] = lt;
        }
    }
}

// Verification (2 <= q_len <= 8), quantized cache, matrix cores: S = Q K^T and O += P V as
// v_wmma_f32_16x16x16_f16 tiles. Rows (q_len x GROUP, query-major) pad to 16-row tiles; K/V tiles
// of TB tokens are staged in LDS as one byte per value (unpacked from the KB / VB bit planes as they are
// committed; V transposed by 4-dim words so a WMMA B fragment is 4 contiguous reads), with a register
// prefetch of the next tile
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <int KB, int VB, int HD, int QL, int GROUP>
__global__ __launch_bounds__(256)
void rdna3_attn_verify_wmma_kernel
(
    const half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,
    void* scratch0,
    void* scratch1
)
{
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX11__)
    // WMMA (v_wmma_f32_16x16x16_f16, wave32) is gfx11 only; the dispatcher never selects this kernel elsewhere
    return;
#else
    using namespace rdna3_attn;
    constexpr int NTH = 256;
    constexpr int W = 8;
    constexpr int BM = QL <= 2 ? 2 : QL <= 4 ? 4 : 8;
    constexpr int BH = 16 / BM;
    constexpr int BLOCK_ROWS = QL * GROUP <= 16 ? 16 : 32;   // partial layout rows (decode_row_layout)
    constexpr int NR = QL * GROUP;
    constexpr int RT = (NR + 15) / 16;              // 16-row tiles
    constexpr int NRP = RT * 16;
    constexpr int G = HD / 32;
    constexpr int KS = HD / 16;                     // k-steps over dims
    constexpr int DT = HD / 16;                     // 16-wide dim tiles
    constexpr int DTW = DT / W;                     // dim tiles per wave (PV)
    constexpr int H_BLOCKS = (QL * GROUP + BLOCK_ROWS - 1) / BLOCK_ROWS;
    constexpr int QSTR = HD + 8;                    // q row stride (halves), conflict-free fragment reads
    constexpr int KSTR = HD + 16;                   // key row stride (bytes)
    constexpr int VSTR = TB + 4;                    // transposed value row stride (words)
    constexpr int PSTR = TB + 8;                    // p row stride (halves)
    constexpr int KMID = 1 << (KB - 1);
    constexpr int VMID = 1 << (VB - 1);
    constexpr int NSC = 2 * TB * G;                 // (K or V, token, 32-value group) tasks: scales and packed words
    constexpr int PER_S = (NSC + NTH - 1) / NTH;
    static_assert(RT * 2 <= W && DT % W == 0, "tile decomposition");

    __shared__ __attribute__((aligned(16))) _Float16 qh[NRP][QSTR];
    __shared__ float qsum[NRP][G];
    __shared__ __attribute__((aligned(16))) uint8_t kt[TB][KSTR];
    __shared__ __attribute__((aligned(16))) uint32_t vtt[HD / 4][VSTR];
    __shared__ half ksc[TB][G];
    __shared__ __attribute__((aligned(16))) half vsc[G][TB];
    __shared__ float sbuf[NRP][TB];
    __shared__ __attribute__((aligned(16))) _Float16 pbuf[NRP][PSTR];
    __shared__ float mrow[NRP], lrow[NRP], arow[NRP];

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;

    const int h_block = pid % H_BLOCKS;
    if (h_block != 0) return;
    const int bh = pid / H_BLOCKS;
    const int NKV = (int) (intptr_t) scratch0;   // kv heads, bound at kernel creation (rdna3_attn_split_kernel)
    const int NQH = NKV * GROUP;
    const int batch = bh / NKV;
    const int kvh = bh % NKV;

    const int total_k_len = cache_seqlens[batch] + QL;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Stage q (rotated, fp16) and per-group sums; zero padding rows
    for (int task = threadIdx.x; task < NRP * G; task += NTH)
    {
        const int gr = task / G, g = task % G;
        float x[32];
        if (gr < NR)
        {
            const int qi = gr / GROUP, hl = gr % GROUP;
            const half* qp = q + ((size_t) (batch * QL + qi) * NQH + kvh * GROUP + hl) * HD + g * 32;
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] = __half2float(qp[i]);
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1)
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] = 0.0f;
        }
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
            const _Float16 v = (_Float16) (x[i] * 0.17677669529663687f);
            sq += (float) v;
            qh[gr][g * 32 + i] = v;
        }
        qsum[gr][g] = sq;
    }
    for (int r = threadIdx.x; r < NRP; r += NTH) { mrow[r] = -INFINITY; lrow[r] = 0.0f; }

    const float sm_scale = rsqrtf((float) HD) / (float) KMID;
    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const uint32_t* kc = (const uint32_t*) k_cache;
    const uint32_t* vc = (const uint32_t*) v_cache;

    uint32_t pw[PER_S][8];                          // the packed words of a task's group (KB or VB of the 8)
    half ps[PER_S];
    auto fetch = [&] (int tb)
    {
        #pragma unroll
        for (int i = 0; i < PER_S; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NSC)
            {
                const int isv = c / (TB * G);
                const int cc = c % (TB * G);
                const int tk = cc / G, g = cc % G;
                const int t = min(tb + tk, n_end - 1);
                const int tr = bt[t / PAGE] * PAGE + (t % PAGE);
                ps[i] = (isv ? v_scales : k_scales)[(size_t) tr * (NKV * G) + kvh * G + g];
                // isv is uniform per wave (TB * G is a multiple of the wave size)
                if (isv) rdna3_attn::qc_load_words<VB>(vc + (size_t) tr * (NKV * G * VB) + (size_t) kvh * (G * VB) + (size_t) g * VB, pw[i]);
                else rdna3_attn::qc_load_words<KB>(kc + (size_t) tr * (NKV * G * KB) + (size_t) kvh * (G * KB) + (size_t) g * KB, pw[i]);
            }
        }
    };
    auto commit = [&] ()
    {
        #pragma unroll
        for (int i = 0; i < PER_S; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NSC)
            {
                const int isv = c / (TB * G);
                const int cc = c % (TB * G);
                const int tk = cc / G, g = cc % G;
                uint32_t w8[8];
                if (isv) rdna3_attn::qc_unpack32<VB>(pw[i], w8);
                else rdna3_attn::qc_unpack32<KB>(pw[i], w8);
                if (isv)
                {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) vtt[g * 8 + k][tk] = w8[k];
                    vsc[g][tk] = ps[i];
                }
                else
                {
                    *(uint4*) &kt[tk][g * 32] = make_uint4(w8[0], w8[1], w8[2], w8[3]);
                    *(uint4*) &kt[tk][g * 32 + 16] = make_uint4(w8[4], w8[5], w8[6], w8[7]);
                    ksc[tk][g] = ps[i];
                }
            }
        }
    };

    // Output accumulators: this wave's dim tiles for every row tile, D layout
    v8f oacc[RT][DTW];
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt)
        #pragma unroll
        for (int k = 0; k < DTW; ++k)
            #pragma unroll
            for (int v = 0; v < 8; ++v) oacc[rt][k][v] = 0.0f;

    if (n_start < n_end) fetch(n_start);
    for (int tb = n_start; tb < n_end; tb += TB)
    {
        __syncthreads();
        commit();
        __syncthreads();
        if (tb + TB < n_end) fetch(tb + TB);

        // Scores: wave (rt, tt) computes a 16 x 16 tile of S
        if (warp < RT * 2)
        {
            const int rt = warp >> 1, tt = warp & 1;
            const int tcol = tt * 16 + l16;
            v8f sacc;
            #pragma unroll
            for (int v = 0; v < 8; ++v) sacc[v] = 0.0f;
            #pragma unroll 2
            for (int g = 0; g < G; ++g)
            {
                v8f c;
                #pragma unroll
                for (int v = 0; v < 8; ++v) c[v] = 0.0f;
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    const int ks = g * 2 + h;
                    v16h a = *(const v16h*) &qh[rt * 16 + l16][ks * 16];
                    const uint4 kb = *(const uint4*) &kt[tcol][ks * 16];
                    const uint32_t w4[4] = {kb.x, kb.y, kb.z, kb.w};
                    uint32_t bw[8];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        bw[2 * i] = rdna3_attn::bytes_to_h2c<KMID>(w4[i], 0x04010400u);
                        bw[2 * i + 1] = rdna3_attn::bytes_to_h2c<KMID>(w4[i], 0x04030402u);
                    }
                    v16h b = *(v16h*) bw;
                    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
                }
                const float gs = __half2float(ksc[tcol][g]);
                #pragma unroll
                for (int v = 0; v < 8; ++v)
                    sacc[v] += (c[v] + 0.5f * qsum[rt * 16 + 2 * v + hi][g]) * gs;
            }
            #pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                const int gr = rt * 16 + 2 * v + hi;
                const int t = tb + tcol;
                const int qi = gr / GROUP;
                const bool ok = gr < NR && t < n_end && t <= total_k_len - QL + qi;
                sbuf[gr][tcol] = ok ? sacc[v] * sm_scale : -INFINITY;
            }
        }
        __syncthreads();

        // Online softmax: 8 lanes per row, 4 tokens each
        for (int task = threadIdx.x; task < NRP * 8; task += NTH)
        {
            const int r = task >> 3, sub = task & 7;
            float sv[4];
            float mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < 4; ++i) { sv[i] = sbuf[r][sub * 4 + i]; mx = fmaxf(mx, sv[i]); }
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
            const float m_old = mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float ps4 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float p = sv[i] == -INFINITY ? 0.0f : __expf(sv[i] - m_new);
                ps4 += p;
                pbuf[r][sub * 4 + i] = (_Float16) p;
            }
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) ps4 += __shfl_xor(ps4, o);
            if (sub == 0)
            {
                const float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                arow[r] = alpha;
                lrow[r] = lrow[r] * alpha + ps4;
                mrow[r] = m_new;
            }
        }
        __syncthreads();

        // O = O * alpha + P V over this wave's dim tiles
        #pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            float al[8];
            #pragma unroll
            for (int v = 0; v < 8; ++v) al[v] = arow[rt * 16 + 2 * v + hi];
            #pragma unroll
            for (int k = 0; k < DTW; ++k)
                #pragma unroll
                for (int v = 0; v < 8; ++v) oacc[rt][k][v] *= al[v];
        }
        #pragma unroll
        for (int k = 0; k < DTW; ++k)
        {
            const int d = (warp * DTW + k) * 16 + l16;
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                const uint32_t* vr = &vtt[d >> 2][ks * 16];
                const uint4 w0 = *(const uint4*) (vr + 0), w1 = *(const uint4*) (vr + 4);
                const uint4 w2 = *(const uint4*) (vr + 8), w3 = *(const uint4*) (vr + 12);
                const uint32_t wv[16] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                                         w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w};
                // byte (d & 3) of tokens 2i (low half) and 2i + 1 (high half), as 0x64bb halves
                const uint32_t bsel = (uint32_t) (d & 3);
                const half2* vs2 = (const half2*) &vsc[d >> 5][ks * 16];
                uint32_t bw[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    // 0x64bb halves (1024 + b) for tokens 2i, 2i + 1, then (b - VMID) + 0.5 exactly, then scale
                    const uint32_t lo = (wv[2 * i] >> (8 * bsel)) & 0xffu;
                    const uint32_t hb = (wv[2 * i + 1] >> (8 * bsel)) & 0xffu;
                    const uint32_t hw = 0x64006400u | lo | (hb << 16);
                    half2 v = *reinterpret_cast<const half2*>(&hw);
                    v = __hsub2(v, __float2half2_rn((float) (1024 + VMID)));
                    v = __hadd2(v, __float2half2_rn(0.5f));
                    v = __hmul2(v, vs2[i]);
                    bw[i] = *reinterpret_cast<uint32_t*>(&v);
                }
                const v16h b = *(v16h*) bw;
                #pragma unroll
                for (int rt = 0; rt < RT; ++rt)
                {
                    const v16h a = *(const v16h*) &pbuf[rt * 16 + l16][ks * 16];
                    oacc[rt][k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, oacc[rt][k]);
                }
            }
        }
    }

    __syncthreads();
    if (split >= num_splits) return;
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt)
        #pragma unroll
        for (int k = 0; k < DTW; ++k)
            #pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                const int gr = rt * 16 + 2 * v + hi;
                if (gr >= NR) continue;
                const int qi = gr / GROUP, hl = gr % GROUP;
                const int hb = (hl * QL + qi) / BLOCK_ROWS, r16 = (hl * QL + qi) % BLOCK_ROWS;
                const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
                partial_o[(pbase * BLOCK_ROWS + r16) * HD + (warp * DTW + k) * 16 + l16] = oacc[rt][k][v] * (1.0f / (float) VMID);   // V grid step
            }
    for (int gr = threadIdx.x; gr < NR; gr += NTH)
    {
        const int qi = gr / GROUP, hl = gr % GROUP;
        const int hb = (hl * QL + qi) / BLOCK_ROWS, r16 = (hl * QL + qi) % BLOCK_ROWS;
        const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
        partial_ml[(pbase * BLOCK_ROWS + r16) * 2] = mrow[gr];
        partial_ml[(pbase * BLOCK_ROWS + r16) * 2 + 1] = lrow[gr];
    }
#endif
}

typedef void (*fp_rdna3_attn)(const half*, const void*, const void*, const int*, const int*, half*, float*, float*,
                              const half*, const half*, const half*, int, int, int, const float*, void*, void*);

// One instance per (cache, head_dim, q_len, GQA group) with at most 64 query rows per kv head: verification on an
// 8-bit cache (q_len 2..8, up to 48 rows, which bounds its LDS at head_dim 256) on matrix cores, everything else on
// the split kernel
template <int KB, int VB, int HD, int QL, int GROUP>
static fp_rdna3_attn pick_one()
{
    if constexpr (QL * GROUP > 64) return nullptr;
    else if constexpr (KB > 0 && QL >= 2 && QL <= 8 && QL * GROUP <= 48) return rdna3_attn_verify_wmma_kernel<KB, VB, HD, QL, GROUP>;
    else return rdna3_attn_decode_split_kernel<KB, VB, HD, QL, GROUP>;
}

template <int KB, int VB, int HD, int GROUP>
static fp_rdna3_attn pick_ql(int ql)
{
    switch (ql)
    {
        case 1:  return pick_one<KB, VB, HD, 1, GROUP>();
        case 2:  return pick_one<KB, VB, HD, 2, GROUP>();
        case 3:  return pick_one<KB, VB, HD, 3, GROUP>();
        case 4:  return pick_one<KB, VB, HD, 4, GROUP>();
        case 5:  return pick_one<KB, VB, HD, 5, GROUP>();
        case 6:  return pick_one<KB, VB, HD, 6, GROUP>();
        case 7:  return pick_one<KB, VB, HD, 7, GROUP>();
        case 8:  return pick_one<KB, VB, HD, 8, GROUP>();
        case 9:  return pick_one<KB, VB, HD, 9, GROUP>();
        case 10: return pick_one<KB, VB, HD, 10, GROUP>();
        case 11: return pick_one<KB, VB, HD, 11, GROUP>();
        case 12: return pick_one<KB, VB, HD, 12, GROUP>();
        case 13: return pick_one<KB, VB, HD, 13, GROUP>();
        case 14: return pick_one<KB, VB, HD, 14, GROUP>();
        case 15: return pick_one<KB, VB, HD, 15, GROUP>();
        case 16: return pick_one<KB, VB, HD, 16, GROUP>();
    }
    return nullptr;
}

template <int KB, int VB, int HD>
static fp_rdna3_attn pick_group(int group, int ql)
{
    switch (group)
    {
        case 2: return pick_ql<KB, VB, HD, 2>(ql);
        case 4: return pick_ql<KB, VB, HD, 4>(ql);
        case 6: return pick_ql<KB, VB, HD, 6>(ql);
        case 8: return pick_ql<KB, VB, HD, 8>(ql);
    }
    return nullptr;
}

// A quantized pair (kb, vb) is instantiated only if listed here; every pair costs ~26 s of compile per architecture, so
// pairs other than 8/8 take q_len <= 8 (the draft windows MTP / n-gram / DFlash produce) and leave the rest to Triton
template <int KB, int VB, int HD>
static fp_rdna3_attn pick_pair(int group, int ql)
{
    if (KB != 8 && ql > 8) return nullptr;
    return pick_group<KB, VB, HD>(group, ql);
}

template <int HD>
static fp_rdna3_attn pick_hd(int kb, int vb, int group, int ql)
{
    if (kb == 0 && vb == 0) return pick_pair<0, 0, HD>(group, ql);
    if (kb == 8 && vb == 8) return pick_pair<8, 8, HD>(group, ql);
    if (kb == 6 && vb == 6) return pick_pair<6, 6, HD>(group, ql);
    if (kb == 6 && vb == 4) return pick_pair<6, 4, HD>(group, ql);
    if (kb == 5 && vb == 5) return pick_pair<5, 5, HD>(group, ql);
    if (kb == 4 && vb == 4) return pick_pair<4, 4, HD>(group, ql);
    return nullptr;
}

#endif

std::shared_ptr<TritonKernel> rdna3_attn_split_kernel(int kb, int vb, int head_dim, int q_len, int n_kv_heads, int group)
{
    #ifdef __HIP_PLATFORM_AMD__
        fp_rdna3_attn k = nullptr;
        if (head_dim == 256) k = pick_hd<256>(kb, vb, group, q_len);
        if (head_dim == 128) k = pick_hd<128>(kb, vb, group, q_len);
        if (!k || n_kv_heads < 1) return nullptr;
        hipFunction_t fn;
        cuda_check(hipGetFuncBySymbol(&fn, (const void*) k));
        // the wave count each kernel is built for (AttnGeom, or the WMMA verify kernel's 8)
        const bool wmma = kb > 0 && q_len >= 2 && q_len <= 8 && q_len * group <= 48;
        const int waves = q_len == 1 ? 4 : wmma ? 8 : q_len;
        // The kv head count rides in the first trailing scratch slot (unused by these kernels otherwise)
        return std::make_shared<TritonKernel>((void*) fn, std::string("rdna3_attn_decode_split"), waves, 0,
                                              (void*) (intptr_t) n_kv_heads);
    #else
        return nullptr;
    #endif
}
