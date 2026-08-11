// On-device causal/windowed attention-mask generation for the Gemma 4 fused
// verify (prefill) kernel. Replaces the host-side per-chunk mask fill
// (O(N * kvLen) CPU work that blocks the GPU) + the H2D upload with a tiny CUDA
// kernel that writes the F16 mask straight into its device buffer.
//
// The pattern EXACTLY mirrors the host fill in ggml_ops_gemma4_verify.cpp's
// get_causal_mask (non-bidirectional / text path):
//   threshold = nPast + qi
//   lo = max(window > 0 ? threshold - window + 1 : 0, 0)
//   hi = min(threshold, validLen - 1)
//   mask[qi][ki] = 0 for ki in [lo, hi], else -inf   (ki in [0, kvLen))
// fp16 0.0 (0x0000) and -inf (0xFC00) match ggml_fp32_to_fp16, so the produced
// mask is bit-identical to the host path.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void tsg_fill_causal_mask_f16_kernel(
    __half* __restrict__ mask, int kvLen, int N, int nPast, int window, int validLen)
{
    const int qi = blockIdx.x;
    if (qi >= N) return;

    const int threshold = nPast + qi;
    int lo = (window > 0) ? (threshold - window + 1) : 0;
    if (lo < 0) lo = 0;
    int hi = (threshold < validLen - 1) ? threshold : (validLen - 1);

    __half* row = mask + (size_t) qi * kvLen;
    const __half zero = __float2half(0.0f);
    const __half ninf = __float2half(-INFINITY);
    for (int ki = threadIdx.x; ki < kvLen; ki += blockDim.x)
        row[ki] = (ki >= lo && ki <= hi) ? zero : ninf;
}

// Fill one [kvLen, N] F16 causal(+windowed) mask on stream 0. Not synchronized;
// the caller batches several fills then calls tsg_cuda_sync_stream0 once before
// enqueuing the graph (which reads the masks on the backend stream).
extern "C" bool tsg_cuda_fill_causal_mask_f16(
    void* mask_dev, int kvLen, int N, int nPast, int window, int validLen)
{
    if (mask_dev == nullptr || kvLen <= 0 || N <= 0) return false;
    const int threads = 256;
    tsg_fill_causal_mask_f16_kernel<<<(unsigned) N, threads, 0, 0>>>(
        (__half*) mask_dev, kvLen, N, nPast, window, validLen);
    return cudaGetLastError() == cudaSuccess;
}

// Block the host until the stream-0 mask fills complete, so the subsequent
// backend-stream graph compute sees the finished masks (host-ordered).
extern "C" bool tsg_cuda_sync_stream0(void)
{
    return cudaStreamSynchronize(0) == cudaSuccess;
}

// ---------------------------------------------------------------------------
// Ring-cache mask, for Muse-Glimmer's sliding-window layers.
//
// Their KV cache is a ring of ringRows slots where slot (p % ringRows) holds
// position p, so columns are NOT in position order and the causal kernel above
// does not apply. Mirrors fill_mg_ring_mask in ggml_ops_muse_glimmer.cpp:
//   lo = max(0, startPos + qi - window + 1)
//   mask[qi][p % ringRows] = 0 for p in [lo, startPos + qi], else -inf
//
// Inverted so each thread owns one slot: the unique position >= lo congruent to
// s (mod ringRows) is p = lo + ((s - lo) mod ringRows), and the slot is live iff
// p <= qp. Uniqueness holds because the caller guarantees
// ringRows >= window + chunk, so a single query's live span never wraps onto
// itself -- the same +1 margin that sizes the ring.
__global__ void tsg_fill_ring_mask_f16_kernel(
    __half* __restrict__ mask, int ringRows, int N, int startPos, int window)
{
    const int qi = blockIdx.x;
    if (qi >= N) return;

    const int qp = startPos + qi;
    int lo = (window > 0) ? (qp - window + 1) : 0;
    if (lo < 0) lo = 0;

    __half* row = mask + (size_t) qi * ringRows;
    const __half zero = __float2half(0.0f);
    const __half ninf = __float2half(-INFINITY);
    const int lo_mod = lo % ringRows;
    for (int s = threadIdx.x; s < ringRows; s += blockDim.x)
    {
        int off = s - lo_mod;
        if (off < 0) off += ringRows;
        row[s] = (lo + off <= qp) ? zero : ninf;
    }
}

// Fill one [ringRows, N] F16 ring mask on stream 0. Same batching contract as
// tsg_cuda_fill_causal_mask_f16: not synchronized, call tsg_cuda_sync_stream0.
extern "C" bool tsg_cuda_fill_ring_mask_f16(
    void* mask_dev, int ringRows, int N, int startPos, int window)
{
    if (mask_dev == nullptr || ringRows <= 0 || N <= 0) return false;
    const int threads = 256;
    tsg_fill_ring_mask_f16_kernel<<<(unsigned) N, threads, 0, 0>>>(
        (__half*) mask_dev, ringRows, N, startPos, window);
    return cudaGetLastError() == cudaSuccess;
}
