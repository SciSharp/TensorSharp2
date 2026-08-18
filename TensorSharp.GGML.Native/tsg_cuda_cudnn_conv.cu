// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan VAE convolutions on cuDNN.
//
// The CUDA twin of tsg_metal_mps_conv.mm. ggml lowers conv2d to im2col +
// mul_mat, and on CUDA that lowering is even more dominant than on Metal
// because cuBLAS makes the GEMM itself cheap. A decode profile on an L4
// (benchmarks/WanVideoBench, 640x480x9f, ggml_cuda):
//
//     IM2COL    46.0%   298 nodes     <- materialising 9x the input
//     MUL_MAT   24.5%   310 nodes
//     CONT      11.8%   593 nodes
//
// So ~70% of a VAE decode goes into the convolution and most of that is the
// lowering, not the math. cuDNN convolves directly, with no im2col tensor.
//
// Layouts line up exactly, as they do for MPS: ggml's activation [W,H,C,T] is
// NCHW and its kernel [KW,KH,IC,OC] is OIHW, byte for byte, so nothing is
// transposed. Unlike the Metal path this one is also zero-copy -- on CUDA a
// ggml tensor's `data` is already a device pointer, so cuDNN reads and writes
// ggml's buffers in place.
//
// Precision: ggml runs these convolutions with F16 kernels and F16 im2col,
// accumulating in F32. Here the (small) kernel is widened to F32 once and cached,
// and the convolution runs F32 in/out with CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
// so the tensor cores still do the work at no worse accuracy than before.
//
// TS_WAN_VAE_CUDNN_CONV=0 falls back to ggml's im2col+GEMM.
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

struct CudnnState
{
    cudnnHandle_t handle = nullptr;
    bool probed = false;
    bool usable = false;
    std::mutex mu;

    void* workspace = nullptr;
    std::size_t workspaceBytes = 0;

    // Scratch for widening the F16 kernel to F32. Deliberately NOT a cache keyed
    // by device pointer: ggml's allocator reuses addresses across graphs, so a
    // pointer-keyed cache hands back another tensor's weights on a later decode.
    // Re-widening costs one small kernel launch per convolution.
    void* kernelScratch = nullptr;
    std::size_t kernelScratchBytes = 0;

    ~CudnnState() = default;
};

CudnnState& state()
{
    static CudnnState s;
    return s;
}

bool ensure_handle(CudnnState& s)
{
    if (s.probed) return s.usable;
    s.probed = true;
    if (const char* e = std::getenv("TS_WAN_VAE_CUDNN_CONV"); e != nullptr && e[0] == '0')
        return s.usable = false;
    if (cudnnCreate(&s.handle) != CUDNN_STATUS_SUCCESS)
    {
        s.handle = nullptr;
        return s.usable = false;
    }
    return s.usable = true;
}

void* ensure_workspace(CudnnState& s, std::size_t need)
{
    if (need == 0) return nullptr;
    if (s.workspace != nullptr && s.workspaceBytes >= need) return s.workspace;
    if (s.workspace != nullptr) cudaFree(s.workspace);
    s.workspace = nullptr;
    s.workspaceBytes = 0;
    if (cudaMalloc(&s.workspace, need) != cudaSuccess) { s.workspace = nullptr; return nullptr; }
    s.workspaceBytes = need;
    return s.workspace;
}

// Widen an F16 kernel to F32 on device, once per distinct kernel pointer.
__global__ void tsg_f16_to_f32(const __half* __restrict__ src, float* __restrict__ dst, long long n)
{
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

float* widen_kernel_f32(CudnnState& s, const void* f16, long long n)
{
    const std::size_t need = (std::size_t) n * sizeof(float);
    if (s.kernelScratch == nullptr || s.kernelScratchBytes < need)
    {
        if (s.kernelScratch != nullptr) cudaFree(s.kernelScratch);
        s.kernelScratch = nullptr;
        s.kernelScratchBytes = 0;
        if (cudaMalloc(&s.kernelScratch, need) != cudaSuccess) { s.kernelScratch = nullptr; return nullptr; }
        s.kernelScratchBytes = need;
    }
    const int threads = 256;
    const int blocks = (int) ((n + threads - 1) / threads);
    tsg_f16_to_f32<<<blocks, threads>>>(static_cast<const __half*>(f16), static_cast<float*>(s.kernelScratch), n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return nullptr;
    return static_cast<float*>(s.kernelScratch);
}

struct Desc
{
    cudnnTensorDescriptor_t x = nullptr, y = nullptr;
    cudnnFilterDescriptor_t w = nullptr;
    cudnnConvolutionDescriptor_t c = nullptr;
    ~Desc()
    {
        if (x) cudnnDestroyTensorDescriptor(x);
        if (y) cudnnDestroyTensorDescriptor(y);
        if (w) cudnnDestroyFilterDescriptor(w);
        if (c) cudnnDestroyConvolutionDescriptor(c);
    }
};

} // namespace

extern "C" {

bool tsg_cudnn_conv2d_available(void)
{
    CudnnState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return ensure_handle(s);
}

// One convolution, in place on ggml's device buffers. `w` is the F16 kernel
// [KW,KH,IC,OC], `x` the F32 activation [W,H,IC,T], `dst` the F32 output
// [OW,OH,OC,T] -- all device pointers in ggml's memory order.
bool tsg_cudnn_conv2d(
    const void* w, int wIsF16, int kw, int kh, int ic, int oc,
    const void* x, int W, int H, int T,
    int stride, int pad,
    void* dst, int OW, int OH)
{
    if (w == nullptr || x == nullptr || dst == nullptr) return false;
    if (kw <= 0 || kh <= 0 || ic <= 0 || oc <= 0 || W <= 0 || H <= 0 || T <= 0) return false;
    if (stride <= 0 || pad < 0 || OW <= 0 || OH <= 0) return false;

    CudnnState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;

    // An F32 kernel (Qwen-Image's VAE) is handed to cuDNN as it lies; an F16 one
    // (Wan's) is widened into scratch first.
    const float* wf32 = wIsF16
        ? widen_kernel_f32(s, w, (long long) kw * kh * ic * oc)
        : static_cast<const float*>(w);
    if (wf32 == nullptr) return false;

    Desc d;
    if (cudnnCreateTensorDescriptor(&d.x) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateTensorDescriptor(&d.y) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateFilterDescriptor(&d.w) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateConvolutionDescriptor(&d.c) != CUDNN_STATUS_SUCCESS) return false;

    // ggml [W,H,C,T] == NCHW(T,C,H,W); ggml [KW,KH,IC,OC] == OIHW(OC,IC,KH,KW).
    if (cudnnSetTensor4dDescriptor(d.x, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T, ic, H, W) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnSetTensor4dDescriptor(d.y, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T, oc, OH, OW) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnSetFilter4dDescriptor(d.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, oc, ic, kh, kw) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnSetConvolution2dDescriptor(d.c, pad, pad, stride, stride, 1, 1,
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) return false;
    // Let cuDNN use the tensor cores on F32 data.
    cudnnSetConvolutionMathType(d.c, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

    // Confirm cuDNN agrees with the output extent ggml allocated for this node.
    int n_ = 0, c_ = 0, h_ = 0, w_ = 0;
    if (cudnnGetConvolution2dForwardOutputDim(d.c, d.x, d.w, &n_, &c_, &h_, &w_) != CUDNN_STATUS_SUCCESS) return false;
    if (n_ != T || c_ != oc || h_ != OH || w_ != OW) return false;

    // Only the implicit-GEMM algorithms. cuDNN's heuristic will happily rank
    // FFT_TILING / WINOGRAD_NONFUSED first, and those carry alignment and padding
    // requirements that ggml's buffers do not always satisfy -- which is how a
    // "successful" convolution turns into an illegal memory access later in the
    // graph. The implicit-GEMM kernels have no such constraints and are what
    // makes the win here anyway (the point is avoiding im2col, not the algorithm).
    const cudnnConvolutionFwdAlgo_t safe[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    };
    const int returned = (int) (sizeof(safe) / sizeof(safe[0]));
    for (int i = 0; i < returned; i++)
    {
        std::size_t need = 0;
        if (cudnnGetConvolutionForwardWorkspaceSize(s.handle, d.x, d.w, d.c, d.y, safe[i], &need) != CUDNN_STATUS_SUCCESS)
            continue;
        void* ws = nullptr;
        if (need > 0)
        {
            ws = ensure_workspace(s, need);
            if (ws == nullptr) continue;   // too large for the device; try the next algo
        }
        const float alpha = 1.0f, beta = 0.0f;
        const cudnnStatus_t st = cudnnConvolutionForward(
            s.handle, &alpha, d.x, x, d.w, wf32, d.c, safe[i],
            ws, need, &beta, d.y, dst);
        if (st != CUDNN_STATUS_SUCCESS) continue;
        return cudaDeviceSynchronize() == cudaSuccess;
    }
    return false;
}

void tsg_cudnn_conv2d_release(void)
{
    CudnnState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.kernelScratch != nullptr) { cudaFree(s.kernelScratch); s.kernelScratch = nullptr; s.kernelScratchBytes = 0; }
    if (s.workspace != nullptr) { cudaFree(s.workspace); s.workspace = nullptr; s.workspaceBytes = 0; }
}

} // extern "C"

#endif // TSG_GGML_USE_CUDA && TSG_HAVE_CUDNN
