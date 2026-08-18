// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan VAE convolutions on MetalPerformanceShadersGraph.
//
// ggml lowers a 2D convolution to im2col + mul_mat. For the Wan video VAE that
// is the single most expensive thing in a generation: a decode profile
// (benchmarks/WanVideoBench, 640x480x9f) attributes 44.4% of the graph to
// MUL_MAT and a further 30.2% to IM2COL alone -- 74.6% in the convolution, with
// im2col moving 865 MB per node at ~57 GB/s, a fifth of the machine's
// bandwidth. The lowering is pure overhead: a 3x3 conv materialises 9x its
// input before the GEMM ever starts.
//
// Apple ships a tuned convolution in MPSGraph. Measured on an M5 Pro at the
// shapes the Wan 2.2 decoder actually runs (ggml im2col+GEMM vs MPSGraph):
//
//     512->512 k3 320x240 t9    666 ms -> 108 ms   6.2x
//     256->256 k3 640x480 t9    947 ms -> 110 ms   8.6x
//     160->160 k3 640x480 t9    491 ms ->  47 ms  10.4x
//     512->512 k1 320x240 t9    184 ms ->  14 ms  13.9x
//
// MPS reaches ~30 TFLOP/s where ggml's Metal GEMM gets ~4.9. Critically it does
// so WITHOUT ggml's mul_mm kernel -- the one whose Metal 4 tensor path corrupts
// this exact graph into black frames (see wan_vae_gemm_budget). This buys the
// matrix-unit throughput while stepping around that defect entirely.
//
// Layout costs nothing: ggml's activation [W,H,C,T] and kernel [KW,KH,IC,OC]
// are, byte for byte, MPS NCHW and OIHW. NCHW measured as fast as NHWC here, so
// no transpose is inserted on either side.
//
// Numerics are unchanged: ggml already runs these convolutions with F16 kernels
// and F16 im2col, accumulating to F32. MPS does the same (the graph casts the
// F32 activation to F16, convolves, and casts back), and the two agree to
// relL2 2.07e-4 / max rel err 4.8e-4 on random data -- F16 rounding.
//
// TS_WAN_VAE_MPS_CONV=0 falls back to ggml's im2col+GEMM.
#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

namespace {

struct MpsState
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    bool probed = false;
    bool usable = false;
    std::mutex mu;

    // One compiled MPSGraph per (kw,kh,ic,oc,stride,pad) -- the shape-independent
    // part. Activation extents are placeholders so a graph is reused across the
    // decoder's temporal chunks and band sizes.
    struct Key
    {
        int kw, kh, ic, oc, stride, pad, w, h, t, wf16;
        bool operator<(const Key& o) const
        {
            return std::tie(kw, kh, ic, oc, stride, pad, w, h, t, wf16) <
                   std::tie(o.kw, o.kh, o.ic, o.oc, o.stride, o.pad, o.w, o.h, o.t, o.wf16);
        }
    };
    struct Entry
    {
        MPSGraph* graph;
        MPSGraphTensor* x;
        MPSGraphTensor* w;
        MPSGraphTensor* y;
    };
    std::map<Key, Entry> cache;

    // Staging buffers, grown on demand and reused: ggml's tensor storage is not
    // guaranteed page-aligned, which newBufferWithBytesNoCopy requires, so the
    // operands are staged. At the shapes that matter the copies are ~3% of what
    // the convolution itself costs.
    id<MTLBuffer> bufX = nil, bufW = nil, bufY = nil;
    std::size_t capX = 0, capW = 0, capY = 0;
};

MpsState& state()
{
    static MpsState s;
    return s;
}

bool ensure_device(MpsState& s)
{
    if (s.probed) return s.usable;
    s.probed = true;
    if (const char* e = std::getenv("TS_WAN_VAE_MPS_CONV"); e != nullptr && e[0] == '0')
        return s.usable = false;
    s.device = MTLCreateSystemDefaultDevice();
    if (s.device == nil) return s.usable = false;
    s.queue = [s.device newCommandQueue];
    if (s.queue == nil) return s.usable = false;
    // MPSGraph convolution2D needs macOS 12.0+ / iOS 15.0+.
    if (@available(macOS 12.0, iOS 15.0, *)) { s.usable = true; }
    else { s.usable = false; }
    return s.usable;
}

id<MTLBuffer> ensure_buffer(MpsState& s, id<MTLBuffer>& buf, std::size_t& cap, std::size_t need)
{
    if (buf != nil && cap >= need) return buf;
    buf = [s.device newBufferWithLength:need options:MTLResourceStorageModeShared];
    cap = (buf != nil) ? need : 0;
    return buf;
}

} // namespace

extern "C" {

// True when MPSGraph convolutions can be used for the Wan VAE on this process.
bool tsg_mps_conv2d_available(void)
{
    MpsState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return ensure_device(s);
}

// One convolution. Activation `x` is F32 [W,H,IC,T] and kernel `w` is F16
// [KW,KH,IC,OC], both in ggml's memory order; `dst` receives F32
// [OW,OH,OC,T]. Returns false when the shape is unsupported, in which case the
// caller must fall back to the ggml path.
bool tsg_mps_conv2d(
    const void* w, int wIsF16, int kw, int kh, int ic, int oc,
    const float* x, int W, int H, int T,
    int stride, int pad,
    float* dst, int OW, int OH)
{
    if (w == nullptr || x == nullptr || dst == nullptr) return false;
    if (kw <= 0 || kh <= 0 || ic <= 0 || oc <= 0 || W <= 0 || H <= 0 || T <= 0) return false;
    if (stride <= 0 || pad < 0 || OW <= 0 || OH <= 0) return false;

    MpsState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_device(s)) return false;

    @autoreleasepool {
        const std::size_t nx = (std::size_t)W * H * ic * T * sizeof(float);
        const std::size_t nw = (std::size_t)kw * kh * ic * oc * (wIsF16 ? sizeof(std::uint16_t) : sizeof(float));
        const std::size_t ny = (std::size_t)OW * OH * oc * T * sizeof(float);

        id<MTLBuffer> bx = ensure_buffer(s, s.bufX, s.capX, nx);
        id<MTLBuffer> bw = ensure_buffer(s, s.bufW, s.capW, nw);
        id<MTLBuffer> by = ensure_buffer(s, s.bufY, s.capY, ny);
        if (bx == nil || bw == nil || by == nil) return false;

        std::memcpy(bx.contents, x, nx);
        std::memcpy(bw.contents, w, nw);

        MpsState::Key key{ kw, kh, ic, oc, stride, pad, W, H, T, wIsF16 };
        auto it = s.cache.find(key);
        if (it == s.cache.end())
        {
            MPSGraph* g = [MPSGraph new];
            // ggml [W,H,C,T] == NCHW(T,C,H,W); ggml [KW,KH,IC,OC] == OIHW(OC,IC,KH,KW).
            MPSShape* xs = @[ @(T), @(ic), @(H), @(W) ];
            MPSShape* ws = @[ @(oc), @(ic), @(kh), @(kw) ];
            MPSGraphTensor* xt = [g placeholderWithShape:xs dataType:MPSDataTypeFloat32 name:@"x"];
            const MPSDataType wdt = wIsF16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            MPSGraphTensor* wt = [g placeholderWithShape:ws dataType:wdt name:@"w"];
            MPSGraphConvolution2DOpDescriptor* d =
                [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride strideInY:stride
                                                           dilationRateInX:1 dilationRateInY:1
                                                                    groups:1
                                                               paddingLeft:pad paddingRight:pad
                                                                paddingTop:pad paddingBottom:pad
                                                              paddingStyle:MPSGraphPaddingStyleExplicit
                                                                dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                             weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            if (d == nil) return false;
            // Match the precision ggml would have used for this node: an F16
            // kernel means ggml lowered through an F16 im2col, so convolve in F16
            // and hand back F32; an F32 kernel (Qwen-Image's VAE) stays F32
            // end to end. Either way this is a drop-in for the node it replaces.
            MPSGraphTensor* yt;
            if (wIsF16)
            {
                MPSGraphTensor* xh = [g castTensor:xt toType:MPSDataTypeFloat16 name:@"xh"];
                MPSGraphTensor* yh = [g convolution2DWithSourceTensor:xh weightsTensor:wt descriptor:d name:nil];
                yt = [g castTensor:yh toType:MPSDataTypeFloat32 name:@"y"];
            }
            else
            {
                yt = [g convolution2DWithSourceTensor:xt weightsTensor:wt descriptor:d name:nil];
            }
            it = s.cache.emplace(key, MpsState::Entry{ g, xt, wt, yt }).first;
        }
        MpsState::Entry& e = it->second;

        MPSGraphTensorData* xd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bx
                                                                        shape:@[ @(T), @(ic), @(H), @(W) ]
                                                                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* wd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bw
                                                                        shape:@[ @(oc), @(ic), @(kh), @(kw) ]
                                                                     dataType:(wIsF16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];
        MPSGraphTensorData* yd = [[MPSGraphTensorData alloc] initWithMTLBuffer:by
                                                                        shape:@[ @(T), @(oc), @(OH), @(OW) ]
                                                                     dataType:MPSDataTypeFloat32];
        if (xd == nil || wd == nil || yd == nil) return false;

        [e.graph runWithMTLCommandQueue:s.queue
                                  feeds:@{ e.x : xd, e.w : wd }
                       targetOperations:nil
                      resultsDictionary:@{ e.y : yd }];

        std::memcpy(dst, by.contents, ny);
        return true;
    }
}

// Drop every cached graph and staging buffer (called when the VAE hands back its
// device residency between generations).
void tsg_mps_conv2d_release(void)
{
    MpsState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.cache.clear();
    s.bufX = s.bufW = s.bufY = nil;
    s.capX = s.capW = s.capY = 0;
}

} // extern "C"

#endif // __APPLE__
