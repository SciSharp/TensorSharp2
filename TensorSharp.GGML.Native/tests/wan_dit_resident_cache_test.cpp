// Regression test: a Wan DiT graph that is BUILT and then ABANDONED must not
// leave the process-wide resident weight cache holding device buffers that were
// never filled.
//
// wan_dit_build_persist() binds every weight through WanBind::bind, which pulls a
// device buffer out of the resident cache (allocating and publishing a cache entry
// on a miss). It then allocates the graph and checks free VRAM; on a 16 GB card at
// the 1088x832x121-frame shape (27404 tokens) that check trips and the whole graph
// is dropped in favour of the rebuild-per-forward path.
//
// When the weight uploads were deferred to a loop *after* that check, the abandoned
// build published ~3 GB of resident cache entries whose device memory had never been
// written. The next build hit those entries with needs_upload == false and computed
// the entire 30-block DiT against them. Freshly mapped VRAM reads as zeros, so the
// result was finite, NaN-free, correctly shaped — and IDENTICAL for every token,
// because norm(0) is 0 and the head then emits one constant vector. Nothing
// asserted; the user just got a video of pure noise.
//
// The test drives the real TSGgml_WanDitForward on a small synthetic DiT:
//   1. reference pass with TS_WAN_NO_RESIDENT=1, which bypasses the resident cache
//      entirely and uploads every weight per call — correct by construction, and
//      independent of the code path under test,
//   2. caches cleared, resident path re-enabled, then a pass with
//      TS_WAN_DIT_CAPTURE=abandon, which makes wan_dit_build_persist drop the graph
//      at exactly the spill-guard point,
//   3. a repeat pass, which is the one that consumes the entries the abandoned
//      build published.
// Phases 2 and 3 must match the reference and must be token-DEPENDENT. Before the
// fix they produced the constant-per-token output described above.
//
// NOTE: this only works because WanBind::bind and wan_dit_capture_enabled() read
// their env knobs per call rather than caching them in a static — an earlier
// version of this test passed against the *broken* build because the first phase's
// TS_WAN_DIT_CAPTURE=0 had latched the persistent builder off for the whole process,
// so phase 2 never reached wan_dit_build_persist at all.
//
// Build with -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON; run with no arguments
// (CUDA when available, otherwise CPU) or pass "cpu" / "cuda".

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

extern "C" {

struct TSGWanW
{
    void* w;
    std::int32_t type;
    std::int32_t reserved;
    std::int64_t ne0, ne1;
    std::int64_t bytes;
    void* b;
};

struct TSGWanDitBlockW
{
    void* modulation;
    TSGWanW sq, sk, sv, so;
    void* s_norm_q;
    void* s_norm_k;
    void* norm3_w;
    void* norm3_b;
    TSGWanW xq, xk, xv, xo;
    void* x_norm_q;
    void* x_norm_k;
    TSGWanW ffn0, ffn2;
};

struct TSGgmlWanDitDesc
{
    float* x;
    float* out;
    float* context;
    float* tsin;
    float* tsin0;
    float* cosf;
    float* sinf;
    TSGWanW patch;
    TSGWanW text0, text2;
    TSGWanW time0, time2;
    TSGWanW tproj;
    TSGWanW head;
    void* head_mod;
    const TSGWanDitBlockW* blocks;
    std::int32_t num_layers, dim, ff, heads, head_dim, seq, ctx_len, freq_dim, text_dim, seq0;
    float eps;
    std::int32_t struct_bytes;
};

const char* TSGgml_GetLastError();
int  TSGgml_IsBackendAvailable(int backendType);
int  TSGgml_WanDitForward(const TSGgmlWanDitDesc* d);
void TSGgml_WanResetForwardCache();
void TSGgml_ClearHostBufferCache();
void TSGgml_InvalidateHostBuffer(void* ptr);

} // extern "C"

namespace {

constexpr int BackendTypeCpu = 2;
constexpr int BackendTypeCuda = 3;
constexpr int GgmlTypeF32 = 0;

// Small but structurally faithful: two blocks, the TI2V two-segment timestep path
// active (seq0 > 0), and a sequence long enough that a constant output cannot be
// mistaken for a legitimately uniform one.
constexpr int kLayers  = 2;
constexpr int kDim     = 128;
constexpr int kFf      = 256;
constexpr int kHeads   = 2;
constexpr int kHeadDim = 64;
constexpr int kSeq     = 96;
constexpr int kSeq0    = 16;
constexpr int kCtxLen  = 32;
constexpr int kFreqDim = 64;
constexpr int kTextDim = 96;
constexpr int kInTok   = 48;   // patchified latent features per token
constexpr int kOutTok  = 32;   // velocity features per token

// Deterministic, reproducible, and small enough in magnitude that two blocks of
// residual growth stay in a comfortable float range.
struct Rng
{
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next(float scale)
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (static_cast<float>((s >> 40) & 0xFFFF) / 32768.0f - 1.0f) * scale;
    }
};

// Every host buffer the descriptor points at, kept alive for the run and
// individually invalidatable (the resident cache is keyed by host pointer).
struct HostArena
{
    std::vector<std::vector<float>> blocks;
    std::vector<void*> ptrs;

    float* alloc(std::size_t n, Rng& rng, float scale)
    {
        blocks.emplace_back(n);
        auto& v = blocks.back();
        for (std::size_t i = 0; i < n; i++) v[i] = rng.next(scale);
        ptrs.push_back(v.data());
        return v.data();
    }

    // Drop every device copy so the next forward re-binds from scratch. Pairs with
    // TSGgml_ClearHostBufferCache: together they put the cache back to the state a
    // freshly loaded model sees.
    void invalidateAll()
    {
        for (void* p : ptrs) TSGgml_InvalidateHostBuffer(p);
    }
};

TSGWanW makeW(HostArena& a, Rng& rng, std::int64_t ne0, std::int64_t ne1, bool bias, float scale)
{
    TSGWanW w{};
    w.w = a.alloc(static_cast<std::size_t>(ne0 * ne1), rng, scale);
    w.type = GgmlTypeF32;
    w.ne0 = ne0;
    w.ne1 = ne1;
    w.bytes = ne0 * ne1 * static_cast<std::int64_t>(sizeof(float));
    w.b = bias ? a.alloc(static_cast<std::size_t>(ne1), rng, scale) : nullptr;
    return w;
}

void buildDesc(HostArena& a, Rng& rng,
               std::vector<TSGWanDitBlockW>& blocks,
               std::vector<float>& xIn, std::vector<float>& out,
               std::vector<float>& ctx, std::vector<float>& tsin, std::vector<float>& tsin0,
               std::vector<float>& cosf, std::vector<float>& sinf,
               TSGgmlWanDitDesc& d)
{
    // 1/sqrt(fan_in)-ish scales keep two residual blocks numerically tame.
    const float wScale = 0.08f;

    xIn.resize(static_cast<std::size_t>(kInTok) * kSeq);
    out.assign(static_cast<std::size_t>(kOutTok) * kSeq, 0.0f);
    ctx.resize(static_cast<std::size_t>(kTextDim) * kCtxLen);
    tsin.resize(kFreqDim);
    tsin0.resize(kFreqDim);
    cosf.resize(static_cast<std::size_t>(kHeadDim) * kSeq);
    sinf.resize(static_cast<std::size_t>(kHeadDim) * kSeq);

    for (auto& v : xIn) v = rng.next(1.0f);
    for (auto& v : ctx) v = rng.next(0.5f);
    for (int i = 0; i < kFreqDim; i++)
    {
        // Same shape as WanDiT.FillSinusoid: [cos | sin] halves.
        const int half = kFreqDim / 2;
        const float w = std::pow(10000.0f, -static_cast<float>(i % half) / half);
        tsin[i]  = i < half ? std::cos(999.0f * w) : std::sin(999.0f * w);
        tsin0[i] = i < half ? 1.0f : 0.0f;   // timestep 0
    }
    for (int t = 0; t < kSeq; t++)
        for (int j = 0; j < kHeadDim; j++)
        {
            const float ang = static_cast<float>(t) * std::pow(10000.0f, -2.0f * (j / 2) / static_cast<float>(kHeadDim));
            cosf[static_cast<std::size_t>(t) * kHeadDim + j] = std::cos(ang);
            sinf[static_cast<std::size_t>(t) * kHeadDim + j] = std::sin(ang);
        }

    blocks.assign(kLayers, TSGWanDitBlockW{});
    for (int l = 0; l < kLayers; l++)
    {
        TSGWanDitBlockW& b = blocks[static_cast<std::size_t>(l)];
        b.modulation = a.alloc(static_cast<std::size_t>(6) * kDim, rng, 0.2f);
        b.sq = makeW(a, rng, kDim, kDim, true, wScale);
        b.sk = makeW(a, rng, kDim, kDim, true, wScale);
        b.sv = makeW(a, rng, kDim, kDim, true, wScale);
        b.so = makeW(a, rng, kDim, kDim, true, wScale);
        b.s_norm_q = a.alloc(kDim, rng, 0.3f);
        b.s_norm_k = a.alloc(kDim, rng, 0.3f);
        b.norm3_w  = a.alloc(kDim, rng, 0.3f);
        b.norm3_b  = a.alloc(kDim, rng, 0.1f);
        b.xq = makeW(a, rng, kDim, kDim, true, wScale);
        b.xk = makeW(a, rng, kDim, kDim, true, wScale);
        b.xv = makeW(a, rng, kDim, kDim, true, wScale);
        b.xo = makeW(a, rng, kDim, kDim, true, wScale);
        b.x_norm_q = a.alloc(kDim, rng, 0.3f);
        b.x_norm_k = a.alloc(kDim, rng, 0.3f);
        b.ffn0 = makeW(a, rng, kDim, kFf, true, wScale);
        b.ffn2 = makeW(a, rng, kFf, kDim, true, wScale);
    }

    d = TSGgmlWanDitDesc{};
    d.x = xIn.data();
    d.out = out.data();
    d.context = ctx.data();
    d.tsin = tsin.data();
    d.tsin0 = tsin0.data();
    d.cosf = cosf.data();
    d.sinf = sinf.data();
    d.patch = makeW(a, rng, kInTok, kDim, true, wScale);
    d.text0 = makeW(a, rng, kTextDim, kDim, true, wScale);
    d.text2 = makeW(a, rng, kDim, kDim, true, wScale);
    d.time0 = makeW(a, rng, kFreqDim, kDim, true, wScale);
    d.time2 = makeW(a, rng, kDim, kDim, true, wScale);
    d.tproj = makeW(a, rng, kDim, 6 * kDim, true, wScale);
    d.head  = makeW(a, rng, kDim, kOutTok, true, wScale);
    d.head_mod = a.alloc(static_cast<std::size_t>(2) * kDim, rng, 0.2f);
    d.blocks = blocks.data();
    d.num_layers = kLayers;
    d.dim = kDim;
    d.ff = kFf;
    d.heads = kHeads;
    d.head_dim = kHeadDim;
    d.seq = kSeq;
    d.ctx_len = kCtxLen;
    d.freq_dim = kFreqDim;
    d.text_dim = kTextDim;
    d.seq0 = kSeq0;
    d.eps = 1e-6f;
    d.struct_bytes = static_cast<std::int32_t>(sizeof(TSGgmlWanDitDesc));
}

void setEnv(const char* name, const char* value)
{
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void resetCaches(HostArena& arena)
{
    TSGgml_WanResetForwardCache();
    TSGgml_ClearHostBufferCache();
    arena.invalidateAll();
}

// Largest absolute spread of one output feature across tokens. Zero (to float
// precision) is the exact signature of the bug: every token got the same vector.
double maxCrossTokenSpread(const std::vector<float>& out)
{
    double worst = 0.0;
    for (int f = 0; f < kOutTok; f++)
    {
        float lo = out[static_cast<std::size_t>(f)];
        float hi = lo;
        for (int t = 1; t < kSeq; t++)
        {
            const float v = out[static_cast<std::size_t>(t) * kOutTok + f];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        worst = std::max(worst, static_cast<double>(hi) - static_cast<double>(lo));
    }
    return worst;
}

double maxAbs(const std::vector<float>& v)
{
    double m = 0.0;
    for (float x : v) m = std::max(m, std::fabs(static_cast<double>(x)));
    return m;
}

bool allFinite(const std::vector<float>& v)
{
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    int backendType = BackendTypeCuda;
    if (argc > 1 && std::strcmp(argv[1], "cpu") == 0) backendType = BackendTypeCpu;
    if (backendType == BackendTypeCuda && TSGgml_IsBackendAvailable(BackendTypeCuda) == 0)
    {
        std::cout << "CUDA backend unavailable; falling back to CPU.\n";
        backendType = BackendTypeCpu;
    }
    if (TSGgml_IsBackendAvailable(backendType) == 0)
    {
        std::cerr << "No usable backend: " << TSGgml_GetLastError() << "\n";
        return 1;
    }
    std::cout << "backend = " << (backendType == BackendTypeCuda ? "cuda" : "cpu") << "\n";

    Rng rng;
    HostArena arena;
    std::vector<TSGWanDitBlockW> blocks;
    std::vector<float> xIn, out, ctx, tsin, tsin0, cosf, sinf;
    TSGgmlWanDitDesc d{};
    buildDesc(arena, rng, blocks, xIn, out, ctx, tsin, tsin0, cosf, sinf, d);

    // ---- phase 1: reference, resident cache bypassed entirely --------------
    // TS_WAN_NO_RESIDENT=1 sends every weight down the gallocr input-slot path,
    // which uploads on every call. That path is not the one under test, so it is a
    // trustworthy oracle no matter what the resident cache does.
    setEnv("TS_WAN_NO_RESIDENT", "1");
    setEnv("TS_WAN_DIT_CAPTURE", "1");
    resetCaches(arena);
    if (TSGgml_WanDitForward(&d) == 0)
    {
        std::cerr << "reference forward failed: " << TSGgml_GetLastError() << "\n";
        return 1;
    }
    const std::vector<float> reference = out;

    if (!allFinite(reference)) { std::cerr << "FAIL: reference output is not finite\n"; return 1; }
    const double refSpread = maxCrossTokenSpread(reference);
    const double refMag = maxAbs(reference);
    std::cout << "reference: max|v| = " << refMag << ", cross-token spread = " << refSpread << "\n";
    if (refSpread <= 1e-4 * std::max(refMag, 1e-6))
    {
        // Not the bug — the synthetic model itself would be degenerate, which
        // would make the real assertion below vacuous.
        std::cerr << "FAIL: reference output is already token-constant; test model is degenerate\n";
        return 1;
    }

    // ---- phase 2: abandon the persistent build after its weights are bound --
    // This is what the VRAM spill guard does on a card that cannot hold the
    // 27404-token graph. The forward must fall back and still be correct.
    std::fill(out.begin(), out.end(), 0.0f);
    setEnv("TS_WAN_NO_RESIDENT", "0");
    setEnv("TS_WAN_DIT_CAPTURE", "abandon");
    resetCaches(arena);
    if (TSGgml_WanDitForward(&d) == 0)
    {
        std::cerr << "post-abandon forward failed: " << TSGgml_GetLastError() << "\n";
        return 1;
    }

    if (!allFinite(out)) { std::cerr << "FAIL: post-abandon output is not finite\n"; return 1; }
    const double spread = maxCrossTokenSpread(out);
    std::cout << "post-abandon: max|v| = " << maxAbs(out) << ", cross-token spread = " << spread << "\n";
    if (spread <= 1e-4 * std::max(refMag, 1e-6))
    {
        std::cerr << "FAIL: post-abandon output is token-CONSTANT (spread " << spread
                  << ") — the abandoned build published un-filled resident weight buffers\n";
        return 1;
    }

    double worst = 0.0;
    for (std::size_t i = 0; i < reference.size(); i++)
        worst = std::max(worst, std::fabs(static_cast<double>(out[i]) - static_cast<double>(reference[i])));
    const double tol = 1e-4 * std::max(refMag, 1e-3);
    std::cout << "max |post-abandon - reference| = " << worst << " (tol " << tol << ")\n";
    if (!(worst <= tol))
    {
        std::cerr << "FAIL: post-abandon output does not match the reference\n";
        return 1;
    }

    // ---- phase 3: a second forward reuses the now-resident cache ------------
    // The entries published by the abandoned build must be usable as-is; this is
    // the call that read zeros before the fix.
    std::fill(out.begin(), out.end(), 0.0f);
    if (TSGgml_WanDitForward(&d) == 0)
    {
        std::cerr << "repeat forward failed: " << TSGgml_GetLastError() << "\n";
        return 1;
    }
    worst = 0.0;
    for (std::size_t i = 0; i < reference.size(); i++)
        worst = std::max(worst, std::fabs(static_cast<double>(out[i]) - static_cast<double>(reference[i])));
    std::cout << "max |repeat - reference| = " << worst << "\n";
    if (!(worst <= tol))
    {
        std::cerr << "FAIL: repeat forward off the resident cache does not match the reference\n";
        return 1;
    }

    setEnv("TS_WAN_DIT_CAPTURE", "1");
    std::cout << "PASS\n";
    return 0;
}
