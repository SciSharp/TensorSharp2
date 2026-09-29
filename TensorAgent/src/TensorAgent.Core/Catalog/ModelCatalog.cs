// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Core.Catalog;

/// <summary>
/// The built-in model list. Downloadable entries' sizes and hashes were read from the
/// Hugging Face tree API (LFS object ids) on 2026-09-01 (Bonsai 2 27B on 2026-09-28),
/// so a download is verified against the exact bytes the publisher uploaded. A
/// sideload-only entry would carry the same immutable size/hash identity but
/// deliberately no URL, for a GGUF that embeds no publisher repository; the app then
/// verifies a user-selected local file instead. No built-in entry needs that today.
///
/// <para>
/// Sizing rule (why these quantizations): on GGML Metal the quantized weights are wrapped
/// as MTLBuffers straight over the GGUF mmap, and Metal keeps those pages resident, so
/// the resident set is about the file size plus the KV cache, the projector (materialised
/// as F32, so roughly twice its file size) and ~0.5 GB of compute buffers. An iPhone with
/// 12 GB grants an app with the increased-memory entitlement roughly 8.5 GB, which is why
/// the 12 GB tier tops out around 7.5 GB of weights and everything larger is offered only
/// to 16 GB iPads.
/// </para>
/// </summary>
public static class ModelCatalog
{
    private const string GemmaLicense = "Gemma Terms of Use";
    private const string ApacheLicense = "Apache-2.0";

    private static string Hf(string repo, string file) => $"https://huggingface.co/{repo}/resolve/main/{file}";

    public static IReadOnlyList<CatalogModel> BuiltIn { get; } = new[]
    {
        new CatalogModel
        {
            Id = "gemma-4-e2b-q8",
            DisplayName = "Gemma 4 E2B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "2B effective (5B with per-layer embeddings)",
            Quantization = "Q8_0",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-E2B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf"),
                    4_967_497_152, "996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-gemma-4-E2B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E2B-it-GGUF", "mmproj-gemma-4-E2B-it-Q8_0.gguf"),
                    557_368_064, "9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292"),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Audio | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            ContextLength = 8192,
            // f16, not q8_0: Gemma 4 declines a block-quantized cache
            // (Gemma4Model.SupportsBlockQuantizedKvCache). Its sliding-window layers
            // use a circular cache whose managed helpers are float-only, and the 26B
            // MoE reaches them on an ordinary prompt: setting q8_0 here crashed with
            // "Requires a Float32 tensor, but found Q8_0" out of CopyToCacheCircular.
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = GemmaLicense,
            Notes = "Fastest option. Sees images and video frames, hears audio, thinks when asked.",
        },
        new CatalogModel
        {
            Id = "gemma-4-e4b-iq4xs",
            DisplayName = "Gemma 4 E4B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "4B effective (8B with per-layer embeddings)",
            Quantization = "IQ4_XS",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-E4B-it-IQ4_XS.gguf",
                    Hf("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-IQ4_XS.gguf"),
                    4_715_416_704, "0847f7300471e9a61abaeb46b45b3d61e393af8fa6d6aa70c563623773670dd9"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-gemma-4-E4B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E4B-it-GGUF", "mmproj-gemma-4-E4B-it-Q8_0.gguf"),
                    559_874_816, "197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1"),
                new CatalogFile(CatalogFileRole.Draft, "mtp-gemma-4-E4B-it-Q8_0.gguf",
                    Hf("ggml-org/gemma-4-E4B-it-GGUF", "mtp-gemma-4-E4B-it-Q8_0.gguf"),
                    98_653_280, "f38ae62962657c7a6303c49bbb147e9ae23634e911cfa532fac0818c2e18b665", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Audio | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            ContextLength = 8192,
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = GemmaLicense,
            Notes = "The recommended default: TensorSharp's verified fast-path tier, multimodal, with an optional speculative draft head.",
        },
        new CatalogModel
        {
            Id = "gemma-4-12b-iq2m",
            DisplayName = "Gemma 4 12B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "12B",
            Quantization = "UD-IQ2_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-12b-it-UD-IQ2_M.gguf",
                    Hf("unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-UD-IQ2_M.gguf"),
                    4_213_353_280, "4bd2461d35398dbcf5f3d5f0c9ad91cac78ae35b556e3a81f315a0cc0815ae8c"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    Hf("unsloth/gemma-4-12b-it-GGUF", "mmproj-F16.gguf"),
                    175_115_840, "91f086971e56d7a7d8d39e271873fccdb49541bd259d6e02c401a4f1cb7a219e", Optional: true),
                // The per-token assistant head, the same shape gemma-4-e4b-iq4xs carries.
                new CatalogFile(CatalogFileRole.Draft, "mtp-gemma-4-12b-it.gguf",
                    Hf("unsloth/gemma-4-12b-it-GGUF", "mtp-gemma-4-12b-it.gguf"),
                    465_109_248, "145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            // 32768, not 8192. The window was small because the LOAD was expensive, not
            // because the cache was: fusing this model's 48 ffn_gate/ffn_up pairs cost
            // gigabytes of anonymous memory duplicating bytes already mapped from the
            // GGUF, which is what jetsam killed the app for. Gemma 4 now runs the pair
            // as two matmuls instead (Gemma4Model.SupportsSplitGateUpFfn), and the
            // window is affordable. MEASURED on the Q4_K_XL build of this same model,
            // ggml_metal, where the fusion was 3,108 MB:
            //   fused,  8192   peak footprint 4,534 MB   31.7 tok/s
            //   split,  8192                   1,423 MB   31.5 tok/s
            //   split, 32768                   2,187 MB   31.3 tok/s
            // Output is byte-identical between the first two, and a 14,294-token needle
            // prompt returns the planted value on both. THIS entry is the UD-IQ2_M
            // build, and it reaches the split path by the same route the measurement
            // above did: on iOS ModelBase.AllowWeightFusionCopies is false, so the
            // fused copy is never made whatever the types are. Nothing here depends on
            // ggml declining to requantize -- VERIFIED against the pinned file, whose
            // 48 ffn_gate/ffn_up pairs match on BOTH sides (43 IQ2_S pairs and five
            // IQ3_XXS pairs), so the mismatch branch that would call for a requantize
            // is never entered at all.
            ContextLength = 32768,
            // f16, like every Gemma entry: Gemma 4 refuses a block-quantized cache
            // (Gemma4Model.SupportsBlockQuantizedKvCache) because its sliding-window
            // layers use a circular cache whose managed helpers are float-only.
            KvCacheDtype = "f16",
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            License = GemmaLicense,
            Notes = "The dense Gemma between E4B and the 26B mixture of experts, in the 2.7 bpw IQ2_M recipe. "
                + "4.2 GB of weights, mapped from the file rather than copied, so the model itself "
                + "costs the phone almost nothing and the 32k window is the larger part of its "
                + "footprint. Vision and a speculative draft head are optional downloads.",
        },
        new CatalogModel
        {
            Id = "bonsai-2-27b-ptq1-0",
            DisplayName = "Bonsai 2 27B",
            Family = CatalogFamily.Bonsai,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "27B",
            Quantization = "PTQ1_0",
            Files = new[]
            {
                // qwen35 hybrid (48 Gated DeltaNet + 16 full-attention layers) with ternary
                // g128 weights in PRISM's rotated basis. Sizes and LFS ids were read from
                // the tree API on 2026-09-28 (revision b072e1d3b35a0a630cece372c2127528e0994386).
                // PTQ1_0 and PQ2_0 hold the same trits, and the loader repacks either one
                // losslessly to GGML Q2_0, so the dense PTQ1_0 packing is simply the
                // smaller download (5.95 GB against 7.21 GB) for the same resident model.
                new CatalogFile(CatalogFileRole.Weights, "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
                    Hf("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
                    5_946_648_928, "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"),
                new CatalogFile(CatalogFileRole.Projector, "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
                    Hf("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"),
                    629_246_976, "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            // 16, not 12: unlike the entries above, these weights are not served from the
            // GGUF mapping. The Q2_0 repack lives in anonymous memory, ~7.7 GB for this
            // file (PTQ1_0 grows by about 29%), which with the KV cache and compute
            // buffers is more than the ~8.5 GB a 12 GB iPhone grants the app.
            MinDeviceMemoryGB = 16,
            ContextLength = 32768,
            KvCacheDtype = "q8_0",
            // The publisher's thinking-mode recommendation (the GGUF does not carry min_p).
            Sampling = new CatalogSampling(1.0f, 20, 0.95f, 0.05f),
            SupportsThinking = true,
            // Bonsai2 is validated on desktop Metal/CPU, not yet on an iPad.
            Experimental = true,
            License = ApacheLicense,
            Notes = "PrismML's ternary Qwen 3.5 hybrid with Hadamard-rotated weights. The 5.9 GB download is "
                + "repacked losslessly to about 7.7 GB when it loads, so it is offered only on 16 GB devices. "
                + "Vision is an optional download.",
        },
        new CatalogModel
        {
            Id = "qwen3.5-9b-iq4xs",
            DisplayName = "Qwen3.5 9B",
            Family = CatalogFamily.Qwen35,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "9B",
            Quantization = "IQ4_XS",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Qwen3.5-9B-IQ4_XS.gguf",
                    Hf("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-IQ4_XS.gguf"),
                    5_168_653_536, "7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    Hf("unsloth/Qwen3.5-9B-GGUF", "mmproj-F16.gguf"),
                    918_166_080, "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            MinDeviceMemoryGB = 12,
            // 32768, not 8192: the reply-length setting is bounded by the CONTEXT
            // (ChatGenerationPipeline.ClampGenerationReserve trims the generation
            // reserve to what the window leaves after the prompt, and the thinking
            // budget is 75% of THAT), so an 8192 window capped a reply at ~7.7k
            // tokens however high the user set the limit -- and a reasoning model
            // that spent it produced no answer at all. A quantized cache pays for
            // the bigger window: MEASURED on Qwen3.5-9B UD-Q4_K_XL, ggml_metal,
            // peak physical footprint is 1697 MB at q8_0/32768 against the 1118 MB
            // that f16/8192 already cost, and q8_0/16384 (1152 MB) is a wash.
            ContextLength = 32768,
            // q8_0, not f16: the KV cache is the only thing that grows with the
            // conversation, and on Metal it is charged twice. MEASURED on ggml_metal
            // after the fused graphs learned block-quantized K/V: 22.4 KiB/token
            // against f16's 41.8, at decode parity (Qwen3.6-35B-A3B 75.5 vs 76.0
            // tok/s, within noise), with a two-needle recall test at 7,490 tokens
            // returning both planted values. Qwen3.5/3.6 take the fused graph, whose
            // native side is dtype-generic; Gemma 4 does not and stays on f16.
            KvCacheDtype = "q8_0",
            Sampling = new CatalogSampling(0.7f, 20, 0.8f, 0.0f),
            SupportsThinking = true,
            License = ApacheLicense,
            Notes = "Strong general model with vision; the projector is optional and costs ~1.8 GB of memory when loaded.",
        },
    };

    public static CatalogModel? Find(string id) =>
        BuiltIn.FirstOrDefault(m => string.Equals(m.Id, id, StringComparison.OrdinalIgnoreCase));

    /// <summary>The entries a device with <paramref name="physicalMemoryGB"/> of RAM is offered.</summary>
    public static IReadOnlyList<CatalogModel> ForDevice(int physicalMemoryGB) =>
        BuiltIn.Where(m => m.MinDeviceMemoryGB <= physicalMemoryGB).ToList();

    /// <summary>
    /// Rounds a reported physical-memory figure (iOS reports slightly under the marketing
    /// number, e.g. 11.6 GB for a "12 GB" phone) to the marketing tier used by
    /// <see cref="CatalogModel.MinDeviceMemoryGB"/>.
    /// </summary>
    public static int DeviceMemoryTier(long physicalMemoryBytes)
    {
        double gb = physicalMemoryBytes / 1_000_000_000.0;
        int[] tiers = { 4, 6, 8, 12, 16, 24, 32, 48, 64, 128 };
        int best = tiers[0];
        foreach (int t in tiers)
        {
            if (gb + 0.75 >= t)
                best = t;
        }
        return best;
    }
}
