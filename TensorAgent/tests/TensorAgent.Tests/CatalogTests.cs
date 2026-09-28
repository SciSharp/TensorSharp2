using System.Text.RegularExpressions;
using TensorAgent.Core.Catalog;

namespace TensorAgent.Tests;

public sealed class CatalogTests
{
    [Fact]
    public void BuiltInContainsExactlyTheApprovedModelIds()
    {
        string[] expected =
        {
            "gemma-4-e2b-q8",
            "gemma-4-e4b-iq4xs",
            "gemma-4-12b-iq2m",
            "bonsai-2-27b-ptq1-0",
            "qwen3.5-9b-iq4xs",
        };

        Assert.Equal(expected, ModelCatalog.BuiltIn.Select(m => m.Id).ToArray());
    }

    [Fact]
    public void EveryEntryIsWellFormed()
    {
        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (CatalogModel m in ModelCatalog.BuiltIn)
        {
            Assert.True(ids.Add(m.Id), $"duplicate id {m.Id}");
            Assert.Matches("^[a-z0-9.-]+$", m.Id);
            Assert.NotEmpty(m.DisplayName);
            Assert.NotEmpty(m.Files);
            Assert.Single(m.Files, f => f.Role == CatalogFileRole.Weights);
            Assert.False(m.Weights.Optional, $"{m.Id}: weights cannot be optional");
            foreach (CatalogFile f in m.Files)
            {
                if (m.SideloadOnly)
                {
                    Assert.Empty(f.Url);
                }
                else
                {
                    Assert.StartsWith("https://huggingface.co/", f.Url);
                    Assert.EndsWith("/resolve/main/" + f.Url.Split("/resolve/main/")[1], f.Url);
                }
                Assert.True(f.Bytes > 1_000_000, $"{m.Id}/{f.FileName}: size {f.Bytes}");
                Assert.Matches("^[0-9a-f]{64}$", f.Sha256);
                Assert.False(f.FileName.Contains('/'), $"{m.Id}: file names are bare ({f.FileName})");
            }
            // Keep the recognized tiers narrow so a typo cannot silently expose an
            // entry on an unintended device class. A future 24 GB entry would remain
            // hidden from current phones while still using the same gating mechanism.
            Assert.Contains(m.MinDeviceMemoryGB, new[] { 6, 8, 12, 16, 24 });
            Assert.NotEmpty(m.License);
        }
    }

    [Fact]
    public void ProjectorsAreDistinctFromWeights()
    {
        foreach (CatalogModel m in ModelCatalog.BuiltIn)
        {
            var names = m.Files.Select(f => f.FileName).ToList();
            Assert.Equal(names.Count, names.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        }
    }

    [Fact]
    public void Gemma4TwelveBUsesThePinnedIq2MArtifact()
    {
        CatalogModel model = Assert.Single(ModelCatalog.BuiltIn,
            m => m.Family == CatalogFamily.Gemma4 && m.Parameters == "12B");

        Assert.Equal("gemma-4-12b-iq2m", model.Id);
        Assert.Equal("UD-IQ2_M", model.Quantization);
        Assert.Equal("gemma-4-12b-it-UD-IQ2_M.gguf", model.Weights.FileName);
        Assert.Equal(4_213_353_280, model.Weights.Bytes);
        Assert.Equal("4bd2461d35398dbcf5f3d5f0c9ad91cac78ae35b556e3a81f315a0cc0815ae8c",
            model.Weights.Sha256);
    }

    /// <summary>
    /// What a device of a given size actually grants the app. A 12 GB iPhone gives
    /// about 8.5 GB with <c>com.apple.developer.kernel.increased-memory-limit</c>, and
    /// the fraction holds well enough across the range to gate a catalog with.
    /// </summary>
    private static double JetsamBudget(int deviceGB) => deviceGB * 1e9 * (8.5 / 12.0);

    /// <summary>
    /// The memory that is actually charged against the jetsam limit: ANONYMOUS memory
    /// only.
    ///
    /// <para>
    /// This used to be <c>Weights.Bytes + 0.5e9 + 2*projector</c>, on the stated premise
    /// that "Metal wires the mmap'd weights, so resident memory is roughly the GGUF".
    /// That premise is wrong, and measurably so. Weights are a file mapping, and Darwin
    /// charges mapped clean pages essentially nothing; ggml-metal wraps them with
    /// newBufferWithBytesNoCopy and the residency set does not fault them in. Measurements
    /// on this repo showed that treating the entire weights file as anonymous memory can
    /// overstate the charged footprint by roughly an order of magnitude.
    /// </para>
    ///
    /// <para>
    /// What IS anonymous: the KV cache (charged TWICE on Metal -- once for the host
    /// tensor and once for the Metal-side buffer, since the zero-copy wrap is refused for
    /// read-write tensors), the projector's dequantized copies (about twice its file),
    /// and the runtime plus graph scratch. 64 KiB/token is the per-token KV rate measured
    /// for Qwen3.5 9B and is used here as a conservative upper bound.
    /// </para>
    ///
    /// <para>
    /// The exception is a weights file the loader cannot map: Bonsai2's PTQ1_0 / PQ2_0
    /// packings are repacked losslessly to GGML Q2_0 at load, into anonymous memory that
    /// is about 29% / 6% larger than the file (<see cref="RepackedWeightBytes"/>).
    /// </para>
    /// </summary>
    private static double EstimatedAnonymous(CatalogModel model) =>
        0.5e9
        + model.ContextLength * 64.0 * 1024.0
        + (model.Projector is { Optional: false } p ? 2.0 * p.Bytes : 0)
        + RepackedWeightBytes(model);

    /// <summary>Anonymous memory the weights occupy when the loader repacks them instead
    /// of mapping the GGUF (zero for every mapped format).</summary>
    private static double RepackedWeightBytes(CatalogModel model) => model.Quantization switch
    {
        "PTQ1_0" => 1.29 * model.Weights.Bytes,
        "PQ2_0" => 1.06 * model.Weights.Bytes,
        _ => 0,
    };

    /// <summary>
    /// The other half, and the one the weights really answer to: they are not charged to
    /// jetsam, but they still have to be READABLE at a usable speed, which means the file
    /// has to fit the DEVICE's RAM alongside iOS rather than the app's jetsam budget.
    /// Past this line every token faults expert weights from flash. It is a performance
    /// bound, not a kill bound, which is why it is a separate number from
    /// <see cref="JetsamBudget"/> instead of folded into it.
    /// </summary>
    private static double WeightsResidencyCeiling(int deviceGB) => deviceGB * 1e9 * 0.87;

    [Fact]
    public void EveryTierStaysUnderTheJetsamBudgetOfTheSmallestDeviceItIsOfferedOn()
    {
        // Checking only the 12 GB tier let two entries through that could never load
        // on the tier they advertised: Gemma 4 E2B claimed 6 GB and needs 6.6, and
        // E4B Q4_K_XL claimed 8 and needs 6.8. A phone would have offered a five
        // gigabyte download and then been killed opening it — the worst outcome the
        // catalog can produce, because the user pays for it twice.
        foreach (CatalogModel m in ModelCatalog.BuiltIn.Where(m => !m.IsImageGenerator))
        {
            double anonymous = EstimatedAnonymous(m);
            double budget = JetsamBudget(m.MinDeviceMemoryGB);
            Assert.True(anonymous < budget,
                $"{m.Id} is offered at {m.MinDeviceMemoryGB} GB, which grants about "
                + $"{budget / 1e9:F1} GB, but charges about {anonymous / 1e9:F1} GB of anonymous memory");

            double ceiling = WeightsResidencyCeiling(m.MinDeviceMemoryGB);
            Assert.True(m.Weights.Bytes < ceiling,
                $"{m.Id} is offered at {m.MinDeviceMemoryGB} GB but its weights are "
                + $"{m.Weights.Bytes / 1e9:F1} GB, past the {ceiling / 1e9:F1} GB that device can "
                + "hold; it would fault every token from flash");
        }
    }

    [Fact]
    public void EveryModelADeviceIsOfferedFitsThatDevice()
    {
        // The other direction: whatever ForDevice hands back must fit the device that
        // asked, for every size a real iPhone comes in.
        foreach (int deviceGB in new[] { 6, 8, 12, 16 })
        {
            foreach (CatalogModel m in ModelCatalog.ForDevice(deviceGB).Where(m => !m.IsImageGenerator))
            {
                Assert.True(EstimatedAnonymous(m) < JetsamBudget(deviceGB),
                    $"a {deviceGB} GB device is offered {m.Id}, which charges about "
                    + $"{EstimatedAnonymous(m) / 1e9:F1} GB against a {JetsamBudget(deviceGB) / 1e9:F1} GB budget");
                Assert.True(m.Weights.Bytes < WeightsResidencyCeiling(deviceGB),
                    $"a {deviceGB} GB device is offered {m.Id}, whose {m.Weights.Bytes / 1e9:F1} GB of "
                    + $"weights exceed the {WeightsResidencyCeiling(deviceGB) / 1e9:F1} GB it can hold");
            }
        }
    }

    [Fact]
    public void DeviceTiersHideTheCatalogBelowTwelveGbAndHoldBonsai2ForSixteenGb()
    {
        Assert.Empty(ModelCatalog.ForDevice(8));
        // Bonsai 2 27B is the one 16 GB entry: its repacked weights do not fit a 12 GB phone.
        Assert.Equal(
            ModelCatalog.BuiltIn.Where(m => m.Id != "bonsai-2-27b-ptq1-0").Select(m => m.Id),
            ModelCatalog.ForDevice(12).Select(m => m.Id));
        Assert.Equal(
            ModelCatalog.BuiltIn.Select(m => m.Id),
            ModelCatalog.ForDevice(16).Select(m => m.Id));
    }

    [Theory]
    [InlineData(11_560_000_000L, 12)]
    [InlineData(12_000_000_000L, 12)]
    [InlineData(8_100_000_000L, 8)]
    [InlineData(7_700_000_000L, 8)]
    [InlineData(16_700_000_000L, 16)]
    [InlineData(5_800_000_000L, 6)]
    public void MemoryTierRoundsToTheMarketingNumber(long bytes, int tier)
    {
        Assert.Equal(tier, ModelCatalog.DeviceMemoryTier(bytes));
    }

    /// <summary>
    /// A weights file sitting loose in the models directory is swept too.
    ///
    /// <para>
    /// The directory holds one sub-folder per catalog id and nothing else, so a file
    /// directly inside it belongs to no entry by construction. It gets there when
    /// weights are pushed onto the device by hand and land beside the per-model folders
    /// instead of inside one — and it is worse off than an orphaned directory, because
    /// the Models list is built from catalog entries: a stray file has no row, no size
    /// attributed to any model, and no delete button, while being several gigabytes.
    /// </para>
    /// <para>
    /// The other half is that a real model's files are NOT strays. They live one level
    /// down, so enumerating only the top level must not reach them.
    /// </para>
    /// </summary>
    [Fact]
    public void ASweepAlsoRemovesAWeightsFileLeftLooseInTheModelsDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "ta-stray-" + Guid.NewGuid().ToString("n"));
        try
        {
            var store = new ModelStore(root);
            CatalogModel offered = ModelCatalog.ForDevice(12)[0];

            Directory.CreateDirectory(Path.Combine(root, offered.Id));
            File.WriteAllBytes(Path.Combine(root, offered.Id, "weights.gguf"), new byte[4096]);
            File.WriteAllBytes(Path.Combine(root, "gemma-4-12b-it-UD-IQ2_M.gguf"), new byte[1024]);
            File.WriteAllBytes(Path.Combine(root, "mmproj-F16.gguf"), new byte[512]);

            long freed = store.SweepOrphanedModels();

            Assert.Equal(1536, freed);
            Assert.False(File.Exists(Path.Combine(root, "gemma-4-12b-it-UD-IQ2_M.gguf")));
            Assert.False(File.Exists(Path.Combine(root, "mmproj-F16.gguf")));
            Assert.True(File.Exists(Path.Combine(root, offered.Id, "weights.gguf")),
                "the sweep reached inside a model's own directory and deleted its weights");
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// Weights left behind when an entry changes which file it points at.
    ///
    /// <para>
    /// The id carries the quantization, so re-pointing Gemma 4 12B from UD-IQ3_XXS to
    /// UD-IQ2_M renames its directory and orphans the old one -- 4.6 GB with no row in
    /// the Models list and therefore no way for the user to remove it. An entry gated
    /// to a bigger device is NOT an orphan, which is the half that would be a data-loss
    /// bug: an iPad-only model's weights must survive a sweep run on a phone.
    /// </para>
    /// </summary>
    [Fact]
    public void ASweepRemovesWeightsNoEntryClaimsAndKeepsTheOnesThatAreMerelyGatedOff()
    {
        string root = Path.Combine(Path.GetTempPath(), "ta-sweep-" + Guid.NewGuid().ToString("n"));
        try
        {
            var store = new ModelStore(root);
            CatalogModel offered = ModelCatalog.ForDevice(12)[0];
            CatalogModel gatedOff = offered with
            {
                Id = "synthetic-16gb-entry",
                MinDeviceMemoryGB = 16,
            };
            CatalogModel[] wholeCatalog = { offered, gatedOff };

            foreach (string id in new[] { offered.Id, gatedOff.Id, "gemma-4-12b-iq3xxs" })
            {
                Directory.CreateDirectory(Path.Combine(root, id));
                File.WriteAllBytes(Path.Combine(root, id, "weights.gguf"), new byte[2048]);
            }

            long freed = store.SweepOrphanedModels(wholeCatalog);

            Assert.Equal(2048, freed);
            Assert.True(Directory.Exists(Path.Combine(root, offered.Id)));
            Assert.True(Directory.Exists(Path.Combine(root, gatedOff.Id)),
                "a sweep on a phone deleted the weights of a model only an iPad is offered");
            Assert.False(Directory.Exists(Path.Combine(root, "gemma-4-12b-iq3xxs")));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    [Fact]
    public void FindIsCaseInsensitive()
    {
        Assert.NotNull(ModelCatalog.Find("GEMMA-4-E4B-IQ4XS"));
        Assert.Null(ModelCatalog.Find("nope"));
    }
}
