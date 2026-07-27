// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;
using TensorSharp.GGML;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;
using TensorSharp.Runtime.Redis;

const string ListenAddress = "http://0.0.0.0:5000";
const long MaxRequestBodyBytes = 500L * 1024L * 1024L;

Console.OutputEncoding = System.Text.Encoding.UTF8;

// Merge in options from a --config <file.json> before anything reads argv.
// File-derived tokens are spliced in ahead of the real command line, so any
// option also passed on the command line overrides the file (every option
// pass below is last-one-wins). The --config flag itself is stripped here.
try
{
    args = ConfigFileArgs.Expand(args);
}
catch (Exception ex) when (ex is ArgumentException or FileNotFoundException)
{
    Console.Error.WriteLine("Configuration error: " + ex.Message);
    Environment.ExitCode = 1;
    return;
}

bool showSarah = Array.Exists(args, a => a == "--xzf");
ConsoleBanner.Print(showSarah);

// Informational invocations print and exit before the web host is built. A
// bare `TensorSharp.Server` shows the usage page instead of silently starting
// a model-less server. Passing another option can still start a status-only
// process, but inference requires --model at startup.
if (args.Length == 0 || ServerUsage.IsHelpRequested(args))
{
    ServerUsage.PrintUsage(Console.Out);
    return;
}

if (ServerUsage.IsListGpusRequested(args))
{
    ServerUsage.PrintVulkanGpus(Console.Out);
    return;
}

string baseDirectory = AppContext.BaseDirectory;
ServerHostingOptions hostingOptions = ServerOptionsBuilder.Build(args, baseDirectory);
LogLevel resolvedLogLevel = LoggingSetup.ResolveMinimumLevel();
string configuredBackendInput = ServerOptionsBuilder.ReadConfiguredBackendInput(args);
// Translate --paged-kv* flags into env vars before startup logging reads
// PagedKvCacheConfig.FromEnvironment().
bool pagedKvFlagsApplied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(args);
// Translate --redis-url into TS_KV_CACHE_REDIS_URL and
// TS_RESPONSES_STORE_REDIS_URL so a single flag enables Redis for both the
// paged KV cache tier and the Responses API store.
bool redisFlagsApplied = ServerOptionsBuilder.ApplyRedisCliFlags(args);
// Translate --continuous-batching / --no-continuous-batching into env vars
// that gate BatchExecutor (TS_SCHED_DISABLE_BATCHED) and Qwen3.5 ForwardBatch
// (TS_QWEN35_BATCHED). Must run before InferenceEngine constructs its
// BatchExecutor and the per-model batched-paged adapters initialise.
bool continuousBatchingFlagApplied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args);
// Translate --mtp-spec / --mtp-draft / --mtp-pmin into the TS_MTP_* env vars
// read by SchedulerConfig.FromEnvironment when the engine is constructed.
bool mtpSpecFlagsApplied = ServerOptionsBuilder.ApplyMtpSpeculativeCliFlags(args);
// Translate --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj into the
// TS_QWEN_IMAGE_* env vars QwenImageModel reads to locate the VAE, Qwen2.5-VL
// text-encoder, and mmproj GGUFs. Must run before the startup model is loaded.
bool qwenImageFlagsApplied = ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args);
// Translate --kv-cache-dtype into the process-wide KvCacheDtypeConfig (or honor
// the KV_CACHE_DTYPE env var) so block-quantized / half-precision KV caches are
// selectable on the server, mirroring the CLI. The fused native decode path used
// by the scheduler is the one that supports block-quantized (q8_0 / q4_0) caches.
// Must run before the startup model is loaded so InitKVCache sees the choice.
TensorSharp.Models.KvCacheDtypeConfig.ConfigureFromEnvironment();
bool kvCacheDtypeFlagApplied = ServerOptionsBuilder.ApplyKvCacheDtypeCliFlag(args);
// Translate --gpu-device into TS_GGML_VULKAN_DEVICE so multi-GPU hosts can pick
// which Vulkan device the ggml_vulkan backend initializes on. Must run before
// the startup model is loaded (the device is fixed at first backend init).
bool gpuDeviceFlagApplied = ServerOptionsBuilder.ApplyGpuDeviceCliFlag(args);

var builder = WebApplication.CreateBuilder(args);
LoggingSetup.Configure(builder.Logging, hostingOptions, resolvedLogLevel);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = MaxRequestBodyBytes;
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = MaxRequestBodyBytes;
});

builder.Services.AddSingleton(hostingOptions);
builder.Services.AddSingleton<ModelService>();
builder.Services.AddSingleton<InferenceQueue>();
builder.Services.AddSingleton<SessionManager>();
// Engine is owned by ModelService now (so its lifecycle is tied to the
// loaded model). Re-export it as a DI service for adapters that wish to
// submit requests directly.
builder.Services.AddSingleton<InferenceEngineHost>(sp =>
    sp.GetRequiredService<ModelService>().EngineHost);

// Demote the high-frequency status-polling endpoints to Debug so the
// default Information-level log isn't dominated by their request entries.
// Set TENSORSHARP_LOG_LEVEL=Debug to see them when troubleshooting.
builder.Services.AddTensorSharpRequestLogging(options =>
{
    options.LowNoisePaths.Add("/api/queue/status");
});

// One adapter per protocol; instances are stateless and free to share between requests.
builder.Services.AddSingleton<WebUiAdapter>();
builder.Services.AddSingleton<OllamaAdapter>();
builder.Services.AddSingleton<OpenAIChatAdapter>();
// Responses API store: use Redis when TS_RESPONSES_STORE_REDIS_URL is set,
// otherwise fall back to the bounded in-memory cache.
string responsesRedisUrl = Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL")?.Trim();
if (!string.IsNullOrEmpty(responsesRedisUrl))
{
    builder.Services.AddSingleton<IResponsesStore>(sp =>
    {
        var logger = sp.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.Responses.RedisResponsesStore");
        var redis = new RedisConnection(responsesRedisUrl, logger);
        return new RedisResponsesStore(redis, logger);
    });
}
else
{
    builder.Services.AddSingleton<IResponsesStore, InMemoryResponsesStore>();
}
builder.Services.AddSingleton<OpenAIResponsesAdapter>();

WebRootSetup.Resolve(builder.Environment, baseDirectory);

var app = builder.Build();

ILogger startupLogger = app.Services.GetRequiredService<ILoggerFactory>()
    .CreateLogger("TensorSharp.Server.Startup");
startupLogger.LogInformation(LogEventIds.LoggingInitialized,
    "Logging initialized: minimumLevel={MinimumLevel} fileLogging={FileLogging} logDir={LogDir}",
    resolvedLogLevel, hostingOptions.FileLoggingEnabled,
    hostingOptions.FileLoggingEnabled ? hostingOptions.LogDirectory : "(disabled)");

if (pagedKvFlagsApplied)
{
    var pagedCfg = PagedKvCacheConfig.FromEnvironment();
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "paged-kv configured via CLI: enabled={Enabled} blockSize={BlockSize} ramMB={RamMB} ssdDir={SsdDir} maxSsdMB={MaxSsdMB}",
        pagedCfg.Enabled, pagedCfg.BlockSize, pagedCfg.MaxRamBytes / (1024 * 1024),
        string.IsNullOrEmpty(pagedCfg.SsdDirectory) ? "(disabled)" : pagedCfg.SsdDirectory,
        pagedCfg.MaxSsdBytes / (1024 * 1024));
}

if (redisFlagsApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Redis configured via CLI: kvCacheUrl={KvRedisUrl} responsesStoreUrl={ResponsesRedisUrl}",
        Environment.GetEnvironmentVariable("TS_KV_CACHE_REDIS_URL") ?? "(disabled)",
        Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL") ?? "(disabled)");
}

if (mtpSpecFlagsApplied)
{
    var schedCfg = TensorSharp.Runtime.Scheduling.SchedulerConfig.FromEnvironment();
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "MTP speculative decoding configured via CLI: enabled={Enabled} maxDraft={MaxDraft} pMin={PMin} (engages on NextN/MTP draft-head models only)",
        schedCfg.MtpSpeculativeEnabled, schedCfg.MtpMaxDraftTokens, schedCfg.MtpMinDraftProb);
}

if (gpuDeviceFlagApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Vulkan GPU device configured via CLI: --gpu-device {DeviceIndex} (applies when the ggml_vulkan backend initializes)",
        Environment.GetEnvironmentVariable(GgmlBasicOps.VulkanDeviceEnvVar));
}

if (qwenImageFlagsApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Qwen-Image-Edit companions configured via CLI: vae={Vae} vl={Vl} mmproj={Mmproj}",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_VAE") ?? "(scan)",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE") ?? "(scan)",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ") ?? "(scan)");
}

StartupBanner.EmitBackendFallback(startupLogger, hostingOptions, configuredBackendInput);

app.UseTensorSharpRequestLogging();
// Serve the bundled static UI at /index.html. The explicit GET / endpoint
// remains the plain liveness response; headless deployments can still start
// when no wwwroot content is present.
app.UseDefaultFiles();
app.UseStaticFiles();
// The default content-type provider has no HEIC/HEIF mapping, so uploaded iPhone
// photos 404'd under /uploads (browsers can't render HEIC in <img> anyway — the
// Web UI displays the server-generated PNG previewUrl — but the original should
// at least stay downloadable).
var uploadContentTypes = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
uploadContentTypes.Mappings[".heic"] = "image/heic";
uploadContentTypes.Mappings[".heif"] = "image/heif";
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(hostingOptions.UploadDirectory),
    RequestPath = "/uploads",
    ContentTypeProvider = uploadContentTypes,
});

app.MapHealthEndpoints(app.Environment);
app.MapSessionEndpoints();
app.MapUploadEndpoints();
app.MapWebUiEndpoints();
app.MapOllamaEndpoints();
app.MapOpenAIEndpoints();

StartupModelLoader.LoadIfConfigured(
    hostingOptions,
    app.Services.GetRequiredService<ModelService>(),
    configuredBackendInput,
    startupLogger);

StartupBanner.Emit(startupLogger, hostingOptions, ListenAddress);

// Tear down the process-global GGML backend after the host stops. On macOS
// the ggml-metal device's C++ static destructor asserts that its resource
// set is empty; if g_backend (and its MTLBuffer wrappers) outlive the .NET
// host the assertion aborts the process during exit. ApplicationStopped
// fires after all hosted services have shut down, so all in-flight
// inference is already complete. The shutdown call is idempotent and a
// no-op when no GGML backend was ever initialised. Also hooked onto
// ProcessExit as a safety net for non-graceful exits.
app.Lifetime.ApplicationStopped.Register(static () => GgmlBasicOps.Shutdown());
AppDomain.CurrentDomain.ProcessExit += static (_, _) => GgmlBasicOps.Shutdown();

app.Run(ListenAddress);
