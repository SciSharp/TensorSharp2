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
using System.Linq;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Verifies the JSON configuration-file layer shared by the CLI and the server:
/// <see cref="ConfigFileArgs.Expand"/> translates a <c>--config &lt;file&gt;</c>
/// into argv tokens, splices them ahead of the real command line so command-line
/// options win, and reports malformed input clearly. A few cases go end-to-end
/// through <see cref="ServerOptionsBuilder.Build"/> to confirm the merged args
/// parse and that override precedence holds through the real option parser.
/// </summary>
public class ConfigFileArgsTests : IDisposable
{
    private readonly string _dir;

    public ConfigFileArgsTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-configfile-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string WriteConfig(string json, string name = "config.json")
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, json);
        return path;
    }

    // ---- video companion keys, old and new ----------------------------------
    // Config keys ARE CLI flags (NormalizeFlag just prefixes '--'), so renaming the
    // companion flags renamed these keys too. Both spellings must expand, or every
    // config file written before the rename breaks.

    [Fact]
    public void Expand_LegacyWanCompanionKeys_StillExpandToFlags()
    {
        string cfg = WriteConfig("""
        {
          "model": "wan.gguf",
          "wan-vae": "vae.safetensors",
          "wan-te": "umt5.gguf",
          "wan-dit2": "low_noise.gguf"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Contains("--wan-vae", result);
        Assert.Contains("--wan-te", result);
        Assert.Contains("--wan-dit2", result);
    }

    [Fact]
    public void Expand_GenericVideoCompanionKeys_ExpandToFlags()
    {
        string cfg = WriteConfig("""
        {
          "model": "wan.gguf",
          "video-vae": "vae.safetensors",
          "video-text-encoder": "umt5.gguf",
          "video-dit2": "low_noise.gguf",
          "audio-vae": "audio_vae.safetensors"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Contains("--video-vae", result);
        Assert.Contains("--video-text-encoder", result);
        Assert.Contains("--video-dit2", result);
        Assert.Contains("--audio-vae", result);
    }

    [Fact]
    public void Expand_ShippedVideoConfigs_UseTheGenericKeys()
    {
        // Guard the repo's own presets: they were migrated to the generic keys, and a
        // stray legacy key here would be silently fine at runtime but inconsistent.
        string repoRoot = FindRepoRoot();
        if (repoRoot is null) return;   // running outside a source checkout

        foreach (string name in new[]
                 {
                     "wan-video-ti2v-5b.json",
                     "wan-video-ti2v-5b-turbo.json",
                     "wan-video-i2v-a14b.json",
                 })
        {
            string path = Path.Combine(repoRoot, "config", name);
            if (!File.Exists(path)) continue;
            string text = File.ReadAllText(path);
            Assert.DoesNotContain("\"wan-vae\"", text);
            Assert.DoesNotContain("\"wan-te\"", text);
            Assert.DoesNotContain("\"wan-dit2\"", text);
        }
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    [Fact]
    public void Expand_NoConfigFlag_ReturnsArgsUnchanged()
    {
        var args = new[] { "--model", "a.gguf", "--backend", "ggml_cpu" };
        var result = ConfigFileArgs.Expand(args);
        Assert.Same(args, result);
    }

    [Fact]
    public void Expand_NullOrEmpty_ReturnsEmpty()
    {
        Assert.Empty(ConfigFileArgs.Expand(null));
        Assert.Empty(ConfigFileArgs.Expand(Array.Empty<string>()));
    }

    [Fact]
    public void Expand_StringAndNumberValues_BecomeFlagValuePairs()
    {
        string cfg = WriteConfig("""
        {
          "model": "a.gguf",
          "backend": "ggml_cuda",
          "max-tokens": 4096,
          "temperature": 0.7
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        // Order within the file is preserved; numbers keep their raw text.
        Assert.Equal(
            new[] { "--model", "a.gguf", "--backend", "ggml_cuda", "--max-tokens", "4096", "--temperature", "0.7" },
            result);
    }

    [Fact]
    public void Expand_BooleanTrue_BecomesBareSwitch_FalseIsSkipped()
    {
        string cfg = WriteConfig("""
        { "continuous-batching": true, "offload-cpu": false }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--continuous-batching" }, result);
    }

    [Fact]
    public void Expand_ArrayValue_BecomesRepeatedFlag()
    {
        string cfg = WriteConfig("""
        { "stop": ["</s>", "<|eot|>"] }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--stop", "</s>", "--stop", "<|eot|>" }, result);
    }

    [Fact]
    public void Expand_KeyWithLeadingDashes_IsAccepted()
    {
        string cfg = WriteConfig("""
        { "--backend": "ggml_cpu" }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--backend", "ggml_cpu" }, result);
    }

    [Fact]
    public void Expand_ConfigTokensComeBeforeCommandLine()
    {
        string cfg = WriteConfig("""
        { "model": "from-config.gguf", "backend": "ggml_cpu" }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg, "--backend", "ggml_cuda" });

        // File tokens first, then the pass-through command-line tokens.
        Assert.Equal(
            new[] { "--model", "from-config.gguf", "--backend", "ggml_cpu", "--backend", "ggml_cuda" },
            result);
    }

    [Fact]
    public void Expand_InlineConfigEqualsForm_IsSupported()
    {
        string cfg = WriteConfig("""
        { "backend": "ggml_cpu" }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config=" + cfg });

        Assert.Equal(new[] { "--backend", "ggml_cpu" }, result);
    }

    [Fact]
    public void Expand_MultipleConfigFiles_LaterFileWins()
    {
        string a = WriteConfig("""{ "backend": "ggml_cpu", "max-tokens": 10 }""", "a.json");
        string b = WriteConfig("""{ "backend": "ggml_cuda" }""", "b.json");

        var result = ConfigFileArgs.Expand(new[] { "--config", a, "--config", b });

        // a's tokens, then b's tokens (b overrides a for backend under last-wins).
        Assert.Equal(
            new[] { "--backend", "ggml_cpu", "--max-tokens", "10", "--backend", "ggml_cuda" },
            result);
    }

    [Fact]
    public void Expand_MissingConfigFile_ThrowsFileNotFound()
    {
        var ex = Assert.Throws<FileNotFoundException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", Path.Combine(_dir, "nope.json") }));
        Assert.Contains("nope.json", ex.Message);
    }

    [Fact]
    public void Expand_InvalidJson_ThrowsArgumentException()
    {
        string cfg = WriteConfig("{ not valid json ");
        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
        Assert.Contains(cfg, ex.Message);
    }

    [Fact]
    public void Expand_NonObjectRoot_ThrowsArgumentException()
    {
        string cfg = WriteConfig("[1, 2, 3]");
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
    }

    [Fact]
    public void Expand_MissingValueForConfigFlag_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config" }));
        Assert.Contains("--config", ex.Message);
    }

    [Fact]
    public void Expand_JsonWithCommentsAndTrailingCommas_IsAccepted()
    {
        string cfg = WriteConfig("""
        {
          // the backend to host on
          "backend": "ggml_cpu",
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--backend", "ggml_cpu" }, result);
    }

    [Fact]
    public void Expand_ObjectWithoutPath_ThrowsArgumentException()
    {
        // An object value is interpreted as a downloadable-file spec; without a
        // "path" field it is not a valid spec.
        string cfg = WriteConfig("""
        { "model": { "foo": "bar" } }
        """);
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
    }

    // ----- End-to-end through the real server option parser -----

    [Fact]
    public void Build_WithConfigFile_AppliesSamplingDefaults()
    {
        string baseDir = Path.Combine(_dir, "base");
        Directory.CreateDirectory(baseDir);
        string cfg = WriteConfig("""
        { "temperature": 0.33, "top-k": 7, "stop": ["END"] }
        """);

        var options = ServerOptionsBuilder.Build(ConfigFileArgs.Expand(new[] { "--config", cfg }), baseDir);

        Assert.Equal(0.33f, options.DefaultSamplingConfig.Temperature);
        Assert.Equal(7, options.DefaultSamplingConfig.TopK);
        Assert.Equal(new[] { "END" }, options.DefaultSamplingConfig.StopSequences);
    }

    [Fact]
    public void Build_CommandLineOverridesConfigFile()
    {
        string baseDir = Path.Combine(_dir, "base2");
        Directory.CreateDirectory(baseDir);
        string cfg = WriteConfig("""
        { "temperature": 0.33 }
        """);

        // Config sets 0.33; the command line pins 0.9 and must win.
        var merged = ConfigFileArgs.Expand(new[] { "--config", cfg, "--temperature", "0.9" });
        var options = ServerOptionsBuilder.Build(merged, baseDir);

        Assert.Equal(0.9f, options.DefaultSamplingConfig.Temperature);
    }

    // ----- Variables -----

    [Fact]
    public void Expand_Variable_IsSubstitutedInStringValue()
    {
        string cfg = WriteConfig("""
        {
          "variables": { "root": "C:/models" },
          "backend": "ggml_cpu",
          "model": "${root}/a.gguf"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--backend", "ggml_cpu", "--model", "C:/models/a.gguf" }, result);
    }

    [Fact]
    public void Expand_Variable_CanReferenceAnotherVariable()
    {
        string cfg = WriteConfig("""
        {
          "variables": { "root": "C:/models", "gemma": "${root}/gemma" },
          "model": "${gemma}/model.gguf"
        }
        """);

        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });

        Assert.Equal(new[] { "--model", "C:/models/gemma/model.gguf" }, result);
    }

    [Fact]
    public void Expand_Variable_FallsBackToEnvironmentVariable()
    {
        string name = "TS_CFG_TEST_ROOT_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "D:/envmodels");
        try
        {
            string cfg = WriteConfig("{ \"model\": \"${" + name + "}/a.gguf\" }");

            var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
            Assert.Equal(new[] { "--model", "D:/envmodels/a.gguf" }, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void Expand_UndefinedVariable_ThrowsArgumentException()
    {
        string cfg = WriteConfig("""
        { "model": "${nope}/a.gguf" }
        """);
        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
        Assert.Contains("nope", ex.Message);
    }

    // ${name:-fallback}. This is what lets the shipped configs name a model root
    // without hard-coding a drive letter: a Windows path like "C:/models/x.gguf"
    // is NOT rooted on Linux/macOS, so it would be treated as relative and glued
    // onto the config's own directory rather than failing loudly.

    [Fact]
    public void Expand_VariableDefault_UsedWhenNameIsUndefined()
    {
        string cfg = WriteConfig("""
        { "model": "${TS_CFG_NO_SUCH_VAR_12345:-../models}/a.gguf" }
        """);
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--model", "../models/a.gguf" }, result);
    }

    [Fact]
    public void Expand_VariableDefault_LosesToAnEnvironmentVariable()
    {
        string name = "TS_CFG_TEST_ROOT_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "D:/envmodels");
        try
        {
            string cfg = WriteConfig("{ \"model\": \"${" + name + ":-../models}/a.gguf\" }");
            var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
            Assert.Equal(new[] { "--model", "D:/envmodels/a.gguf" }, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void Expand_VariableDefault_LosesToADeclaredVariable()
    {
        string cfg = WriteConfig("""
        { "variables": { "root": "/declared" }, "model": "${root:-../models}/a.gguf" }
        """);
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--model", "/declared/a.gguf" }, result);
    }

    [Fact]
    public void Expand_VariableDefault_UsedWhenEnvironmentVariableIsEmpty()
    {
        // `set VAR=` on Windows leaves an empty value rather than removing it;
        // treating that as "set" would resolve the root to "" and silently build
        // a path off the filesystem root.
        string name = "TS_CFG_TEST_EMPTY_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "");
        try
        {
            string cfg = WriteConfig("{ \"model\": \"${" + name + ":-../models}/a.gguf\" }");
            var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
            Assert.Equal(new[] { "--model", "../models/a.gguf" }, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void Expand_VariableDefault_MayItselfReferenceAVariable()
    {
        string cfg = WriteConfig("""
        { "variables": { "base": "/base" }, "model": "${missing:-${base}/models}/a.gguf" }
        """);
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg });
        Assert.Equal(new[] { "--model", "/base/models/a.gguf" }, result);
    }

    [Fact]
    public void Expand_CyclicVariable_ThrowsArgumentException()
    {
        string cfg = WriteConfig("""
        { "variables": { "a": "${b}", "b": "${a}" }, "model": "${a}" }
        """);
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }));
    }

    [Fact]
    public void Expand_Variable_SubstitutedInsideDownloadSpecPath()
    {
        // Pre-create the file so no download is attempted; only the path
        // substitution/resolution is under test.
        string modelDir = Path.Combine(_dir, "m");
        Directory.CreateDirectory(modelDir);
        string modelFile = Path.Combine(modelDir, "a.gguf");
        File.WriteAllText(modelFile, "stub");

        string cfg = WriteConfig($$"""
        {
          "variables": { "root": {{JsonQuote(modelDir)}} },
          "model": { "path": "${root}/a.gguf", "urls": ["http://127.0.0.1:9/never"] }
        }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal("--model", result[0]);
        Assert.Equal(Path.GetFullPath(modelFile), result[1]);
        Assert.Contains("using cached file", sink.ToString());
    }

    // ----- Auto-download -----

    [Fact]
    public void Expand_DownloadSpec_ExistingFile_UsesCachedWithoutDownloading()
    {
        string modelFile = Path.Combine(_dir, "cached.gguf");
        File.WriteAllText(modelFile, "already here");

        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(modelFile)}}, "urls": ["http://127.0.0.1:9/never"] } }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal(new[] { "--model", Path.GetFullPath(modelFile) }, result);
        Assert.Contains("using cached", sink.ToString());
    }

    [Fact]
    public void Expand_DownloadSpec_MissingFileNoUrls_ThrowsFileNotFound()
    {
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(Path.Combine(_dir, "missing.gguf"))}} } }
        """);
        Assert.Throws<FileNotFoundException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));
    }

    [Fact]
    public void Expand_DownloadSpec_DownloadsFromUrl_ThenReusesLocalCopy()
    {
        byte[] payload = System.Text.Encoding.ASCII.GetBytes("hello-model-bytes");
        using var server = new TinyHttpServer();
        server.AddFile("/model.gguf", payload);

        string dest = Path.Combine(_dir, "downloaded", "model.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(dest)}}, "urls": [ {{JsonQuote(server.UrlFor("/model.gguf"))}} ] } }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal(new[] { "--model", Path.GetFullPath(dest) }, result);
        Assert.True(File.Exists(dest));
        Assert.Equal(payload, File.ReadAllBytes(dest));
        string log = sink.ToString();
        Assert.Contains("downloading from", log);
        Assert.Contains("done", log);
        Assert.Equal(1, server.RequestCount("/model.gguf"));

        // Second run: the file exists, so no second request is made.
        var sink2 = new StringWriter();
        ConfigFileArgs.Expand(new[] { "--config", cfg }, sink2, interactiveProgress: false);
        Assert.Equal(1, server.RequestCount("/model.gguf"));
        Assert.Contains("using cached", sink2.ToString());
    }

    [Fact]
    public void Expand_DownloadSpec_FirstUrlFails_FallsBackToSecond()
    {
        byte[] payload = System.Text.Encoding.ASCII.GetBytes("mirror-two-content");
        using var server = new TinyHttpServer();
        server.AddNotFound("/primary.gguf");
        server.AddFile("/backup.gguf", payload);

        string dest = Path.Combine(_dir, "fallback.gguf");
        string cfg = WriteConfig($$"""
        {
          "model": {
            "path": {{JsonQuote(dest)}},
            "urls": [ {{JsonQuote(server.UrlFor("/primary.gguf"))}}, {{JsonQuote(server.UrlFor("/backup.gguf"))}} ]
          }
        }
        """);

        var sink = new StringWriter();
        var result = ConfigFileArgs.Expand(new[] { "--config", cfg }, sink, interactiveProgress: false);

        Assert.Equal(payload, File.ReadAllBytes(dest));
        string log = sink.ToString();
        Assert.Contains("source failed", log);
        Assert.Contains("trying next source", log);
    }

    [Fact]
    public void Expand_DownloadSpec_AllUrlsFail_ThrowsIOException()
    {
        using var server = new TinyHttpServer();
        server.AddNotFound("/a.gguf");

        string dest = Path.Combine(_dir, "never.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{JsonQuote(dest)}}, "urls": [ {{JsonQuote(server.UrlFor("/a.gguf"))}} ] } }
        """);

        Assert.Throws<IOException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));
        Assert.False(File.Exists(dest));
    }

    [Fact]
    public void Expand_DownloadSpec_Sha256Mismatch_IsTreatedAsFailure()
    {
        byte[] payload = System.Text.Encoding.ASCII.GetBytes("some-bytes");
        using var server = new TinyHttpServer();
        server.AddFile("/m.gguf", payload);

        string dest = Path.Combine(_dir, "hashed.gguf");
        string cfg = WriteConfig($$"""
        {
          "model": {
            "path": {{JsonQuote(dest)}},
            "urls": [ {{JsonQuote(server.UrlFor("/m.gguf"))}} ],
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
          }
        }
        """);

        Assert.Throws<IOException>(() =>
            ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null, interactiveProgress: false));
        // A failed transfer must not leave a file behind that a later run would trust.
        Assert.False(File.Exists(dest));
    }

    private static string JsonQuote(string s) =>
        System.Text.Json.JsonSerializer.Serialize(s);

    /// <summary>
    /// Minimal loopback HTTP server for exercising the downloader without a
    /// network dependency. Serves registered byte payloads with a Content-Length
    /// and can return 404 for URL-fallback tests.
    /// </summary>
    private sealed class TinyHttpServer : IDisposable
    {
        private readonly System.Net.HttpListener _listener = new();
        private readonly string _prefix;
        private readonly Dictionary<string, byte[]> _files = new();
        private readonly HashSet<string> _notFound = new();
        private readonly Dictionary<string, int> _hits = new();
        private readonly object _gate = new();

        public TinyHttpServer()
        {
            int port = GetFreePort();
            _prefix = $"http://127.0.0.1:{port}/";
            _listener.Prefixes.Add(_prefix);
            _listener.Start();
            System.Threading.Tasks.Task.Run(Loop);
        }

        public void AddFile(string path, byte[] content)
        {
            lock (_gate) { _files[path] = content; _hits[path] = 0; }
        }

        public void AddNotFound(string path)
        {
            lock (_gate) { _notFound.Add(path); _hits[path] = 0; }
        }

        public string UrlFor(string path) => _prefix.TrimEnd('/') + path;

        public int RequestCount(string path)
        {
            lock (_gate) { return _hits.TryGetValue(path, out int n) ? n : 0; }
        }

        private void Loop()
        {
            while (_listener.IsListening)
            {
                System.Net.HttpListenerContext ctx;
                try { ctx = _listener.GetContext(); }
                catch { return; }

                string path = ctx.Request.Url.AbsolutePath;
                byte[] body = null;
                bool notFound;
                lock (_gate)
                {
                    if (_hits.ContainsKey(path)) _hits[path]++;
                    notFound = _notFound.Contains(path);
                    if (!notFound) _files.TryGetValue(path, out body);
                }

                if (notFound || body == null)
                {
                    ctx.Response.StatusCode = 404;
                    ctx.Response.Close();
                    continue;
                }

                ctx.Response.StatusCode = 200;
                ctx.Response.ContentLength64 = body.Length;
                ctx.Response.OutputStream.Write(body, 0, body.Length);
                ctx.Response.OutputStream.Close();
            }
        }

        private static int GetFreePort()
        {
            var l = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
            l.Start();
            int port = ((System.Net.IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        public void Dispose()
        {
            try { _listener.Stop(); } catch { }
            try { _listener.Close(); } catch { }
        }
    }
}
