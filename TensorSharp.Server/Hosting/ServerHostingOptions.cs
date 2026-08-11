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
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Immutable bag of values resolved at process start-up from CLI arguments
    /// and environment variables. Registered as a DI singleton so every
    /// endpoint, adapter, and helper can pull the same view of "what is hosted
    /// on this server" without each one re-parsing argv.
    /// </summary>
    internal sealed class ServerHostingOptions
    {
        /// <summary>
        /// Address the server binds when the operator does not choose one.
        /// <c>0.0.0.0</c> (all interfaces) rather than <c>localhost</c> because
        /// the server is routinely reached from another machine or from outside
        /// a container.
        /// </summary>
        public const string DefaultListenUrls = "http://0.0.0.0:5000";

        /// <summary>Port used by <see cref="DefaultListenUrls"/>.</summary>
        public const int DefaultPort = 5000;

        public ServerHostingOptions(
            string startupModelPath,
            string startupMmProjPath,
            string defaultBackend,
            IReadOnlyList<BackendOption> supportedBackends,
            int defaultMaxTokens,
            bool maxTokensPinned,
            string uploadDirectory,
            string logDirectory,
            bool fileLoggingEnabled,
            SamplingDefaults samplingDefaults,
            string listenUrls = DefaultListenUrls)
        {
            ListenUrls = string.IsNullOrWhiteSpace(listenUrls) ? DefaultListenUrls : listenUrls;
            StartupModelPath = startupModelPath;
            StartupMmProjPath = startupMmProjPath;
            DefaultBackend = defaultBackend;
            SupportedBackends = supportedBackends ?? Array.Empty<BackendOption>();
            SupportedBackendValues = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < SupportedBackends.Count; i++)
                SupportedBackendValues.Add(SupportedBackends[i].Value);
            DefaultMaxTokens = defaultMaxTokens;
            MaxTokensPinned = maxTokensPinned;
            UploadDirectory = uploadDirectory;
            LogDirectory = logDirectory;
            FileLoggingEnabled = fileLoggingEnabled;
            SamplingDefaults = samplingDefaults ?? new SamplingDefaults(new SamplingConfig());
        }

        /// <summary>
        /// Semicolon-separated URL(s) Kestrel binds, resolved from
        /// <c>--port</c>/<c>--host</c>, <c>--urls</c>, the <c>PORT</c>/<c>HOST</c>
        /// or <c>ASPNETCORE_URLS</c> environment variables, or
        /// <see cref="DefaultListenUrls"/>. Never null or empty.
        /// </summary>
        public string ListenUrls { get; }

        /// <summary>Absolute path of the model the server was launched with, or null when no model is hosted.</summary>
        public string StartupModelPath { get; }

        /// <summary>Absolute path of the projector the server was launched with, or null when none is hosted.</summary>
        public string StartupMmProjPath { get; }

        /// <summary>Canonical name of the backend chosen at startup (e.g. <c>ggml_metal</c>).</summary>
        public string DefaultBackend { get; }

        /// <summary>Backends actually supported by this host (after probing the GGML runtime).</summary>
        internal IReadOnlyList<BackendOption> SupportedBackends { get; }

        /// <summary>Fast lookup over <see cref="SupportedBackends"/>.</summary>
        internal HashSet<string> SupportedBackendValues { get; }

        /// <summary>
        /// Default generation budget applied by every endpoint (Web UI, Ollama,
        /// OpenAI chat + responses) when the request does not carry its own
        /// limit. Resolved from <c>--max-tokens</c> / <c>MAX_TOKENS</c>, falling
        /// back to 20000.
        /// </summary>
        public int DefaultMaxTokens { get; }

        /// <summary>
        /// True when <see cref="DefaultMaxTokens"/> came from <c>--max-tokens</c>
        /// or <c>MAX_TOKENS</c> rather than the built-in fallback. A pinned value
        /// also caps requests that ask for more (see <see cref="ResolveMaxTokens"/>).
        /// </summary>
        public bool MaxTokensPinned { get; }

        /// <summary>Absolute path to the directory used for user uploads.</summary>
        public string UploadDirectory { get; }

        /// <summary>Resolved log directory (used by the file logger when it is enabled).</summary>
        public string LogDirectory { get; }

        /// <summary>True when the file logger should be wired in.</summary>
        public bool FileLoggingEnabled { get; }

        /// <summary>
        /// Default sampling parameters resolved from CLI flags / environment,
        /// together with which of them the operator pinned and whether those
        /// pins outrank a client's request. Adapters seed per-request configs
        /// from this object so unspecified fields take the operator-configured
        /// defaults instead of the hard-coded library defaults. Never null.
        /// </summary>
        public SamplingDefaults SamplingDefaults { get; }

        /// <summary>The resolved default sampling values (without the pinning metadata).</summary>
        public SamplingConfig DefaultSamplingConfig => SamplingDefaults.Values;

        /// <summary>
        /// Resolve the generation budget for one request. An absent (or
        /// non-positive, e.g. Ollama's <c>num_predict: -1</c>) request value
        /// takes the server default. A request that asks for more than a pinned
        /// <c>--max-tokens</c> is clamped to it — the flag names a *maximum*, and
        /// an operator who sized it against their KV cache should not have a
        /// client talk them out of it. A request asking for less is always
        /// honoured, so short completions stay short.
        /// </summary>
        public int ResolveMaxTokens(int? requestedMaxTokens)
        {
            if (!requestedMaxTokens.HasValue || requestedMaxTokens.Value <= 0)
                return DefaultMaxTokens;
            return MaxTokensPinned
                ? Math.Min(requestedMaxTokens.Value, DefaultMaxTokens)
                : requestedMaxTokens.Value;
        }
    }
}
