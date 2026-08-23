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
using System.Diagnostics;
using System.IO;
using Microsoft.Extensions.Logging;
using TensorSharp.Cuda;

namespace TensorSharp.Server
{
    internal sealed class ModelLifecycleService : IDisposable
    {
        private readonly ILogger _logger;
        private readonly Func<string, BackendType, ITensorParallelGroup, string, ModelBase> _createModel;

        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        public ModelLifecycleService(ILogger logger)
            : this(logger, static (path, backend, tpGroup, draftPath) =>
                ModelBase.Create(path, backend, tpGroup: tpGroup, draftModelPath: draftPath))
        {
        }

        /// <summary>Test seam: <paramref name="createModel"/> stands in for
        /// <see cref="ModelBase.Create(string, BackendType, int, ITensorParallelGroup, string)"/>.</summary>
        internal ModelLifecycleService(ILogger logger, Func<string, BackendType, ITensorParallelGroup, string, ModelBase> createModel)
        {
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
            _createModel = createModel;
        }

        public bool IsLoaded => _model != null;

        /// <summary>
        /// When the operator explicitly named an MTP draft via
        /// <c>--mtp-draft-model</c> (<c>TS_MTP_DRAFT_MODEL</c>) but it could not be
        /// activated on the loaded target (missing file, wrong architecture, an
        /// incompatible draft, or an incomplete GGUF), this holds a
        /// human-readable reason. It is <c>null</c> when no draft was requested or
        /// when the draft loaded successfully (<c>HasDraftHead</c>). The startup loader
        /// promotes a non-null value to a fail-fast error so an explicit but
        /// unusable draft can't silently leave speculation disabled — matching the
        /// fail-fast contract the rest of startup configuration follows. Runtime
        /// (Web UI) model switches read the warning log but do not fail.
        /// </summary>
        public string DraftHeadActivationError { get; private set; }

        public string LoadedModelName => _loadedModelPath != null ? Path.GetFileName(_loadedModelPath) : null;
        public string LoadedModelPath => _loadedModelPath;
        public string LoadedMmProjName => _loadedMmProjPath != null ? Path.GetFileName(_loadedMmProjPath) : null;
        public string LoadedMmProjPath => _loadedMmProjPath;
        public string LoadedBackend => _model != null ? BackendCatalog.ToBackendValue(_backend) : null;
        public string Architecture => _model?.Config?.Architecture;
        public ModelBase Model => _model;
        public BackendType Backend => _backend;

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _model != null && string.Equals(LoadedModelName, modelName, StringComparison.OrdinalIgnoreCase);
        }

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _logger.LogInformation(LogEventIds.ModelLoadStarted,
                "Loading model {ModelFile} (mmproj={MmProjFile}, backend={Backend}, fullPath={ModelPath}, mmprojPath={MmProjPath})",
                Path.GetFileName(modelPath), Path.GetFileName(mmProjPath ?? string.Empty),
                backendStr ?? "(default)", modelPath, mmProjPath ?? "(none)");

            try
            {
                ValidateModelFiles(modelPath, mmProjPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Rejected model load {ModelFile}: file validation failed; current model {CurrentModel} stays loaded",
                    Path.GetFileName(modelPath), LoadedModelName ?? "(none)");
                throw;
            }

            string previousModelPath = _loadedModelPath;
            string previousMmProjPath = _loadedMmProjPath;
            string previousBackendValue = LoadedBackend;

            UnloadCurrentModel();

            try
            {
                LoadModelCore(modelPath, mmProjPath, backendStr);
            }
            catch
            {
                // Best-effort rollback so a failed reload doesn't leave the
                // server with no model. The original exception still reaches
                // the caller; the rollback outcome is only logged.
                if (previousModelPath != null)
                {
                    try
                    {
                        LoadModelCore(previousModelPath, previousMmProjPath, previousBackendValue);
                        _logger.LogWarning("Restored previous model {PreviousModel} after failed load of {ModelFile}",
                            Path.GetFileName(previousModelPath), Path.GetFileName(modelPath));
                    }
                    catch (Exception rollbackEx)
                    {
                        _logger.LogError(LogEventIds.ModelLoadFailed, rollbackEx,
                            "Could not restore previous model {PreviousModel} after failed load of {ModelFile}; no model is loaded",
                            Path.GetFileName(previousModelPath), Path.GetFileName(modelPath));
                    }
                }
                throw;
            }
        }

        /// <summary>
        /// Header-only validation of the files a load is about to commit to,
        /// run BEFORE the current model is disposed: a missing, non-GGUF, or
        /// truncated file is rejected while there is still a working model.
        /// </summary>
        private static void ValidateModelFiles(string modelPath, string mmProjPath)
        {
            using (var gguf = new GgufFile(modelPath))
                gguf.ThrowIfTruncated();

            // A missing projector file is skipped by the load itself, but one
            // that exists must parse.
            if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
            {
                using var mmProj = new GgufFile(mmProjPath);
                mmProj.ThrowIfTruncated();
            }
        }

        private void UnloadCurrentModel()
        {
            string previousModel = LoadedModelName;
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            DraftHeadActivationError = null;

            if (!string.IsNullOrEmpty(previousModel))
            {
                _logger.LogInformation(LogEventIds.ModelUnloaded,
                    "Unloaded previous model {PreviousModel}", previousModel);
            }
        }

        private void LoadModelCore(string modelPath, string mmProjPath, string backendStr)
        {
            _backend = ResolveBackend(backendStr);

            var loadSw = Stopwatch.StartNew();
            try
            {
                // Check for distributed TP configuration via environment variables.
                ITensorParallelGroup tpGroup = null;
                var distConfig = TensorSharp.Distributed.DistributedTpConfig.TryFromEnvironment(
                    localDegree: GetLocalTpDegree());
                if (distConfig != null)
                {
                    // The on-node group has to match the backend: direct CUDA
                    // drives CudaAllocators, the ggml backends drive per-rank
                    // ggml backends.
                    tpGroup = _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan
                        ? new TensorSharp.Distributed.DistributedTensorParallelGroup(
                            ModelBase.CreateGgmlLocalTpGroup(_backend, distConfig.LocalDegree),
                            distConfig.NodeId, distConfig.PeerEndpoints)
                        : new TensorSharp.Distributed.DistributedTensorParallelGroup(
                            distConfig.LocalDegree, distConfig.NodeId, distConfig.PeerEndpoints);
                }

                // DeepSeek V4's DSpark drafter ships as a SEPARATE GGUF named by
                // --draft-model (TS_DSV4_DSPARK). It goes to the factory rather
                // than being attached afterwards like Gemma 4's draft head: the
                // drafter's weights have to be counted by the layer split and
                // uploaded with the trunk.
                string blockDraftPath = Environment.GetEnvironmentVariable("TS_DSV4_DSPARK");
                _model = _createModel(modelPath, _backend, tpGroup, blockDraftPath);

                // Say so when a drafter was named but the loaded model has no
                // block-draft head to put it in, instead of leaving the operator
                // to wonder why speculation never engages.
                if (!string.IsNullOrEmpty(blockDraftPath)
                    && _model is not TensorSharp.Runtime.Speculative.IDraftHead { HasDraftHead: true })
                {
                    _logger.LogWarning(
                        "--draft-model '{Draft}' was given but the loaded model ({Architecture} on {Backend}) has no " +
                        "block drafter; serving standard decode. DSpark is implemented for DeepSeek V4 on the " +
                        "cuda and ggml_cuda backends.",
                        Path.GetFileName(blockDraftPath), Architecture ?? "unknown", _backend);
                }

                // A worker node (--tp-node-id > 0) spends its life blocked in a
                // mirror loop and cannot also serve HTTP requests, so the server
                // only takes the driver role (node 0). Fail fast with the
                // supported topology instead of hanging on the first inference.
                if (_model.IsDistributedWorker)
                {
                    throw new InvalidOperationException(
                        "TensorSharp.Server cannot run as a distributed tensor-parallel WORKER node " +
                        "(--tp-node-id > 0 / TENSORSHARP_TP_NODE_ID > 0). Run this server as node 0 (the driver) " +
                        "and start each worker node with TensorSharp.Cli, e.g.: " +
                        "TensorSharp.Cli --model <same.gguf> --backend <same> --tp <localGpus> --tp-node-id <N> --tp-peers <same list>.");
                }

                _loadedModelPath = modelPath;

                if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
                {
                    LoadEncoders(mmProjPath);
                    _loadedMmProjPath = mmProjPath;
                }

                // Speculator weights that ship as their own file (Gemma 4's
                // gemma4-assistant draft head) are attached here, by the same
                // shared loader the CLI uses. DraftHeadActivationError was
                // cleared when the previous model was unloaded.
                if (!SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(_model, out string draftError))
                {
                    DraftHeadActivationError = draftError;
                    _logger.LogWarning("{Error}; speculation disabled.", draftError);
                }
                else if (SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath() is { } attachedDraft)
                {
                    _logger.LogInformation("Loaded draft head {Draft} (HasDraftHead=True)",
                        Path.GetFileName(attachedDraft));
                }

                loadSw.Stop();
                long modelBytes = SafeGetFileSize(modelPath);
                long mmProjBytes = SafeGetFileSize(mmProjPath);
                _logger.LogInformation(LogEventIds.ModelLoadCompleted,
                    "Loaded model {Model} (architecture={Architecture}, backend={Backend}, modelBytes={ModelBytes}, mmproj={MmProjFile}, mmprojBytes={MmProjBytes}) in {ElapsedMs:F1} ms",
                    LoadedModelName, Architecture ?? "(unknown)", LoadedBackend ?? "(unknown)",
                    modelBytes, LoadedMmProjName ?? "(none)", mmProjBytes, loadSw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                loadSw.Stop();
                _logger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Failed to load model {ModelFile} on backend {Backend} after {ElapsedMs:F1} ms",
                    Path.GetFileName(modelPath), backendStr ?? "(default)", loadSw.Elapsed.TotalMilliseconds);
                // Drop any partially initialized model so the service holds
                // either a fully loaded model or none at all.
                _model?.Dispose();
                _model = null;
                _loadedModelPath = null;
                _loadedMmProjPath = null;
                DraftHeadActivationError = null;
                throw;
            }
        }

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
        }

        private void LoadEncoders(string mmProjPath)
        {
            _model?.MultimodalInjector.LoadProjectors(mmProjPath);
        }

        private static BackendType ResolveBackend(string backendStr)
        {
            return BackendCatalog.Canonicalize(backendStr) switch
            {
                "mlx" => BackendType.Mlx,
                "cuda" => BackendType.Cuda,
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_cuda" => BackendType.GgmlCuda,
                "ggml_vulkan" => BackendType.GgmlVulkan,
                "cpu" => BackendType.Cpu,
                _ => BackendType.GgmlCpu
            };
        }

        private static long SafeGetFileSize(string path)
        {
            if (string.IsNullOrEmpty(path))
                return 0;
            try
            {
                var fi = new FileInfo(path);
                return fi.Exists ? fi.Length : 0;
            }
            catch
            {
                return 0;
            }
        }

        private static int GetLocalTpDegree()
        {
            string envTp = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE");
            if (int.TryParse(envTp, out int degree) && degree > 1)
                return degree;
            return 1;
        }
    }
}
