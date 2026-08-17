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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Logging;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.StreamingWriters;

namespace TensorSharp.Server.ProtocolAdapters
{
    /// <summary>
    /// Implements the request handlers used by the bundled Web UI:
    /// <list type="bullet">
    ///   <item>queue status (<c>GET /api/queue/status</c>)</item>
    ///   <item>session lifecycle (<c>POST /api/sessions</c>, <c>DELETE /api/sessions/{id}</c>)</item>
    ///   <item>model state + reload (<c>GET /api/models</c>, <c>POST /api/models/load</c>)</item>
    ///   <item>file upload (<c>POST /api/upload</c>)</item>
    ///   <item>SSE chat stream (<c>POST /api/chat</c>)</item>
    /// </list>
    ///
    /// The adapter owns NO state of its own; everything is injected (model
    /// service, queue, session manager, configuration, loggers). That means a
    /// single instance can be reused across requests and easily faked in tests.
    /// </summary>
    internal sealed class WebUiAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly SessionManager _sessions;
        private readonly ServerHostingOptions _options;
        private readonly ILoggerFactory _loggerFactory;

        public WebUiAdapter(
            ModelService svc,
            InferenceQueue queue,
            SessionManager sessions,
            ServerHostingOptions options,
            ILoggerFactory loggerFactory)
        {
            _svc = svc ?? throw new ArgumentNullException(nameof(svc));
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _sessions = sessions ?? throw new ArgumentNullException(nameof(sessions));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        }

        // ---- Queue ------------------------------------------------------------

        public IResult GetQueueStatus()
        {
            // Real concurrency now lives in the per-model inference engine, not the
            // legacy InferenceQueue (which always reports zero). Peek the engine's
            // live counters so the Web UI can show, in real time, how many requests
            // are being processed concurrently versus waiting for admission.
            // TryGetLiveStats is side-effect free and returns false before the
            // engine is built (no model loaded / no request yet), in which case
            // everything is idle.
            _svc.EngineHost.TryGetLiveStats(out int processing, out int waiting, out long totalCompleted);

            // total_processed kept for API compatibility; sourced from the engine's
            // completed count (per loaded model) rather than the legacy queue.
            long totalProcessed = totalCompleted != 0 ? totalCompleted : _queue.GetStatus().TotalProcessed;

            return Results.Ok(new
            {
                busy = processing > 0,
                // Number of requests currently being generated concurrently.
                processing,
                // Requests admitted to the engine but still waiting for a batch slot.
                pending_requests = waiting,
                total_processed = totalProcessed,
            });
        }

        // ---- Sessions ---------------------------------------------------------

        public IResult CreateSession()
        {
            var sessionsLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Sessions");
            var session = _sessions.CreateSession();
            sessionsLogger.LogInformation(LogEventIds.SessionCreated,
                "Created session via /api/sessions: {SessionId}", session.Id);
            return Results.Json(new
            {
                sessionId = session.Id,
                createdAt = session.CreatedAt.ToString("o"),
            });
        }

        public async Task<IResult> DisposeSessionAsync(string id, HttpContext ctx)
        {
            var sessionsLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Sessions");
            if (string.Equals(id, SessionManager.DefaultSessionId, StringComparison.Ordinal))
            {
                sessionsLogger.LogWarning(LogEventIds.SessionRemoved,
                    "Refused to dispose default session via API: {SessionId}", id);
                return Results.BadRequest(new { ok = false, error = "Cannot dispose the default session." });
            }

            var removed = _sessions.TryRemove(id);
            if (removed == null)
            {
                sessionsLogger.LogWarning(LogEventIds.SessionRemoved,
                    "Session not found for disposal: {SessionId}", id);
                return Results.NotFound(new { ok = false, error = $"Session '{id}' not found." });
            }

            // Keep the legacy queue handshake for the API contract. The queue is a
            // no-op now; in-flight KV state is owned by the engine, while disposing
            // the session only clears tracked chat history.
            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            await ticket.WaitUntilReadyAsync();
            _svc.DisposeSession(removed);
            sessionsLogger.LogInformation(LogEventIds.SessionDisposed,
                "Disposed session via /api/sessions: {SessionId}", id);
            return Results.Json(new { ok = true, sessionId = id });
        }

        // ---- Models ----------------------------------------------------------

        public IResult GetModels()
        {
            var files = string.IsNullOrWhiteSpace(_options.StartupModelPath)
                ? new List<string>()
                : new List<string> { Path.GetFileName(_options.StartupModelPath) };
            var mmProjFiles = string.IsNullOrWhiteSpace(_options.StartupMmProjPath)
                ? new List<string>()
                : new List<string> { Path.GetFileName(_options.StartupMmProjPath) };
            return Results.Json(new
            {
                models = files,
                mmProjModels = mmProjFiles,
                loaded = _svc.LoadedModelName,
                loadedMmProj = _svc.LoadedMmProjName,
                loadedBackend = _svc.LoadedBackend,
                defaultBackend = _options.DefaultBackend,
                supportedBackends = _options.SupportedBackends,
                architecture = _svc.Architecture,
                defaultMaxTokens = _options.DefaultMaxTokens,
            });
        }

        public async Task<IResult> LoadModelAsync(HttpContext ctx, HttpRequest req)
        {
            var modelLoadLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.ModelLoad");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(req.Body);
            string modelName = body.GetProperty("model").GetString();
            string requestedBackend = body.TryGetProperty("backend", out var b) ? b.GetString() : null;
            string mmproj = body.TryGetProperty("mmproj", out var m) ? m.GetString() : null;

            modelLoadLogger.LogInformation(LogEventIds.ModelLoadStarted,
                "Web UI model load request: model={Model} backend={Backend} mmproj={MmProj}",
                modelName, requestedBackend ?? "(default)", mmproj ?? "(none)");

            if (!BackendSelector.TryResolveSupportedBackend(_options, requestedBackend, out string backend, out string backendError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI model load rejected: {Reason}", backendError);
                return Results.BadRequest(new { ok = false, error = backendError });
            }

            if (!HostedModelGuard.TryResolveHostedModelRequest(modelName, _options.StartupModelPath, out string modelPath, out string modelError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI model load rejected: {Reason}", modelError);
                return Results.BadRequest(new { ok = false, error = modelError });
            }

            if (!HostedModelGuard.TryValidateHostedMmProjRequest(mmproj, _options.StartupMmProjPath, out string mmProjError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI mmproj validation failed: {Reason}", mmProjError);
                return Results.BadRequest(new { ok = false, error = mmProjError });
            }

            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            await ticket.WaitUntilReadyAsync();

            try
            {
                _svc.LoadModel(modelPath, _options.StartupMmProjPath, backend);
                return Results.Json(new
                {
                    ok = true,
                    model = _svc.LoadedModelName,
                    loadedMmProj = _svc.LoadedMmProjName,
                    architecture = _svc.Architecture,
                });
            }
            catch (Exception ex)
            {
                modelLoadLogger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Web UI model load failed: model={Model} backend={Backend}", modelName, backend);
                return Results.Json(new { ok = false, error = ex.Message }, statusCode: 500);
            }
        }

        // ---- Upload ----------------------------------------------------------

        public async Task<IResult> UploadAsync(HttpRequest req)
        {
            var uploadLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Upload");
            if (!req.HasFormContentType)
            {
                uploadLogger.LogWarning(LogEventIds.UploadRejected,
                    "Upload rejected: missing multipart form data");
                return Results.BadRequest(new { error = "Expected multipart form data" });
            }

            var form = await req.ReadFormAsync();
            var file = form.Files.FirstOrDefault();
            if (file == null)
            {
                uploadLogger.LogWarning(LogEventIds.UploadRejected,
                    "Upload rejected: no file in request");
                return Results.BadRequest(new { error = "No file uploaded" });
            }

            string ext = Path.GetExtension(file.FileName).ToLowerInvariant();
            // Classify before anything touches disk: an upload with an extension
            // outside the allow-list is rejected without ever being written, so
            // /uploads can only ever hold files the serve-side policy covers.
            string mediaType = UploadContentPolicy.Classify(ext);
            if (mediaType == "unknown")
            {
                uploadLogger.LogWarning(LogEventIds.UploadRejected,
                    "Upload rejected: unsupported extension {Extension} (name={FileName})",
                    ext.Length == 0 ? "(none)" : ext, file.FileName);
                return Results.BadRequest(new
                {
                    error = ext.Length == 0
                        ? "Files without an extension are not supported. Upload an image, video, audio, PDF, or plain-text/code file."
                        : $"Unsupported file type '{ext}'. Upload an image, video, audio, PDF, or plain-text/code file.",
                });
            }

            string safeFileName = $"{Guid.NewGuid():N}{ext}";
            string savePath = Path.Combine(_options.UploadDirectory, safeFileName);
            string uploadUrl = BuildUploadUrl(safeFileName);

            using (var stream = File.Create(savePath))
                await file.CopyToAsync(stream);

            // Include the full saved path and the classified media type so this entry
            // is self-sufficient for tracing back from the per-turn chat log
            // (which records each attachment by its saved path).
            uploadLogger.LogInformation(LogEventIds.UploadReceived,
                "Upload received: name={FileName} ext={Extension} mediaType={MediaType} bytes={Length} savedAs={SavedFile} savedPath={SavedPath}",
                file.FileName, ext, mediaType, file.Length, safeFileName, savePath);

            if (mediaType == "video")
            {
                var frames = MediaHelper.ExtractVideoFrames(savePath);
                return Results.Json(new
                {
                    ok = true,
                    path = savePath,
                    url = uploadUrl,
                    mediaType,
                    fileName = file.FileName,
                    frames = frames.Select(f => Path.GetFileName(f)).ToList(),
                    frameUrls = frames.Select(f => BuildUploadUrl(Path.GetFileName(f))).ToList(),
                    framePaths = frames,
                });
            }

            if (mediaType == "text")
            {
                string textContent = TextUploadHelper.PreserveFullText(
                    await File.ReadAllTextAsync(savePath));

                return Results.Json(new
                {
                    ok = true,
                    path = savePath,
                    url = uploadUrl,
                    mediaType,
                    fileName = file.FileName,
                    textContent,
                    truncated = false,
                    truncateLimit = (int?)null,
                    truncateUnit = (string)null,
                    modelContextLimit = _svc.Model?.MaxContextLength,
                    originalTokenCount = (int?)null,
                    returnedTokenCount = (int?)null,
                });
            }

            if (mediaType == "pdf")
            {
                // PDFs are a document modality. First try the cheap text path: extract the
                // text layer and hand it back with the same contract as a plain-text upload,
                // so the Web UI inlines it into the message and the normal prefill path runs
                // it. A scanned / image-only PDF has no text layer, so we fall back to
                // recovering its page images and letting a vision model read them (mirroring
                // the video -> frames path) — or, if no vision model is loaded, we tell the
                // user exactly why the document can't be read instead of silently dropping it.
                PdfTextResult pdf;
                try
                {
                    pdf = await Task.Run(() => PdfTextExtractor.ExtractFromFile(savePath, ResolvePdfMaxPages()));
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadRejected,
                        "PDF text extraction failed: name={FileName} savedPath={SavedPath} error={Error}",
                        file.FileName, savePath, ex.Message);
                    return Results.BadRequest(new { ok = false, error = "Could not read the PDF: " + ex.Message });
                }

                if (!pdf.LooksTextless)
                {
                    string textContent = TextUploadHelper.PreserveFullText(pdf.Text);
                    bool allPagesExtracted = pdf.ExtractedPageCount == pdf.PageCount;

                    if (allPagesExtracted)
                    {
                        uploadLogger.LogInformation(LogEventIds.UploadReceived,
                            "PDF text extracted without upload truncation: name={FileName} pages={Pages} extractedPages={ExtractedPages} chars={Chars}",
                            file.FileName, pdf.PageCount, pdf.ExtractedPageCount, textContent.Length);
                    }
                    else
                    {
                        uploadLogger.LogWarning(LogEventIds.UploadReceived,
                            "PDF text extraction did not read every page: name={FileName} pages={Pages} extractedPages={ExtractedPages} chars={Chars}",
                            file.FileName, pdf.PageCount, pdf.ExtractedPageCount, textContent.Length);
                    }

                    return Results.Json(new
                    {
                        ok = true,
                        path = savePath,
                        url = uploadUrl,
                        mediaType,
                        fileName = file.FileName,
                        renderedAsImages = false,
                        pageCount = pdf.PageCount,
                        extractedPageCount = pdf.ExtractedPageCount,
                        textContent,
                        truncated = false,
                        complete = allPagesExtracted,
                        warning = allPagesExtracted
                            ? null
                            : $"Only {pdf.ExtractedPageCount} of {pdf.PageCount} PDF pages could be read. The extracted pages were not token-truncated.",
                        truncateLimit = (int?)null,
                        truncateUnit = (string)null,
                        modelContextLimit = _svc.Model?.MaxContextLength,
                        originalTokenCount = (int?)null,
                        returnedTokenCount = (int?)null,
                    });
                }

                // Scanned / image-only PDF (no selectable text layer).
                bool visionLoaded = _svc.LoadedMmProjName != null || (_svc.Model?.HasVisionEncoder() ?? false);
                if (!visionLoaded)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "PDF has no text layer and no vision model is loaded: name={FileName} pages={Pages}",
                        file.FileName, pdf.PageCount);
                    return Results.Json(new
                    {
                        ok = true,
                        path = savePath,
                        url = uploadUrl,
                        mediaType,
                        fileName = file.FileName,
                        renderedAsImages = false,
                        needsVision = true,
                        pageCount = pdf.PageCount,
                        textContent = "",
                        warning = $"\"{file.FileName}\" has no selectable text — it looks scanned or image-only. " +
                                  "To analyze it, run the server with a vision-capable model and its projector (--mmproj <projector.gguf>).",
                    });
                }

                PdfImageResult pdfImages;
                try
                {
                    pdfImages = await Task.Run(() => PdfPageImageExtractor.ExtractPageImages(
                        savePath, _options.UploadDirectory, ResolvePdfMaxPages(),
                        Path.GetFileNameWithoutExtension(safeFileName)));
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadRejected,
                        "PDF page-image extraction failed: name={FileName} savedPath={SavedPath} error={Error}",
                        file.FileName, savePath, ex.Message);
                    return Results.BadRequest(new { ok = false, error = "Could not read the PDF: " + ex.Message });
                }

                if (pdfImages.ImagePaths.Count == 0)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "PDF yielded neither text nor images: name={FileName} pages={Pages}", file.FileName, pdf.PageCount);
                    return Results.Json(new
                    {
                        ok = true,
                        path = savePath,
                        url = uploadUrl,
                        mediaType,
                        fileName = file.FileName,
                        renderedAsImages = false,
                        pageCount = pdf.PageCount,
                        textContent = "",
                        warning = $"Could not extract any text or images from \"{file.FileName}\".",
                    });
                }

                var framePaths = pdfImages.ImagePaths.ToList();
                var frameNames = framePaths.Select(Path.GetFileName).ToList();
                var frameUrls = frameNames.Select(BuildUploadUrl).ToList();
                bool allPagesRendered = pdfImages.ExtractedPageCount == pdfImages.PageCount;
                string incompleteWarning = BuildIncompletePdfImageWarning(
                    pdfImages.ExtractedPageCount, pdfImages.PageCount);

                if (allPagesRendered)
                {
                    uploadLogger.LogInformation(LogEventIds.UploadReceived,
                        "PDF rendered as page images: name={FileName} pages={Pages} images={Images} complete=true",
                        file.FileName, pdf.PageCount, framePaths.Count);
                }
                else
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "PDF page-image extraction was incomplete: name={FileName} pages={Pages} images={Images}",
                        file.FileName, pdf.PageCount, framePaths.Count);
                }

                return Results.Json(new
                {
                    ok = true,
                    path = savePath,
                    url = uploadUrl,
                    mediaType,
                    fileName = file.FileName,
                    renderedAsImages = true,
                    pageCount = pdf.PageCount,
                    extractedPageCount = pdfImages.ExtractedPageCount,
                    complete = allPagesRendered,
                    warning = incompleteWarning,
                    frames = frameNames,
                    frameUrls,
                    framePaths,
                    note = $"This PDF has no selectable text; {framePaths.Count} page image(s) were attached for the vision model to read.",
                });
            }

            // HEIC/HEIF images (e.g. iPhone photos): the server-side pipelines decode them
            // fine (Magick.NET), but no mainstream browser renders them in <img> — and the
            // default static-file content-type provider doesn't even serve the extension —
            // so the chat bubble showed a blank/broken preview. Convert a lightweight PNG
            // preview at upload time; the Web UI displays previewUrl while path (the
            // original file, full fidelity) is what the edit/vision pipelines consume.
            if (mediaType == "image" && ext is ".heic" or ".heif")
            {
                try
                {
                    string previewName = Path.GetFileNameWithoutExtension(safeFileName) + "-preview.png";
                    string previewPath = Path.Combine(_options.UploadDirectory, previewName);
                    await Task.Run(() =>
                    {
                        var img = TensorSharp.Models.QwenImage.ImageIO.Load(savePath);
                        const long previewArea = 768L * 768;   // plenty for the ~300 px bubble preview
                        if ((long)img.Width * img.Height > previewArea)
                            img = TensorSharp.Models.QwenImage.ImageIO.ResizeToArea(img, previewArea, multiple: 1);
                        TensorSharp.Models.QwenImage.ImageIO.SavePng(previewPath, img);
                    });
                    return Results.Json(new
                    {
                        ok = true,
                        path = savePath,
                        url = uploadUrl,
                        previewUrl = BuildUploadUrl(previewName),
                        mediaType,
                        fileName = file.FileName,
                    });
                }
                catch (Exception ex)
                {
                    uploadLogger.LogWarning(LogEventIds.UploadReceived,
                        "HEIC preview conversion failed for {FileName}: {Error} (chat preview will be blank; the edit itself is unaffected)",
                        file.FileName, ex.Message);
                }
            }

            return Results.Json(new { ok = true, path = savePath, url = uploadUrl, mediaType, fileName = file.FileName });
        }

        // ---- Image editing (Qwen-Image-Edit) ---------------------------------

        private static readonly object _imageEditLock = new();

        /// <summary>
        /// <c>POST /api/image-edit</c> — multipart form with one or more <c>image</c> files and a
        /// <c>prompt</c> (plus optional <c>steps</c>, <c>cfg</c>, <c>seed</c>). Runs the loaded
        /// Qwen-Image-Edit model and returns a downloadable URL to the generated PNG. With
        /// multiple images the first drives the output geometry and the prompt can reference
        /// them as "Picture 1", "Picture 2", ... in upload order.
        /// </summary>
        public async Task<IResult> ImageEditAsync(HttpRequest req)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            if (_svc.Model is not TensorSharp.Models.QwenImage.QwenImageModel editModel)
                return Results.BadRequest(new { error = "The loaded model is not a Qwen-Image-Edit model." });

            string prompt; int steps; float cfg; long seed; long targetArea = 0;
            var imageBytesList = new List<byte[]>();

            if (req.HasFormContentType)
            {
                // Multipart: image file(s) + fields (direct API use). All parts named 'image'
                // (or every file part when none is) are taken in order.
                var form = await req.ReadFormAsync();
                var files = form.Files.GetFiles("image");
                var fileList = files.Count > 0 ? files : form.Files;
                if (fileList.Count == 0)
                    return Results.BadRequest(new { error = "No image uploaded (field 'image')." });
                prompt = form["prompt"].ToString();
                steps = int.TryParse(form["steps"], out int s) ? s : 0;   // 0 = auto (30, or the Lightning LoRA's step count)
                cfg = float.TryParse(form["cfg"], out float c) ? c : 0f;  // 0 = auto (2.5, or 1.0 with a Lightning LoRA)
                seed = long.TryParse(form["seed"], out long sd) ? sd : 0;
                if (long.TryParse(form["targetArea"], out long taf) && taf > 0) targetArea = taf;
                foreach (var file in fileList)
                {
                    using var ms = new MemoryStream();
                    await file.CopyToAsync(ms);
                    imageBytesList.Add(ms.ToArray());
                }
            }
            else
            {
                // JSON: { imagePaths[] or imagePath (server paths from /api/upload), prompt, steps, cfg, seed } (Web UI).
                var body = await System.Text.Json.JsonDocument.ParseAsync(req.Body);
                var root = body.RootElement;
                prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                steps = root.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si) ? si : 0;   // 0 = auto
                cfg = root.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv) ? cv : 0f;  // 0 = auto
                seed = root.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) ? sv : 0;
                if (root.TryGetProperty("targetArea", out var ta) && ta.TryGetInt64(out long tav) && tav > 0)
                    targetArea = tav;
                string error = await ReadUploadedImagesAsync(root, imageBytesList, CancellationToken.None);
                if (error != null)
                    return Results.BadRequest(new { error });
            }

            string outName = $"edit-{Guid.NewGuid():N}.png";
            string outPath = Path.Combine(_options.UploadDirectory, outName);

            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit: prompt='{Prompt}' steps={Steps} cfg={Cfg} images={Count} bytes={Bytes}",
                prompt, steps, cfg, imageBytesList.Count, imageBytesList.Sum(b => (long)b.Length));

            var sw = System.Diagnostics.Stopwatch.StartNew();
            (int w, int h) = await Task.Run(() =>
            {
                // The model is not thread-safe; serialize edit requests.
                lock (_imageEditLock)
                {
                    var inputs = imageBytesList.ConvertAll(TensorSharp.Models.QwenImage.ImageIO.Decode);
                    var p = new TensorSharp.Models.QwenImage.QwenImageParams { Steps = steps, CfgScale = cfg, Seed = seed };
                    if (targetArea > 0) p.TargetArea = targetArea;
                    var output = editModel.EditImage(prompt, inputs, p);
                    TensorSharp.Models.QwenImage.ImageIO.SavePng(outPath, output);
                    return (output.Width, output.Height);
                }
            });
            sw.Stop();

            string url = BuildUploadUrl(outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit done: {W}x{H} -> {Url} ({Sec:F1}s)", w, h, url, sw.Elapsed.TotalSeconds);
            return Results.Json(new { ok = true, url, width = w, height = h, elapsedSeconds = sw.Elapsed.TotalSeconds });
        }

        /// <summary>
        /// Read the referenced upload(s) from a JSON edit request into <paramref name="images"/>:
        /// <c>imagePaths</c> (array, multi-image) or legacy <c>imagePath</c> (single). Every path
        /// must resolve inside the upload directory. Returns an error message, or null on success.
        /// </summary>
        internal async Task<string> ReadUploadedImagesAsync(JsonElement root, List<byte[]> images, CancellationToken ct)
        {
            var paths = new List<string>();
            if (root.TryGetProperty("imagePaths", out var ips) && ips.ValueKind == JsonValueKind.Array)
                foreach (var el in ips.EnumerateArray())
                    if (el.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(el.GetString()))
                        paths.Add(el.GetString());
            if (paths.Count == 0 && root.TryGetProperty("imagePath", out var ip) && ip.ValueKind == JsonValueKind.String)
                paths.Add(ip.GetString());
            if (paths.Count == 0)
                return "imagePath (or imagePaths) must reference a previously uploaded file.";

            string uploadRoot = Path.GetFullPath(_options.UploadDirectory);
            foreach (var path in paths)
            {
                string full = path == null ? null : Path.GetFullPath(path);
                if (full == null || !UploadPathGuard.IsInsideDirectory(uploadRoot, full) || !File.Exists(full))
                    return "imagePath must reference a previously uploaded file.";
                images.Add(await File.ReadAllBytesAsync(full, ct));
            }
            return null;
        }

        // ---- Text-to-video (Wan) ---------------------------------------------

        private static readonly object _videoGenLock = new();

        private sealed class VideoFrame
        {
            public int Step, Total;
            public bool Final;
            public string Url;
            public int Width, Height, Frames, Fps;
            public long Seed;
            public string Codec;
            public double Seconds;
            public string Error;
        }

        private TensorSharp.Models.WanVideo.WanVideoParams ParseVideoParams(JsonElement root, out string error)
        {
            return WanVideoParamsParser.Parse(root, _options, out error);
        }

        /// <summary>
        /// <c>POST /api/video-generate</c> — JSON <c>{ prompt, width?, height?, frames?, steps?,
        /// cfg?, seed?, fps?, flowShift?, negativePrompt? }</c>. Runs the loaded Wan text-to-video
        /// model and returns a downloadable URL to the generated MP4.
        /// </summary>
        public async Task<IResult> VideoGenerateAsync(HttpRequest req)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.VideoGenerate");
            if (_svc.Model is not TensorSharp.Models.WanVideo.WanVideoModel videoModel)
                return Results.BadRequest(new { error = "The loaded model is not a Wan video-generation model." });

            var body = await JsonDocument.ParseAsync(req.Body);
            var root = body.RootElement;
            string prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
            if (string.IsNullOrWhiteSpace(prompt))
                return Results.BadRequest(new { error = "prompt is required." });
            var p = ParseVideoParams(root, out string imgError);
            if (imgError != null)
                return Results.BadRequest(new { error = imgError });

            string outName = $"video-{Guid.NewGuid():N}.mp4";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Video generate: prompt='{Prompt}' {W}x{H}x{F} steps={Steps} i2v={I2V}",
                prompt, p.Width, p.Height, p.Frames, p.Steps, p.ImageBytes != null);

            var sw = Stopwatch.StartNew();
            var result = await Task.Run(() =>
            {
                lock (_videoGenLock)
                {
                    var video = videoModel.GenerateVideo(prompt, p);
                    string codec = TensorSharp.Models.WanVideo.VideoIO.SaveMp4(outPath, video.Frames, video.Fps);
                    return (video, codec);
                }
            });
            sw.Stop();

            string url = BuildUploadUrl(outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Video generate done: {F} frames -> {Url} ({Sec:F1}s)", result.video.Frames.Length, url, sw.Elapsed.TotalSeconds);
            return Results.Json(new
            {
                ok = true, url,
                width = result.video.Frames[0].Width, height = result.video.Frames[0].Height,
                frames = result.video.Frames.Length, fps = result.video.Fps,
                seed = result.video.Seed, codec = result.codec,
                elapsedSeconds = sw.Elapsed.TotalSeconds,
            });
        }

        /// <summary>
        /// <c>POST /v1/videos/generations</c> — OpenAI-images-style envelope for Wan
        /// text-to-video: <c>{ prompt, size?: "832x480", frames?, steps?, cfg?, seed?,
        /// fps?, negative_prompt?, response_format?: "url"|"b64_json" }</c> returns
        /// <c>{ created, data: [{ url, b64_json? }], ... }</c>.
        /// </summary>
        public async Task<IResult> OpenAIVideoGenerationsAsync(HttpRequest req)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.VideoGenerate");
            if (_svc.Model is not TensorSharp.Models.WanVideo.WanVideoModel videoModel)
                return Results.BadRequest(new { error = new { message = "The loaded model is not a Wan video-generation model.", type = "invalid_request_error" } });

            var body = await JsonDocument.ParseAsync(req.Body);
            var root = body.RootElement;
            string prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
            if (string.IsNullOrWhiteSpace(prompt))
                return Results.BadRequest(new { error = new { message = "prompt is required.", type = "invalid_request_error" } });

            var p = ParseVideoParams(root, out string imgError);
            if (imgError != null)
                return Results.BadRequest(new { error = new { message = imgError, type = "invalid_request_error" } });
            if (root.TryGetProperty("size", out var sz) && sz.ValueKind == JsonValueKind.String)
            {
                var parts = (sz.GetString() ?? "").Split('x');
                if (parts.Length == 2 && int.TryParse(parts[0], out int sw_) && int.TryParse(parts[1], out int sh_))
                {
                    p.Width = sw_;
                    p.Height = sh_;
                }
            }
            if (root.TryGetProperty("negative_prompt", out var np2) && np2.ValueKind == JsonValueKind.String)
                p.NegativePrompt = np2.GetString();
            bool wantB64 = root.TryGetProperty("response_format", out var rf) &&
                           rf.ValueKind == JsonValueKind.String && rf.GetString() == "b64_json";

            string outName = $"video-{Guid.NewGuid():N}.mp4";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            var sw = Stopwatch.StartNew();
            var result = await Task.Run(() =>
            {
                lock (_videoGenLock)
                {
                    var video = videoModel.GenerateVideo(prompt, p);
                    string codec = TensorSharp.Models.WanVideo.VideoIO.SaveMp4(outPath, video.Frames, video.Fps);
                    return (video, codec);
                }
            });
            sw.Stop();

            string url = BuildUploadUrl(outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "OpenAI video generation done: {F} frames -> {Url} ({Sec:F1}s)",
                result.video.Frames.Length, url, sw.Elapsed.TotalSeconds);
            string b64 = wantB64 ? Convert.ToBase64String(await File.ReadAllBytesAsync(outPath)) : null;
            return Results.Json(new
            {
                created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                data = new[] { new { url, b64_json = b64 } },
                width = result.video.Frames[0].Width,
                height = result.video.Frames[0].Height,
                frames = result.video.Frames.Length,
                fps = result.video.Fps,
                seed = result.video.Seed,
                codec = result.codec,
                elapsed_seconds = sw.Elapsed.TotalSeconds,
            });
        }

        /// <summary>
        /// <c>POST /api/video-generate/stream</c> — same JSON body as
        /// <see cref="VideoGenerateAsync"/> but streams SSE progress events
        /// (<c>{ videoGen, step, total }</c> per denoising step, then
        /// <c>{ done, url, ... }</c>) so the Web UI can show live progress.
        /// </summary>
        public async Task VideoGenerateStreamAsync(HttpContext ctx)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.VideoGenerate");
            SseWriter.ApplyHeaders(ctx.Response);
            var ct = ctx.RequestAborted;

            if (_svc.Model is not TensorSharp.Models.WanVideo.WanVideoModel videoModel)
            {
                await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "The loaded model is not a Wan video-generation model." }, ct);
                return;
            }

            string prompt;
            TensorSharp.Models.WanVideo.WanVideoParams p;
            try
            {
                var body = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ct);
                var root = body.RootElement;
                prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                p = ParseVideoParams(root, out string imgError);
                if (imgError != null)
                {
                    await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = imgError }, ct);
                    return;
                }
                if (string.IsNullOrWhiteSpace(prompt))
                {
                    await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "prompt is required." }, ct);
                    return;
                }
            }
            catch (Exception ex)
            {
                await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "Bad request: " + ex.Message }, ct);
                return;
            }

            string outName = $"video-{Guid.NewGuid():N}.mp4";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Video generate (stream): prompt='{Prompt}' {W}x{H}x{F}", prompt, p.Width, p.Height, p.Frames);

            var channel = Channel.CreateUnbounded<VideoFrame>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

            var genTask = Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                try
                {
                    lock (_videoGenLock)
                    {
                        p.OnStep = (step, total) =>
                        {
                            if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
                            channel.Writer.TryWrite(new VideoFrame { Step = step, Total = total });
                        };
                        var video = videoModel.GenerateVideo(prompt, p);
                        string codec = TensorSharp.Models.WanVideo.VideoIO.SaveMp4(outPath, video.Frames, video.Fps);
                        channel.Writer.TryWrite(new VideoFrame
                        {
                            Final = true, Url = BuildUploadUrl(outName),
                            Width = video.Frames[0].Width, Height = video.Frames[0].Height,
                            Frames = video.Frames.Length, Fps = video.Fps, Seed = video.Seed,
                            Codec = codec, Seconds = sw.Elapsed.TotalSeconds,
                        });
                    }
                }
                catch (OperationCanceledException)
                {
                    channel.Writer.TryWrite(new VideoFrame { Final = true, Error = "cancelled" });
                }
                catch (Exception ex)
                {
                    logger.LogError(LogEventIds.ChatFailed, ex, "Video generate (stream) failed");
                    channel.Writer.TryWrite(new VideoFrame { Final = true, Error = ex.Message });
                }
                finally { channel.Writer.Complete(); }
            }, CancellationToken.None);

            try
            {
                await foreach (var f in channel.Reader.ReadAllAsync(ct))
                {
                    if (f.Final)
                    {
                        if (f.Error == "cancelled") break;
                        if (f.Error != null)
                            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = f.Error }, ct);
                        else
                            await SseWriter.WriteEventAsync(ctx.Response, new
                            {
                                done = true, url = f.Url, width = f.Width, height = f.Height,
                                frames = f.Frames, fps = f.Fps, seed = f.Seed, codec = f.Codec,
                                elapsedSeconds = f.Seconds,
                            }, ct);
                        logger.LogInformation(LogEventIds.UploadReceived,
                            "Video generate (stream) done: {F} frames -> {Url} ({Sec:F1}s)", f.Frames, f.Url, f.Seconds);
                    }
                    else
                    {
                        await SseWriter.WriteEventAsync(ctx.Response,
                            new { videoGen = true, step = f.Step, total = f.Total }, ct);
                    }
                }
            }
            catch (OperationCanceledException) { /* client went away; the worker sees ct and stops */ }
            await genTask;
        }

        // A live denoising frame surfaced from the edit worker to the SSE writer: a progress tick
        // (Png == null) or a decoded preview image; the terminal frame carries the final result.
        private sealed class EditFrame
        {
            public int Step, Total, Width, Height;
            public byte[] Png;        // preview PNG bytes (null = progress-only tick)
            public bool Final;        // true on the terminal frame
            public string Url;        // final image URL (Final only)
            public double Seconds;    // total elapsed (Final only)
            public string Error;      // set if the edit threw
        }

        /// <summary>
        /// <c>POST /api/image-edit/stream</c> — same JSON body as <see cref="ImageEditAsync"/> but
        /// streams Server-Sent Events so the Web UI can show live denoising progress: a
        /// <c>{ preview, step, total, image? }</c> event per step (with a decoded snapshot on
        /// throttled steps) and a final <c>{ done, url, width, height, elapsedSeconds }</c>. This
        /// keeps the user informed that the (slow) diffusion is progressing instead of looking stuck.
        /// </summary>
        public async Task ImageEditStreamAsync(HttpContext ctx)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            SseWriter.ApplyHeaders(ctx.Response);
            var ct = ctx.RequestAborted;

            if (_svc.Model is not TensorSharp.Models.QwenImage.QwenImageModel editModel)
            {
                await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "The loaded model is not a Qwen-Image-Edit model." }, ct);
                return;
            }

            // Parse the Web UI JSON body (mirrors the JSON branch of ImageEditAsync).
            string prompt; int steps; float cfg; long seed; long targetArea = 0;
            var imageBytesList = new List<byte[]>();
            try
            {
                var body = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ct);
                var root = body.RootElement;
                prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                steps = root.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si) ? si : 0;   // 0 = auto
                cfg = root.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv) ? cv : 0f;  // 0 = auto
                seed = root.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) ? sv : 0;
                if (root.TryGetProperty("targetArea", out var ta) && ta.TryGetInt64(out long tav) && tav > 0)
                    targetArea = tav;
                string error = await ReadUploadedImagesAsync(root, imageBytesList, ct);
                if (error != null)
                {
                    await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error }, ct);
                    return;
                }
            }
            catch (Exception ex)
            {
                await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "Bad request: " + ex.Message }, ct);
                return;
            }

            string outName = $"edit-{Guid.NewGuid():N}.png";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit (stream): prompt='{Prompt}' steps={Steps} cfg={Cfg} images={Count} bytes={Bytes}",
                prompt, steps, cfg, imageBytesList.Count, imageBytesList.Sum(b => (long)b.Length));

            // The edit worker pushes frames into this channel; the SSE loop drains it. The callback
            // never blocks on the network (unbounded TryWrite) so it can't stall the denoise.
            var channel = Channel.CreateUnbounded<EditFrame>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
            // steps == 0 means "auto": the pipeline resolves the real count only later (e.g. a
            // Lightning LoRA's trained step count), so request the full preview budget and let
            // the pipeline's interval math fit it to the resolved steps. Clamping against the
            // raw 0 here disabled previews entirely for auto-step requests (the Web UI default).
            int previewCount = steps > 0 ? Math.Clamp(steps - 1, 0, 8) : 8;

            var editTask = Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                try
                {
                    // The model is not thread-safe; serialize edit requests (shared with ImageEditAsync).
                    lock (_imageEditLock)
                    {
                        var inputs = imageBytesList.ConvertAll(TensorSharp.Models.QwenImage.ImageIO.Decode);
                        var p = new TensorSharp.Models.QwenImage.QwenImageParams
                        {
                            Steps = steps,
                            CfgScale = cfg,
                            Seed = seed,
                            PreviewCount = previewCount,
                            OnStep = (step, total, preview) =>
                            {
                                if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
                                // Preview encoding is best-effort: a failure here must degrade to a
                                // plain progress tick (like the pipeline's own preview-decode guard),
                                // not abort a nearly-finished edit.
                                byte[] png = null;
                                if (preview != null)
                                {
                                    try { png = TensorSharp.Models.QwenImage.ImageIO.EncodePng(preview); }
                                    catch (Exception ex) { logger.LogWarning(LogEventIds.ChatFailed, ex, "Preview PNG encode failed; sending progress tick only"); }
                                }
                                channel.Writer.TryWrite(new EditFrame
                                {
                                    Step = step, Total = total, Png = png,
                                    Width = png != null ? preview.Width : 0, Height = png != null ? preview.Height : 0,
                                });
                            },
                        };
                        if (targetArea > 0) p.TargetArea = targetArea;
                        var output = editModel.EditImage(prompt, inputs, p);
                        TensorSharp.Models.QwenImage.ImageIO.SavePng(outPath, output);
                        channel.Writer.TryWrite(new EditFrame
                        {
                            Final = true, Url = BuildUploadUrl(outName),
                            Width = output.Width, Height = output.Height, Seconds = sw.Elapsed.TotalSeconds,
                        });
                    }
                }
                catch (OperationCanceledException)
                {
                    channel.Writer.TryWrite(new EditFrame { Final = true, Error = "cancelled" });
                }
                catch (Exception ex)
                {
                    logger.LogError(LogEventIds.ChatFailed, ex, "Image edit (stream) failed");
                    channel.Writer.TryWrite(new EditFrame { Final = true, Error = ex.Message });
                }
                finally { channel.Writer.Complete(); }
            }, CancellationToken.None);

            try
            {
                await foreach (var f in channel.Reader.ReadAllAsync(ct))
                {
                    if (f.Final)
                    {
                        if (f.Error == "cancelled") break;
                        if (f.Error != null)
                            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = f.Error }, ct);
                        else
                            await SseWriter.WriteEventAsync(ctx.Response,
                                new { done = true, url = f.Url, width = f.Width, height = f.Height, elapsedSeconds = f.Seconds }, ct);
                        logger.LogInformation(LogEventIds.UploadReceived,
                            "Image edit (stream) done: {W}x{H} -> {Url} ({Sec:F1}s)", f.Width, f.Height, f.Url, f.Seconds);
                    }
                    else
                    {
                        string image = f.Png != null ? "data:image/png;base64," + Convert.ToBase64String(f.Png) : null;
                        await SseWriter.WriteEventAsync(ctx.Response,
                            new { imageEdit = true, step = f.Step, total = f.Total, image, width = f.Width, height = f.Height }, ct);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Client disconnected; the worker observes ct via OnStep and unwinds.
            }

            // Drain the worker so its lock/VRAM is released before the next request (it finishes
            // promptly once cancellation is seen). Swallow — any error was already streamed.
            try { await editTask; } catch { /* already reported */ }
        }

        private static string BuildUploadUrl(string fileName)
        {
            return "/uploads/" + Uri.EscapeDataString(fileName);
        }

        internal static string BuildIncompletePdfImageWarning(int extractedPages, int totalPages)
        {
            if (totalPages <= 0 || extractedPages >= totalPages)
                return null;

            return $"Only {extractedPages} of {totalPages} PDF pages could be extracted as images. " +
                "The missing pages will not be sent to the model. If TS_PDF_MAX_PAGES is set, " +
                "unset or increase it; otherwise repair or convert the PDF.";
        }

        /// <summary>
        /// Optional cap on the number of PDF pages read during upload, from the
        /// <c>TS_PDF_MAX_PAGES</c> environment variable. Returns <c>0</c> (all pages)
        /// when unset or invalid. Extracted text is otherwise preserved in full.
        /// </summary>
        private static int ResolvePdfMaxPages()
        {
            string raw = Environment.GetEnvironmentVariable("TS_PDF_MAX_PAGES");
            if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out int v) && v > 0)
                return v;
            return 0;
        }

        // ---- Chat (SSE) -------------------------------------------------------

        public async Task ChatStreamAsync(HttpContext ctx)
        {
            var webUiLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.Chat");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

            string requestedModel = body.TryGetProperty("model", out var modelEl) ? modelEl.GetString() : null;
            string requestedBackend = body.TryGetProperty("backend", out var beEl) ? beEl.GetString() : null;
            bool newChat = body.TryGetProperty("newChat", out var ncProp) && ncProp.GetBoolean();
            string requestedSessionId = body.TryGetProperty("sessionId", out var sidEl) ? sidEl.GetString() : null;

            if (!WebUiChatPolicy.TryValidateChatRequest(requestedModel, requestedBackend, out string selectionError))
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat rejected: {Reason} (requestedModel={Model}, requestedBackend={Backend})",
                    selectionError, requestedModel ?? "(none)", requestedBackend ?? "(none)");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = selectionError });
                return;
            }

            ChatSession chatSession;
            if (!string.IsNullOrWhiteSpace(requestedSessionId))
            {
                chatSession = _sessions.GetSession(requestedSessionId);
                if (chatSession == null || chatSession.IsDisposed)
                {
                    webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/api/chat rejected: session '{SessionId}' not found or disposed", requestedSessionId);
                    ctx.Response.StatusCode = 404;
                    await ctx.Response.WriteAsJsonAsync(new { error = $"Session '{requestedSessionId}' not found or has been disposed." });
                    return;
                }
            }
            else
            {
                chatSession = _sessions.DefaultSession;
            }

            if (newChat)
            {
                webUiLogger.LogInformation(LogEventIds.SessionReset,
                    "/api/chat newChat=true; resetting session {SessionId}", chatSession.Id);
                _svc.ResetSession(chatSession);
            }

            if (!_svc.IsLoaded)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "No model loaded" });
                return;
            }

            var messagesEl = body.GetProperty("messages");
            int maxTokens = _options.ResolveMaxTokens(
                SamplingConfigParser.ReadRequestedMaxTokens(body, "maxTokens", "max_tokens"));

            var samplingConfig = SamplingConfigParser.ParseWebUi(body, _options.SamplingDefaults);
            bool uiThink = body.TryGetProperty("think", out var uiThinkProp) && uiThinkProp.GetBoolean();
            List<ToolFunction> uiTools = null;
            if (body.TryGetProperty("tools", out var uiToolsEl) && uiToolsEl.ValueKind == JsonValueKind.Array)
                uiTools = ToolFunctionParser.ParseOllama(body);

            var messages = ChatMessageParser.ParseWebUi(messagesEl);

            string attachmentError = ChatMessageParser.ValidateAttachmentPaths(messages, _options.UploadDirectory);
            if (attachmentError != null)
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat rejected: attachment path outside upload directory");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = attachmentError });
                return;
            }

            SseWriter.ApplyHeaders(ctx.Response);

            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            while (!ticket.IsReady)
            {
                await SseWriter.WriteEventAsync(ctx.Response,
                    WebUiSseEvents.QueueProgress(ticket.Position, _queue.PendingCount),
                    ctx.RequestAborted);
                await ticket.WaitAsync(TimeSpan.FromSeconds(1));
            }

            // DiffusionGemma streams a live "denoising preview" (whole-message replace per step)
            // rather than appended tokens, so it has its own SSE loop.
            if (_svc.IsDiffusionModel)
            {
                await ChatStreamDiffusionAsync(ctx, chatSession, messages, maxTokens, webUiLogger);
                return;
            }

            var sw = Stopwatch.StartNew();
            int tokenCount = 0;
            bool alwaysNeedsParsing = OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            bool useUiParser = uiThink || (uiTools != null && uiTools.Count > 0) || alwaysNeedsParsing;

            IOutputParser uiParser = null;
            if (useUiParser)
            {
                uiParser = OutputParserFactory.Create(_svc.Architecture);
                uiParser.Init(uiThink, uiTools);
            }

            bool aborted = false;
            string inferenceError = null;
            // Captured from the metrics tuple's done item so the final SSE event can
            // report how much of this turn's prompt was served from the prior turn's
            // KV cache. Defaults to zero in case the stream is aborted before
            // generation finishes.
            int turnPromptTokens = 0;
            int turnKvReusedTokens = 0;
            // Whether the answer was cut off by the token budget. The UI renders this
            // as a "response was truncated" hint, so a user staring at a sentence that
            // stops mid-word knows to raise max tokens rather than blame the model.
            bool turnTruncated = false;
            try
            {
                await foreach (var update
                    in _svc.ChatStreamWithMetricsAsync(chatSession, messages, maxTokens, ctx.RequestAborted, samplingConfig,
                        uiTools, uiThink))
                {
                    if (update.Done)
                    {
                        turnPromptTokens = update.PromptTokens;
                        turnKvReusedTokens = update.KvCacheReusedTokens;
                        turnTruncated = FinishReasonMapper.IsTruncated(update.FinishReason);
                        continue;
                    }

                    string piece = update.Piece;
                    if (string.IsNullOrEmpty(piece))
                        continue;

                    tokenCount++;
                    if (uiParser != null)
                    {
                        var parsed = uiParser.Add(piece, false);
                        if (!string.IsNullOrEmpty(parsed.Thinking))
                            await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Thinking(parsed.Thinking), ctx.RequestAborted);
                        if (!string.IsNullOrEmpty(parsed.Content))
                            await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Token(parsed.Content), ctx.RequestAborted);
                        if (parsed.ToolCalls != null)
                            await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.ToolCalls(parsed.ToolCalls), ctx.RequestAborted);
                    }
                    else
                    {
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Token(piece), ctx.RequestAborted);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                aborted = true;
                var chatLogger = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.WebUI.Chat");
                chatLogger.LogWarning(LogEventIds.ChatAborted,
                    "Web UI chat aborted by client (sessionId={SessionId}, partialTokens={PartialTokens})",
                    chatSession.Id, tokenCount);
            }
            catch (Exception ex)
            {
                var chatLogger = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.WebUI.Chat");
                chatLogger.LogError(LogEventIds.ChatFailed, ex,
                    "Web UI chat failed (sessionId={SessionId}, partialTokens={PartialTokens})",
                    chatSession.Id, tokenCount);
                inferenceError = ex.Message;
            }

            await FinalizeChatStreamAsync(ctx, uiParser, aborted, inferenceError, chatSession, sw, tokenCount,
                turnPromptTokens, turnKvReusedTokens, turnTruncated);
        }

        // ---- Chat (SSE) for DiffusionGemma: live denoising preview ------------

        private async Task ChatStreamDiffusionAsync(
            HttpContext ctx, ChatSession chatSession, List<ChatMessage> messages, int maxTokens, ILogger webUiLogger)
        {
            var sw = Stopwatch.StartNew();
            bool aborted = false;
            string inferenceError = null;
            int finalTokenCount = 0;
            int turnPromptTokens = 0;
            try
            {
                await foreach (var u in _svc.DiffusionChatStreamAsync(chatSession, messages, maxTokens, ctx.RequestAborted))
                {
                    if (u.Done)
                    {
                        finalTokenCount = u.EvalTokens;
                        turnPromptTokens = u.PromptTokens;
                        continue;
                    }
                    // Both intermediate previews and the final answer use whole-message replace.
                    await SseWriter.WriteEventAsync(ctx.Response,
                        WebUiSseEvents.Replace(u.Text, u.Step, u.TotalSteps, u.IsPreview), ctx.RequestAborted);
                }
            }
            catch (OperationCanceledException)
            {
                aborted = true;
                webUiLogger.LogWarning(LogEventIds.ChatAborted,
                    "Web UI diffusion chat aborted by client (sessionId={SessionId})", chatSession.Id);
            }
            catch (Exception ex)
            {
                webUiLogger.LogError(LogEventIds.ChatFailed, ex,
                    "Web UI diffusion chat failed (sessionId={SessionId})", chatSession.Id);
                inferenceError = ex.Message;
            }

            try
            {
                sw.Stop();
                double tokPerSec = finalTokenCount > 0 ? finalTokenCount / sw.Elapsed.TotalSeconds : 0;
                await SseWriter.WriteEventAsync(ctx.Response,
                    WebUiSseEvents.Done(finalTokenCount, sw.Elapsed.TotalSeconds, tokPerSec, aborted, inferenceError,
                        chatSession.Id, turnPromptTokens, 0));
            }
            catch (Exception)
            {
                // Best-effort final flush.
            }
        }

        private static async Task FinalizeChatStreamAsync(
            HttpContext ctx, IOutputParser uiParser, bool aborted, string inferenceError,
            ChatSession chatSession, Stopwatch sw, int tokenCount, int turnPromptTokens, int turnKvReusedTokens,
            bool truncated = false)
        {
            try
            {
                if (uiParser != null && !aborted)
                {
                    var finalParsed = uiParser.Add("", true);
                    if (!string.IsNullOrEmpty(finalParsed.Thinking))
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Thinking(finalParsed.Thinking));
                    if (!string.IsNullOrEmpty(finalParsed.Content))
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Token(finalParsed.Content));
                    if (finalParsed.ToolCalls != null)
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.ToolCalls(finalParsed.ToolCalls));
                }

                sw.Stop();
                double tokPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
                await SseWriter.WriteEventAsync(ctx.Response,
                    WebUiSseEvents.Done(tokenCount, sw.Elapsed.TotalSeconds, tokPerSec, aborted, inferenceError, chatSession.Id,
                        turnPromptTokens, turnKvReusedTokens, truncated));
            }
            catch (Exception)
            {
                // Best-effort final flush; if the client has already left we silently drop the trailing frames.
            }
        }
    }
}
