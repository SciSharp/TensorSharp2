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
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Pure parsers that translate request bodies for each protocol into the
    /// shared <see cref="ChatMessage"/> shape consumed by <see cref="ModelService"/>.
    ///
    /// The Web UI sends already-uploaded file paths inline, while the Ollama and
    /// OpenAI flavours embed binary content as base64. Whenever the parser
    /// materialises new bytes to disk it writes them into the shared upload
    /// directory so subsequent attachments (and ChatStream's media injector)
    /// resolve the same way regardless of how the request arrived.
    /// </summary>
    internal static class ChatMessageParser
    {
        /// <summary>
        /// Parse the messages array from the Web UI's <c>/api/chat</c> body.
        /// The UI references previously uploaded files by absolute path, so we
        /// don't decode any binary content here.
        /// </summary>
        public static List<ChatMessage> ParseWebUi(JsonElement messagesEl)
        {
            var messages = new List<ChatMessage>();
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.GetProperty("role").GetString(),
                    Content = msgEl.GetProperty("content").GetString()
                };

                if (msgEl.TryGetProperty("imagePaths", out var imgs) && imgs.GetArrayLength() > 0)
                    msg.ImagePaths = imgs.EnumerateArray().Select(e => e.GetString()).ToList();

                if (msgEl.TryGetProperty("audioPaths", out var auds) && auds.GetArrayLength() > 0)
                    msg.AudioPaths = auds.EnumerateArray().Select(e => e.GetString()).ToList();

                // Text uploads inline their content into msg.Content; the original file
                // paths are surfaced separately so the per-turn audit log can record
                // which uploaded files belonged to this message.
                if (msgEl.TryGetProperty("textFilePaths", out var texts) && texts.GetArrayLength() > 0)
                    msg.TextFilePaths = texts.EnumerateArray().Select(e => e.GetString()).ToList();

                if (msgEl.TryGetProperty("isVideo", out var iv))
                    msg.IsVideo = iv.GetBoolean();

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// Resolve every client-supplied attachment reference (imagePaths /
        /// audioPaths / textFilePaths) to a full path inside the upload
        /// directory, rewriting the lists in place. The Web UI sends the bare
        /// server filenames returned by <c>/api/upload</c>; absolute paths from
        /// older clients are still accepted when they resolve inside the upload
        /// directory. Returns an error message, or null when everything
        /// resolved.
        /// </summary>
        public static string ResolveAttachmentPaths(List<ChatMessage> messages, string uploadRoot)
        {
            if (messages == null)
                return null;

            foreach (var msg in messages)
            {
                string error = ResolvePathList(msg?.ImagePaths, uploadRoot)
                    ?? ResolvePathList(msg?.AudioPaths, uploadRoot)
                    ?? ResolvePathList(msg?.TextFilePaths, uploadRoot);
                if (error != null)
                    return error;
            }
            return null;
        }

        private static string ResolvePathList(List<string> paths, string uploadRoot)
        {
            if (paths == null)
                return null;

            for (int i = 0; i < paths.Count; i++)
            {
                if (!UploadFileReference.TryResolve(uploadRoot, paths[i], out string full))
                    return "Attachment path must reference a previously uploaded file.";
                paths[i] = full;
            }
            return null;
        }

        /// <summary>
        /// Parse Ollama's messages array. Ollama embeds images per-message as a
        /// base64 array under <c>"images"</c>; we materialise each one as a PNG
        /// in the upload directory and reference them by absolute path.
        /// </summary>
        public static List<ChatMessage> ParseOllama(JsonElement messagesEl, string uploadDir)
        {
            var messages = new List<ChatMessage>();
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user",
                    Content = msgEl.TryGetProperty("content", out var c) ? c.GetString() : ""
                };

                if (msgEl.TryGetProperty("images", out var imgs) && imgs.ValueKind == JsonValueKind.Array)
                {
                    msg.ImagePaths = new List<string>();
                    foreach (var imgEl in imgs.EnumerateArray())
                    {
                        string b64 = imgEl.GetString();
                        if (string.IsNullOrEmpty(b64))
                            continue;

                        string path = WriteBase64Image(b64, uploadDir);
                        msg.ImagePaths.Add(path);
                    }
                }

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// Parse OpenAI's messages array which uses an "input parts" structure:
        /// either a plain string content, or an array of parts where each part
        /// is either text or an image (data URL or external URL).
        /// </summary>
        public static List<ChatMessage> ParseOpenAI(JsonElement messagesEl, string uploadDir)
        {
            var messages = new List<ChatMessage>();
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user"
                };

                if (msgEl.TryGetProperty("content", out var contentEl))
                {
                    if (contentEl.ValueKind == JsonValueKind.String)
                    {
                        msg.Content = contentEl.GetString();
                    }
                    else if (contentEl.ValueKind == JsonValueKind.Array)
                    {
                        var textParts = new List<string>();
                        msg.ImagePaths = new List<string>();
                        msg.AudioPaths = new List<string>();

                        foreach (var part in contentEl.EnumerateArray())
                        {
                            string type = part.TryGetProperty("type", out var t) ? t.GetString() : "";
                            if (type == "text" && part.TryGetProperty("text", out var txt))
                            {
                                textParts.Add(txt.GetString());
                            }
                            else if (type == "image_url" && part.TryGetProperty("image_url", out var imgUrl))
                            {
                                string url = imgUrl.TryGetProperty("url", out var u) ? u.GetString() : "";
                                if (!string.IsNullOrEmpty(url) && url.StartsWith("data:"))
                                {
                                    int commaIdx = url.IndexOf(',');
                                    if (commaIdx > 0)
                                    {
                                        string b64 = url.Substring(commaIdx + 1);
                                        string path = WriteBase64Image(b64, uploadDir);
                                        msg.ImagePaths.Add(path);
                                    }
                                }
                            }
                            else if (type == "input_audio" && part.TryGetProperty("input_audio", out var audioEl))
                            {
                                // OpenAI audio part: {"data": "<base64>", "format": "wav"|"mp3"}.
                                string path = WriteBase64Audio(
                                    audioEl.TryGetProperty("data", out var aData) ? aData.GetString() : null,
                                    audioEl.TryGetProperty("format", out var aFmt) ? aFmt.GetString() : null,
                                    uploadDir);
                                if (path != null)
                                    msg.AudioPaths.Add(path);
                            }
                            else if (type == "audio_url" && part.TryGetProperty("audio_url", out var audioUrl))
                            {
                                // Some clients mirror the image_url shape for audio.
                                string url = audioUrl.ValueKind == JsonValueKind.String
                                    ? audioUrl.GetString()
                                    : (audioUrl.TryGetProperty("url", out var au) ? au.GetString() : null);
                                string path = WriteBase64AudioDataUri(url, uploadDir);
                                if (path != null)
                                    msg.AudioPaths.Add(path);
                            }
                        }

                        msg.Content = string.Join("\n", textParts);
                        if (msg.ImagePaths.Count == 0) msg.ImagePaths = null;
                        if (msg.AudioPaths.Count == 0) msg.AudioPaths = null;
                    }
                }

                // Message-level base64 audio array (the same shorthand Ollama uses
                // for images), accepted alongside the OpenAI content-part form.
                AppendMessageLevelAudios(msgEl, msg, uploadDir);

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// Parse the Responses API <c>input</c> field, which is either a plain
        /// string (shorthand for a single user turn) or an array of message
        /// items: <c>{role, content: string | [{type:"input_text"|"output_text", text} |
        /// {type:"input_image", image_url}]}</c>. An optional <c>instructions</c>
        /// string is prepended as a system message, matching how the real API
        /// folds it into the model's system prompt.
        /// </summary>
        public static List<ChatMessage> ParseResponsesInput(JsonElement inputEl, string instructions, string uploadDir)
        {
            var messages = new List<ChatMessage>();

            if (!string.IsNullOrEmpty(instructions))
                messages.Add(new ChatMessage { Role = "system", Content = instructions });

            if (inputEl.ValueKind == JsonValueKind.String)
            {
                messages.Add(new ChatMessage { Role = "user", Content = inputEl.GetString() ?? "" });
                return messages;
            }

            if (inputEl.ValueKind != JsonValueKind.Array)
                return messages;

            foreach (var itemEl in inputEl.EnumerateArray())
            {
                // Only message-shaped items are supported; other item types
                // (function_call, function_call_output, reasoning, ...) are
                // ignored for this MVP surface.
                string itemType = itemEl.TryGetProperty("type", out var t) ? t.GetString() : "message";
                if (itemType != "message")
                    continue;

                var msg = new ChatMessage
                {
                    Role = itemEl.TryGetProperty("role", out var r) ? r.GetString() : "user",
                };

                if (!itemEl.TryGetProperty("content", out var contentEl))
                {
                    messages.Add(msg);
                    continue;
                }

                if (contentEl.ValueKind == JsonValueKind.String)
                {
                    msg.Content = contentEl.GetString();
                    messages.Add(msg);
                    continue;
                }

                if (contentEl.ValueKind == JsonValueKind.Array)
                {
                    var textParts = new List<string>();
                    msg.ImagePaths = new List<string>();
                    msg.AudioPaths = new List<string>();

                    foreach (var part in contentEl.EnumerateArray())
                    {
                        string partType = part.TryGetProperty("type", out var pt) ? pt.GetString() : "";
                        if ((partType == "input_text" || partType == "output_text") &&
                            part.TryGetProperty("text", out var txt))
                        {
                            textParts.Add(txt.GetString());
                        }
                        else if (partType == "input_audio" && part.TryGetProperty("input_audio", out var audioEl))
                        {
                            string audioPath = WriteBase64Audio(
                                audioEl.TryGetProperty("data", out var aData) ? aData.GetString() : null,
                                audioEl.TryGetProperty("format", out var aFmt) ? aFmt.GetString() : null,
                                uploadDir);
                            if (audioPath != null)
                                msg.AudioPaths.Add(audioPath);
                        }
                        else if (partType == "input_image")
                        {
                            string url = part.TryGetProperty("image_url", out var u)
                                ? (u.ValueKind == JsonValueKind.String ? u.GetString() : null)
                                : null;
                            if (!string.IsNullOrEmpty(url) && url.StartsWith("data:"))
                            {
                                int commaIdx = url.IndexOf(',');
                                if (commaIdx > 0)
                                {
                                    string b64 = url.Substring(commaIdx + 1);
                                    msg.ImagePaths.Add(WriteBase64Image(b64, uploadDir));
                                }
                            }
                        }
                    }

                    msg.Content = string.Join("\n", textParts);
                    if (msg.ImagePaths.Count == 0) msg.ImagePaths = null;
                    if (msg.AudioPaths.Count == 0) msg.AudioPaths = null;
                }

                messages.Add(msg);
            }

            return messages;
        }

        /// <summary>
        /// Decode the top-level <c>"images"</c> base64 array used by Ollama's
        /// <c>/api/generate</c>. Returns null when no images are present so the
        /// downstream code path can short-circuit cleanly.
        /// </summary>
        public static List<string> DecodeBase64Images(JsonElement body, string uploadDir)
        {
            if (!body.TryGetProperty("images", out var imgs) || imgs.ValueKind != JsonValueKind.Array)
                return null;

            var paths = new List<string>();
            foreach (var imgEl in imgs.EnumerateArray())
            {
                string b64 = imgEl.GetString();
                if (string.IsNullOrEmpty(b64))
                    continue;

                paths.Add(WriteBase64Image(b64, uploadDir));
            }
            return paths.Count > 0 ? paths : null;
        }

        private static string WriteBase64Image(string base64, string uploadDir)
        {
            byte[] imgData = Convert.FromBase64String(base64);
            string path = Path.Combine(uploadDir, $"{Guid.NewGuid():N}.png");
            File.WriteAllBytes(path, imgData);
            return path;
        }

        /// <summary>
        /// Materialise a base64 audio attachment in the upload directory. The
        /// audio preprocessor picks its decoder from the file extension, so the
        /// declared <paramref name="format"/> (OpenAI sends "wav" / "mp3") is
        /// mapped to one — falling back to sniffing the container header when it
        /// is missing or unrecognised. Returns null for empty/invalid content.
        /// </summary>
        private static string WriteBase64Audio(string base64, string format, string uploadDir)
        {
            if (string.IsNullOrWhiteSpace(base64))
                return null;

            // Tolerate a full data URI in the data field.
            int commaIdx = base64.StartsWith("data:") ? base64.IndexOf(',') : -1;
            if (commaIdx > 0)
            {
                if (string.IsNullOrEmpty(format))
                    format = MimeToAudioFormat(base64.Substring(5, commaIdx - 5));
                base64 = base64.Substring(commaIdx + 1);
            }

            byte[] data;
            try
            {
                data = Convert.FromBase64String(base64);
            }
            catch (FormatException)
            {
                return null;
            }
            if (data.Length == 0)
                return null;

            string path = Path.Combine(uploadDir, $"{Guid.NewGuid():N}{AudioExtension(format, data)}");
            File.WriteAllBytes(path, data);
            return path;
        }

        private static string WriteBase64AudioDataUri(string url, string uploadDir)
        {
            if (string.IsNullOrEmpty(url) || !url.StartsWith("data:"))
                return null;
            return WriteBase64Audio(url, null, uploadDir);
        }

        private static string MimeToAudioFormat(string mimeAndParams)
        {
            int semi = mimeAndParams.IndexOf(';');
            string mime = semi >= 0 ? mimeAndParams.Substring(0, semi) : mimeAndParams;
            int slash = mime.IndexOf('/');
            return slash >= 0 ? mime.Substring(slash + 1) : mime;
        }

        private static string AudioExtension(string format, byte[] data)
        {
            switch ((format ?? string.Empty).Trim().TrimStart('.').ToLowerInvariant())
            {
                case "wav":
                case "wave":
                case "x-wav":
                case "pcm":
                    return ".wav";
                case "mp3":
                case "mpeg":
                case "mpga":
                    return ".mp3";
                case "ogg":
                case "oga":
                case "vorbis":
                    return ".ogg";
            }

            if (data.Length >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F'
                && data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E')
                return ".wav";
            if (data.Length >= 4 && data[0] == 'O' && data[1] == 'g' && data[2] == 'g' && data[3] == 'S')
                return ".ogg";
            return ".mp3";                 // ID3-tagged or bare MPEG frames
        }

        /// <summary>
        /// Read a message-level <c>"audios"</c> array of base64 clips (optionally
        /// data URIs) and append them to the message's audio attachments.
        /// </summary>
        private static void AppendMessageLevelAudios(JsonElement msgEl, ChatMessage msg, string uploadDir)
        {
            if (!msgEl.TryGetProperty("audios", out var auds) || auds.ValueKind != JsonValueKind.Array)
                return;

            foreach (var audEl in auds.EnumerateArray())
            {
                string b64 = audEl.ValueKind == JsonValueKind.String
                    ? audEl.GetString()
                    : (audEl.TryGetProperty("data", out var d) ? d.GetString() : null);
                string format = audEl.ValueKind == JsonValueKind.Object &&
                                audEl.TryGetProperty("format", out var f) ? f.GetString() : null;
                string path = WriteBase64Audio(b64, format, uploadDir);
                if (path == null)
                    continue;
                (msg.AudioPaths ??= new List<string>()).Add(path);
            }
        }
    }
}
