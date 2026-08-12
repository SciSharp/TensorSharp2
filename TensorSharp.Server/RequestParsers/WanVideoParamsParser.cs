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
using System.Text.Json;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Parses the parameters shared by all Wan video-generation endpoints.
    /// Startup <c>--video-frames</c>/<c>--fps</c> values seed the request;
    /// explicitly supplied JSON fields then override those defaults.
    /// </summary>
    internal static class WanVideoParamsParser
    {
        public static WanVideoParams Parse(
            JsonElement root,
            ServerHostingOptions options,
            out string error)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));

            error = null;
            var p = new WanVideoParams
            {
                Frames = options.DefaultWanVideoFrames,
                Fps = options.DefaultWanVideoFps,
            };

            if (root.TryGetProperty("width", out var w) && w.TryGetInt32(out int wi)) p.Width = wi;
            if (root.TryGetProperty("height", out var h) && h.TryGetInt32(out int hi)) p.Height = hi;
            if (root.TryGetProperty("frames", out var f) && f.TryGetInt32(out int fi)) p.Frames = fi;
            if (root.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si)) p.Steps = si;
            if (root.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv)) p.CfgScale = cv;
            if (root.TryGetProperty("cfg2", out var cf2) && cf2.TryGetSingle(out float cv2)) p.CfgScale2 = cv2;
            if (root.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) && sv != 0) p.Seed = sv;
            if (root.TryGetProperty("fps", out var fp) && fp.TryGetInt32(out int fv)) p.Fps = fv;
            if (root.TryGetProperty("flowShift", out var fs) && fs.TryGetSingle(out float fsv)) p.FlowShift = fsv;
            if (root.TryGetProperty("negativePrompt", out var np) && np.ValueKind == JsonValueKind.String)
                p.NegativePrompt = np.GetString();
            if (root.TryGetProperty("sampler", out var sm) && sm.ValueKind == JsonValueKind.String)
                p.Sampler = sm.GetString();

            // Conditioning image for Wan 2.2 image-to-video: either a previously uploaded
            // file ("imagePath", Web UI flow) or inline base64 ("image", API flow; a
            // data:...;base64, prefix is accepted).
            if (root.TryGetProperty("imagePath", out var ip) && ip.ValueKind == JsonValueKind.String
                && !string.IsNullOrWhiteSpace(ip.GetString()))
            {
                string uploadRoot = Path.GetFullPath(options.UploadDirectory);
                string full = Path.GetFullPath(ip.GetString());
                if (!full.StartsWith(uploadRoot, StringComparison.OrdinalIgnoreCase) || !File.Exists(full))
                {
                    error = "imagePath must reference a previously uploaded file.";
                    return p;
                }
                p.ImageBytes = File.ReadAllBytes(full);
            }
            else if (root.TryGetProperty("image", out var ib) && ib.ValueKind == JsonValueKind.String
                     && !string.IsNullOrWhiteSpace(ib.GetString()))
            {
                string b64 = ib.GetString();
                int comma = b64.IndexOf(',');
                if (b64.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && comma > 0)
                    b64 = b64[(comma + 1)..];
                try { p.ImageBytes = Convert.FromBase64String(b64); }
                catch (FormatException) { error = "image must be base64-encoded (optionally a data: URL)."; }
            }

            return p;
        }
    }
}
