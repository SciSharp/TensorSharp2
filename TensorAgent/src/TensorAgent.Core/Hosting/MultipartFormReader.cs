// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;

namespace TensorAgent.Core.Hosting;

/// <summary>One file part of a multipart form, spooled to a temp file so a video upload
/// never has to fit in memory twice.</summary>
public sealed class MultipartFile : IDisposable
{
    public string FieldName { get; init; } = string.Empty;
    public string FileName { get; init; } = string.Empty;
    public string ContentType { get; init; } = "application/octet-stream";
    public string TempPath { get; init; } = string.Empty;
    public long Length { get; init; }

    public FileStream OpenRead() => new(TempPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: true);

    public void Dispose()
    {
        try { if (File.Exists(TempPath)) File.Delete(TempPath); } catch { }
    }
}

/// <summary>The parsed form: text fields and files.</summary>
public sealed class MultipartForm : IDisposable
{
    public Dictionary<string, string> Fields { get; } = new(StringComparer.OrdinalIgnoreCase);
    public List<MultipartFile> Files { get; } = new();

    public string? this[string name] => Fields.TryGetValue(name, out string? v) ? v : null;

    public void Dispose()
    {
        foreach (MultipartFile f in Files) f.Dispose();
    }
}

/// <summary>
/// Streams a <c>multipart/form-data</c> body (what the Web UI's FormData upload sends)
/// without ASP.NET: boundary-delimited parts, headers, file parts spooled to disk.
/// </summary>
public static class MultipartFormReader
{
    public static Task<MultipartForm> ReadAsync(Stream body, string contentType, CancellationToken ct, string? spoolDirectory = null) =>
        ReadAsync(body, contentType, ct, long.MaxValue, int.MaxValue, spoolDirectory);

    internal static async Task<MultipartForm> ReadAsync(Stream body, string contentType, CancellationToken ct,
        long maxBodyBytes, int maxFiles, string? spoolDirectory = null)
    {
        string? boundary = GetBoundary(contentType);
        if (string.IsNullOrEmpty(boundary) || boundary.Length > 128)
            throw new InvalidDataException("multipart/form-data has a missing or invalid boundary");

        byte[] delimiter = Encoding.ASCII.GetBytes("\r\n--" + boundary);
        byte[] first = Encoding.ASCII.GetBytes("--" + boundary);
        var form = new MultipartForm();
        spoolDirectory ??= Path.GetTempPath();

        string? currentTemp = null;
        try
        {
            var reader = new BufferedReader(body, maxBodyBytes);
            // Skip the preamble up to the first boundary line.
            if (!await reader.SkipPastAsync(first, ct).ConfigureAwait(false))
                return form;
            while (true)
            {
                // After a boundary: either "--" (final) or CRLF then headers.
                byte[] tail = await reader.ReadExactAsync(2, ct).ConfigureAwait(false);
                if (tail.Length < 2)
                    throw new InvalidDataException("multipart body ended before the closing boundary");
                if (tail[0] == '-' && tail[1] == '-')
                    break;
                if (tail[0] != '\r' || tail[1] != '\n')
                    throw new InvalidDataException("multipart boundary has an invalid terminator");
                var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                while (true)
                {
                    string? line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
                    if (line is null)
                        throw new InvalidDataException("multipart headers ended unexpectedly");
                    if (line.Length == 0)
                        break;
                    int colon = line.IndexOf(':');
                    if (colon > 0)
                        headers[line[..colon].Trim()] = line[(colon + 1)..].Trim();
                }

                string disposition = headers.TryGetValue("Content-Disposition", out string? d) ? d : string.Empty;
                string name = ParseParam(disposition, "name") ?? string.Empty;
                string? fileName = ParseParam(disposition, "filename");
                string partType = headers.TryGetValue("Content-Type", out string? t) ? t : "application/octet-stream";

                if (fileName is null)
                {
                    using var ms = new MemoryStream();
                    await reader.CopyUntilAsync(delimiter, ms, ct).ConfigureAwait(false);
                    form.Fields[name] = Encoding.UTF8.GetString(ms.ToArray());
                }
                else
                {
                    if (form.Files.Count >= maxFiles)
                        throw new InvalidDataException($"At most {maxFiles} files can be uploaded in one request.");
                    string temp = Path.Combine(spoolDirectory, "upload-" + Guid.NewGuid().ToString("N") + ".part");
                    currentTemp = temp;
                    long length;
                    await using (var fs = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true))
                    {
                        await reader.CopyUntilAsync(delimiter, fs, ct).ConfigureAwait(false);
                        length = fs.Length;
                    }
                    form.Files.Add(new MultipartFile
                    {
                        FieldName = name,
                        FileName = Path.GetFileName(fileName),
                        ContentType = partType,
                        TempPath = temp,
                        Length = length,
                    });
                    currentTemp = null;
                }
            }
            return form;
        }
        catch
        {
            form.Dispose();
            if (currentTemp != null)
            {
                try { File.Delete(currentTemp); } catch { }
            }
            throw;
        }
    }

    private static string? GetBoundary(string contentType)
    {
        foreach (string part in contentType.Split(';'))
        {
            string p = part.Trim();
            if (p.StartsWith("boundary=", StringComparison.OrdinalIgnoreCase))
                return p["boundary=".Length..].Trim('"');
        }
        return null;
    }

    private static string? ParseParam(string header, string name)
    {
        foreach (string part in header.Split(';'))
        {
            string p = part.Trim();
            if (p.StartsWith(name + "=", StringComparison.OrdinalIgnoreCase))
                return p[(name.Length + 1)..].Trim('"');
        }
        return null;
    }

    /// <summary>Byte-level reader with delimiter search over a rolling buffer.</summary>
    private sealed class BufferedReader(Stream stream, long maxBodyBytes)
    {
        private readonly byte[] _buf = new byte[1 << 16];
        private int _start, _end;
        private bool _eof;
        private long _readBytes;

        private async Task<bool> FillAsync(CancellationToken ct)
        {
            if (_eof) return false;
            if (_start > 0)
            {
                Buffer.BlockCopy(_buf, _start, _buf, 0, _end - _start);
                _end -= _start;
                _start = 0;
            }
            if (_end == _buf.Length)
                return true;
            int n = await stream.ReadAsync(_buf.AsMemory(_end, _buf.Length - _end), ct).ConfigureAwait(false);
            if (n == 0) { _eof = true; return false; }
            if (n > maxBodyBytes - _readBytes)
                throw new TensorSharp.Server.Hosting.UploadLimitExceededException("Upload exceeds the request size limit.", 413);
            _readBytes += n;
            _end += n;
            return true;
        }

        /// <summary>
        /// Advance past the next occurrence of <paramref name="marker"/>, reading as much
        /// as it takes. False means end of stream with no match.
        ///
        /// <para>
        /// Search, THEN discard what was searched, THEN refill — in that order, which is
        /// the same shape <see cref="CopyUntilAsync"/> uses and which this did not. It
        /// discarded after filling and only when the buffer came back completely full,
        /// so a read that returned exactly 65536 bytes — a whole buffer — had those bytes
        /// thrown away before anything looked at them. The very first read of a request
        /// is the one that matters: the opening <c>--boundary</c> sits at offset 0, and
        /// losing it means the parse runs to end of stream and reports a form with no
        /// files. The route above answers that with "no file was uploaded", which is what
        /// every photo attached on the phone was told.
        /// </para>
        /// <para>
        /// The tail kept back is a possible SPLIT marker — one whose first bytes are at
        /// the end of this buffer and whose rest has not been read yet — so it is
        /// <c>marker.Length - 1</c> bytes, never the whole marker: keeping the whole of
        /// it would re-search bytes already rejected, and keeping none of it would miss a
        /// boundary that straddles two reads.
        /// </para>
        /// </summary>
        public async Task<bool> SkipPastAsync(byte[] marker, CancellationToken ct)
        {
            while (true)
            {
                int idx = IndexOf(marker);
                if (idx >= 0) { _start = idx + marker.Length; return true; }
                _start = _end - Math.Min(marker.Length - 1, _end - _start);
                if (!await FillAsync(ct).ConfigureAwait(false)) return false;
            }
        }

        public async Task<byte[]> ReadExactAsync(int count, CancellationToken ct)
        {
            while (_end - _start < count)
                if (!await FillAsync(ct).ConfigureAwait(false)) break;
            int n = Math.Min(count, _end - _start);
            byte[] r = _buf[_start..(_start + n)];
            _start += n;
            return r;
        }

        public async Task<string?> ReadLineAsync(CancellationToken ct)
        {
            while (true)
            {
                for (int i = _start; i < _end; i++)
                {
                    if (_buf[i] == (byte)'\n')
                    {
                        int len = i - _start;
                        if (len > 0 && _buf[i - 1] == (byte)'\r') len--;
                        string line = Encoding.UTF8.GetString(_buf, _start, len);
                        _start = i + 1;
                        return line;
                    }
                }
                if (_end - _start == _buf.Length)
                    throw new InvalidDataException("multipart header is too long");
                if (!await FillAsync(ct).ConfigureAwait(false))
                {
                    if (_end == _start) return null;
                    string line = Encoding.UTF8.GetString(_buf, _start, _end - _start);
                    _start = _end;
                    return line;
                }
            }
        }

        /// <summary>Copies bytes to <paramref name="target"/> until the delimiter, consuming it.</summary>
        public async Task CopyUntilAsync(byte[] delimiter, Stream target, CancellationToken ct)
        {
            while (true)
            {
                int idx = IndexOf(delimiter);
                if (idx >= 0)
                {
                    await target.WriteAsync(_buf.AsMemory(_start, idx - _start), ct).ConfigureAwait(false);
                    _start = idx + delimiter.Length;
                    return;
                }
                // Flush everything except a possible partial delimiter at the end.
                int keep = Math.Min(delimiter.Length - 1, _end - _start);
                int flush = _end - _start - keep;
                if (flush > 0)
                {
                    await target.WriteAsync(_buf.AsMemory(_start, flush), ct).ConfigureAwait(false);
                    _start += flush;
                }
                if (!await FillAsync(ct).ConfigureAwait(false))
                {
                    await target.WriteAsync(_buf.AsMemory(_start, _end - _start), ct).ConfigureAwait(false);
                    _start = _end;
                    throw new InvalidDataException("multipart body ended before the closing boundary");
                }
            }
        }

        private int IndexOf(byte[] marker)
        {
            int limit = _end - marker.Length;
            for (int i = _start; i <= limit; i++)
            {
                if (_buf[i] != marker[0]) continue;
                int j = 1;
                while (j < marker.Length && _buf[i + j] == marker[j]) j++;
                if (j == marker.Length) return i;
            }
            return -1;
        }
    }
}
