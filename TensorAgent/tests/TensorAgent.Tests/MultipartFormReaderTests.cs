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
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// The multipart parser, read the way a socket actually hands it bytes.
///
/// <para>
/// Every other test of this path drives it through a real <c>HttpListener</c> on
/// loopback, and that is exactly why it hid a bug for months: the managed listener
/// parses the request headers out of an 8 kB buffer and hands the leftover body bytes
/// back first, so the parser's FIRST read is always short and never fills its own
/// buffer. The bug needed a read that returned all 65536 bytes the parser asked for —
/// which is what happens once that leftover is drained, and which is what the phone was
/// doing every time somebody attached a photo.
/// </para>
/// <para>
/// So these drive the parser directly, over a stream whose read size is the variable.
/// </para>
/// </summary>
public sealed class MultipartFormReaderTests
{
    private const string Boundary = "----WebKitFormBoundaryQrJ0k9y2mLc8Xv3P";

    /// <summary>
    /// The size that used to break it, and every size around it.
    ///
    /// <para>
    /// 65536 is the parser's own buffer. A read that returns exactly that many bytes
    /// filled the buffer, and the line that dropped everything but a possible split
    /// marker ran AFTER the fill instead of before the search — so the opening
    /// <c>--boundary</c>, sitting at offset 0, was thrown away unread, and the parse ran
    /// to end of stream and reported a form with no files in it. The route above it says
    /// that as "no file was uploaded", which is what every iPhone photo was answered
    /// with.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(1460)]      // one ethernet frame, which is what a socket usually gives
    [InlineData(8192)]
    [InlineData(65535)]     // one short of the buffer
    [InlineData(65536)]     // exactly the buffer: the failing case
    [InlineData(1 << 20)]   // more than the buffer, so every read fills it
    public async Task APhotoParsesWhateverSizeTheReadsComeBackIn(int readSize)
    {
        byte[] photo = Photo(1_234_567);
        byte[] body = Body("IMG_0004.jpeg", "image/jpeg", photo);

        using MultipartForm form = await MultipartFormReader.ReadAsync(
            new ChoppyStream(body, readSize), "multipart/form-data; boundary=" + Boundary,
            CancellationToken.None, Path.GetTempPath());

        MultipartFile file = Assert.Single(form.Files);
        Assert.Equal("IMG_0004.jpeg", file.FileName);
        Assert.Equal("image/jpeg", file.ContentType);
        Assert.Equal(photo.Length, file.Length);
        Assert.Equal(photo, await File.ReadAllBytesAsync(file.TempPath));
    }

    /// <summary>
    /// The same reads, with a text field in front of the file: the preamble scan and the
    /// per-part scans are different loops, and only one of them had the fault.
    /// </summary>
    [Theory]
    [InlineData(65536)]
    [InlineData(1 << 20)]
    public async Task AFieldBeforeTheFileSurvivesTheSameReads(int readSize)
    {
        byte[] photo = Photo(200_000);
        byte[] body = Body("shot.png", "image/png", photo, field: ("overwrite", "true"));

        using MultipartForm form = await MultipartFormReader.ReadAsync(
            new ChoppyStream(body, readSize), "multipart/form-data; boundary=" + Boundary,
            CancellationToken.None, Path.GetTempPath());

        Assert.Equal("true", form["overwrite"]);
        Assert.Equal(photo.Length, Assert.Single(form.Files).Length);
    }

    /// <summary>
    /// A body with no boundary in it at all still ends, rather than looping forever over
    /// a buffer it keeps refilling. This is the case the discarded tail was there for,
    /// and moving it must not lose it.
    /// </summary>
    [Fact]
    public async Task ABodyThatNeverContainsTheBoundaryEndsWithAnEmptyForm()
    {
        using MultipartForm form = await MultipartFormReader.ReadAsync(
            new ChoppyStream(Photo(300_000), 65536), "multipart/form-data; boundary=" + Boundary,
            CancellationToken.None, Path.GetTempPath());

        Assert.Empty(form.Files);
        Assert.Empty(form.Fields);
    }

    [Theory]
    [InlineData("limit")]
    [InlineData("count")]
    [InlineData("truncated")]
    [InlineData("boundary")]
    public async Task FailedBatchParsingDeletesBothCompleteAndPartialSpoolFiles(string failure)
    {
        string spool = Path.Combine(Path.GetTempPath(), "tensoragent-batch-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(spool);
        try
        {
            using var multipart = new MultipartFormDataContent();
            multipart.Add(new StringContent("first"), "files", "one.txt");
            multipart.Add(new StringContent(new string('x', 2000)), "files", "two.txt");
            byte[] bytes = await multipart.ReadAsByteArrayAsync();
            if (failure == "truncated") bytes = bytes[..^100];
            if (failure == "boundary")
            {
                string boundary = multipart.Headers.ContentType!.Parameters.Single(p => p.Name == "boundary").Value!.Trim('"');
                string text = Encoding.UTF8.GetString(bytes);
                bytes = Encoding.UTF8.GetBytes(text.Replace($"\r\n--{boundary}\r\n", $"\r\n--{boundary}XY\r\n", StringComparison.Ordinal));
            }
            using var stream = new ChoppyStream(bytes, 31);
            Exception ex = await Assert.ThrowsAnyAsync<Exception>(() => MultipartFormReader.ReadAsync(
                stream, multipart.Headers.ContentType!.ToString(), CancellationToken.None,
                failure == "limit" ? 500 : 10000, failure == "count" ? 1 : 32, spool));
            if (failure == "limit")
                Assert.Equal(413, Assert.IsType<TensorSharp.Server.Hosting.UploadLimitExceededException>(ex).StatusCode);
            else
                Assert.IsType<InvalidDataException>(ex);
            Assert.Empty(Directory.GetFiles(spool));
        }
        finally
        {
            Directory.Delete(spool, recursive: true);
        }
    }

    /// <summary>Bytes that are not compressible into a pattern the search could shortcut.</summary>
    private static byte[] Photo(int length)
    {
        byte[] bytes = new byte[length];
        new Random(20260903).NextBytes(bytes);
        // A JPEG magic number at the front, so the fixture is at least plausible as the
        // thing it stands in for.
        bytes[0] = 0xFF; bytes[1] = 0xD8; bytes[2] = 0xFF;
        return bytes;
    }

    private static byte[] Body(string fileName, string contentType, byte[] content, (string Name, string Value)? field = null)
    {
        var body = new MemoryStream();
        void Write(string text) => body.Write(Encoding.ASCII.GetBytes(text));

        if (field is { } f)
        {
            Write($"--{Boundary}\r\nContent-Disposition: form-data; name=\"{f.Name}\"\r\n\r\n{f.Value}\r\n");
        }
        Write($"--{Boundary}\r\n");
        Write($"Content-Disposition: form-data; name=\"file\"; filename=\"{fileName}\"\r\n");
        Write($"Content-Type: {contentType}\r\n\r\n");
        body.Write(content);
        Write($"\r\n--{Boundary}--\r\n");
        return body.ToArray();
    }

    /// <summary>A stream that hands back at most <paramref name="most"/> bytes per read.</summary>
    private sealed class ChoppyStream(byte[] bytes, int most) : Stream
    {
        private int _at;

        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(Math.Min(count, most), bytes.Length - _at);
            Array.Copy(bytes, _at, buffer, offset, n);
            _at += n;
            return n;
        }

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
        {
            int n = Math.Min(Math.Min(buffer.Length, most), bytes.Length - _at);
            bytes.AsSpan(_at, n).CopyTo(buffer.Span);
            _at += n;
            return ValueTask.FromResult(n);
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => bytes.Length;
        public override long Position { get => _at; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
