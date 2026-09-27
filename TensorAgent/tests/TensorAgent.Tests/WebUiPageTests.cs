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
using System.Text.Json;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Tests;

/// <summary>
/// The page, RUN.
///
/// <para>
/// Everything else that guards <c>tensoragent.js</c> reads it as text — does it
/// mention this route, does it parse — and none of that can answer the question
/// that mattered: when a chat with a photo in it is closed and opened again, is the
/// photo there. It was not, for two reasons a source-reading test cannot see. The
/// page sent the model its attachments and told the transcript nothing about them,
/// so nothing was saved; and its own history was flattened to role and text on the
/// way out, so even within one session the second question about a picture was asked
/// with the picture removed.
/// </para>
/// <para>
/// So these load the real script into JavaScriptCore — the same engine family the
/// WebView runs — on top of <c>PageDom.js</c>, a DOM the size of what the script
/// touches, and then do what a person does: attach something, send it, reopen the
/// chat. The assertions are on the request the page made and on the elements it
/// built, because those two are the whole contract between the page, the host and
/// the user.
/// </para>
/// </summary>
[Collection(LivePythonCollection.Name)]
public sealed class WebUiPageTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-page-" + Guid.NewGuid().ToString("N"));
    private readonly JavaScriptCoreEngine _engine = new();

    public WebUiPageTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { }
    }

    // =====================================================================================
    // running the page
    // =====================================================================================

    private static readonly string Repo = FindRepoRoot();

    private static string PageScript =>
        File.ReadAllText(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Core", "WebUi", "tensoragent.js"));

    private static string Dom =>
        File.ReadAllText(Path.Combine(Repo, "TensorAgent", "tests", "TensorAgent.Tests", "PageDom.js"));

    /// <summary>
    /// Load the DOM, the routes a launch needs, the page itself, and then
    /// <paramref name="drive"/> — which does the user's part and prints one JSON
    /// object with whatever it wants asserted on.
    /// </summary>
    private JsonElement Run(string routes, string drive)
    {
        string source = Dom + "\n" + Boot(routes) + "\n" + PageScript + "\n" + Settle(drive);
        var policy = new ExecutionPolicy(
            AllowScripts: true,
            AllowNetwork: false,
            WorkRoot: _root,
            ReadableRoots: Array.Empty<string>(),
            TempRoot: _root)
        {
            DefaultTimeout = TimeSpan.FromSeconds(30),
        };
        var context = new InterpreterContext(_root, new Dictionary<string, string> { ["HOME"] = _root }, policy);

        ExecutionResult result = _engine
            .RunCodeAsync(source, Array.Empty<string>(), context, CancellationToken.None)
            .GetAwaiter().GetResult();

        Assert.True(result.ExitCode == 0,
            $"the page script failed to run.{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");

        int start = result.Stdout.IndexOf("<<RESULT>>", StringComparison.Ordinal);
        Assert.True(start >= 0, "the driver printed no result: " + result.Stdout + result.Stderr);
        return JsonSerializer.Deserialize<JsonElement>(
            result.Stdout[(start + "<<RESULT>>".Length)..].Trim());
    }

    /// <summary>The answers a launch needs before anything a test does can happen.</summary>
    private static string Boot(string routes) => """
        var R = __page.routes;
        R['/api/agent/engine'] = { model: { id: 'gemma', name: 'Gemma', state: 'ready' }, networkDisabledMessage: 'network access is disabled by the user' };
        R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: [], skillsEnabled: true, maxTokens: 2048 };
        R['/api/models'] = { loaded: 'gemma.gguf', architecture: 'gemma3', backend: 'ggml_metal', visionReady: true };
        R['/api/skills'] = { enabled: true, installable: true, skills: [{ name: 'documents', description: 'make documents' }] };
        R['/api/agent/conversations'] = { conversations: [] };
        // Not a launch: the default is a page coming back inside an app that is
        // already running, which is the case that resumes the last chat. The tests
        // below that hand the page a saved conversation are testing THAT path, and a
        // cold launch would open an empty chat instead and assert nothing.
        R['/api/agent/launch'] = { cold: false };
        R['/api/agent/events'] = {};
        R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
        """ + "\n" + routes;

    /// <summary>
    /// Let the page finish booting, then drive it, then print. Everything the page
    /// does is promise-chained off its own boot, so the driver waits by giving the
    /// event loop a few turns rather than by reaching into the page's internals.
    /// </summary>
    /// <remarks>
    /// Two waits are offered to a driver. <c>settle(n)</c> gives the event loop
    /// <c>n</c> turns of one millisecond each, which is how a promise chain is let
    /// run to its end. <c>wait(ms)</c> is a single real <c>setTimeout</c> of that
    /// many milliseconds, for a page timer a test must outlast — a retry the page
    /// schedules 800 ms out is waited for with <c>wait(900)</c>, not with
    /// <c>settle(900)</c>, which would take as long and say nothing about order.
    /// The engine's timers are wall-clock, so both waits are real time; a run has
    /// thirty seconds in total.
    /// </remarks>
    private static string Settle(string drive) => """
        function settle(times) {
          return new Promise(function (done) {
            var left = times;
            (function tick() {
              if (left-- <= 0) { done(); return; }
              setTimeout(tick, 1);
            })();
          });
        }
        function wait(ms) {
          return new Promise(function (done) { setTimeout(done, ms); });
        }
        settle(40)
          .then(function () { return (function () {
        """ + drive + """
          })(); })
          .then(function (value) { return settle(40).then(function () { return value; }); })
          .then(function (value) { console.log('<<RESULT>>' + JSON.stringify(value)); })
          .catch(function (e) { console.log('<<RESULT>>' + JSON.stringify({ error: String((e && e.stack) || e) })); });
        """;

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException($"no TensorAgent above {AppContext.BaseDirectory}");
    }

    [Theory]
    [InlineData(131072, 8192, "128K model context", "8K active")]
    [InlineData(262144, 16384, "256K model context", "16K active")]
    [InlineData(262144, 32768, "256K model context", "32K active")]
    public void HeaderShowsTheModelsOwnContextSeparatelyFromTheActiveLimit(
        int modelContext, int activeContext, string expectedModel, string expectedActive)
    {
        JsonElement result = Run($$"""
            R['/api/models'] = { loaded: 'model.gguf', architecture: 'qwen35',
                loadedBackend: 'ggml_metal', visionReady: true,
                contextTokens: {{activeContext}}, modelContextTokens: {{modelContext}} };
            """, """
            var detail = __page.byId['model'].querySelector('.sub');
            return { detail: detail ? detail.textContent : '' };
            """);

        string detail = result.GetProperty("detail").GetString()!;
        Assert.Contains(expectedModel, detail, StringComparison.Ordinal);
        Assert.Contains(expectedActive, detail, StringComparison.Ordinal);
    }

    [Fact]
    public void HeaderKeepsCompatibilityWhenTheHostReportsOnlyOneContext()
    {
        JsonElement result = Run("""
            R['/api/models'] = { loaded: 'model.gguf', architecture: 'gemma4',
                loadedBackend: 'ggml_metal', visionReady: true, contextTokens: 8192 };
            """, """
            var detail = __page.byId['model'].querySelector('.sub');
            return { detail: detail ? detail.textContent : '' };
            """);

        string detail = result.GetProperty("detail").GetString()!;
        Assert.Contains("8K context", detail, StringComparison.Ordinal);
        Assert.DoesNotContain("active", detail, StringComparison.Ordinal);
    }

    [Fact]
    public void HeaderDoesNotPresentAnEffectiveFallbackAsModelMetadata()
    {
        JsonElement result = Run("""
            R['/api/models'] = { loaded: 'model.gguf', architecture: 'unknown',
                loadedBackend: 'ggml_cpu', visionReady: false,
                contextTokens: 4096, modelContextTokens: 0 };
            """, """
            var detail = __page.byId['model'].querySelector('.sub');
            return { detail: detail ? detail.textContent : '' };
            """);

        string detail = result.GetProperty("detail").GetString()!;
        Assert.Contains("4K active context", detail, StringComparison.Ordinal);
        Assert.DoesNotContain("model context", detail, StringComparison.Ordinal);
    }

    // =====================================================================================
    // what a message with attachments carries
    // =====================================================================================

    /// <summary>
    /// A photo, a recording, a clip and a document, attached and sent.
    ///
    /// <para>
    /// Six lists, and the request needs all of them to be right at once: the vision
    /// encoder reads <c>imagePaths</c> (a clip's frames go in there too), the
    /// transcript and the interpreter read <c>attachments</c>, and a document's text
    /// has to be in the content or the model is answering about a file it never saw.
    /// The one that was missing entirely is <c>attachments</c> — which is why nothing
    /// came back — and the one that was WRONG is <c>filePaths</c>, a name no parser on
    /// the other side has ever read, so every PDF and every clip was silently dropped.
    /// </para>
    /// </summary>
    [Fact]
    public void EveryKindOfAttachmentReachesTheRequestUnderTheNameTheServerReads()
    {
        JsonElement result = Run(string.Empty, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'cat.png', mediaType: 'image', url: '/uploads/a1.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'a2.wav', fileName: 'note.wav', mediaType: 'audio', url: '/uploads/a2.wav' });
            window.TensorAgent.addAttachment({ ok: true, file: 'a3.mp4', fileName: 'clip.mp4', mediaType: 'video', url: '/uploads/a3.mp4',
                                               frames: ['a3_0001.png', 'a3_0002.png'] });
            window.TensorAgent.addAttachment({ ok: true, file: 'a4.md', fileName: 'notes.md', mediaType: 'text', url: '/uploads/a4.md',
                                               textContent: 'SECRET-MARKER' });
            __page.byId['text'].value = 'What is in these?';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement sent = result.GetProperty("sent");
        Assert.Equal(1, sent.GetArrayLength());
        JsonElement message = sent[0].GetProperty("messages")[0];

        Assert.Equal(new[] { "a1.png", "a3_0001.png", "a3_0002.png" }, Strings(message, "imagePaths"));
        Assert.Equal(new[] { "a1.png" }, Strings(message, "stillImagePaths"));
        Assert.Equal(new[] { "a3.mp4" }, Strings(message, "videoFilePaths"));
        Assert.Equal(new[] { "a2.wav" }, Strings(message, "audioPaths"));
        Assert.Equal(new[] { "a4.md" }, Strings(message, "textFilePaths"));
        Assert.Equal(new[] { "notes.md" }, Strings(message, "textFileNames"));
        Assert.True(message.GetProperty("isVideo").GetBoolean());

        // The document's own text, in the message. Without it the model is answering
        // about a file whose name is all it was ever given.
        string content = message.GetProperty("content").GetString()!;
        Assert.Contains("SECRET-MARKER", content, StringComparison.Ordinal);
        Assert.Contains("[File: notes.md]", content, StringComparison.Ordinal);
        Assert.EndsWith("What is in these?", content, StringComparison.Ordinal);

        // And the chips, which is what a reopened chat is rebuilt from.
        JsonElement attachments = message.GetProperty("attachments");
        Assert.Equal(4, attachments.GetArrayLength());
        Assert.Equal("cat.png", attachments[0].GetProperty("fileName").GetString());
        Assert.Equal("image", attachments[0].GetProperty("mediaType").GetString());
        Assert.Equal(new[] { "a3_0001.png", "a3_0002.png" }, Strings(attachments[2], "frames"));

        // The name that was never read by anything, and must not come back.
        Assert.False(message.TryGetProperty("filePaths", out _),
            "filePaths is not a field any parser on the server reads; anything sent under it is dropped");
    }

    [Fact]
    public void SelectedFilesUploadTogetherInOrderAndAllReachTheChatRequest()
    {
        JsonElement result = Run("""
            R['/api/upload'] = { ok: true, files: [
              { ok: true, file: 'first.png', fileName: 'photo.png', mediaType: 'image', url: '/uploads/first.png' },
              { ok: true, file: 'second.txt', fileName: 'notes.txt', mediaType: 'text', textContent: 'SECOND-FILE-EVIDENCE' },
              { ok: true, file: 'third.csv', fileName: 'table.csv', mediaType: 'text', fileBacked: true }
            ] };
            """, """
            var input = __page.byId['file-input'];
            input.files = [{ name: 'photo.png' }, { name: 'notes.txt' }, { name: 'table.csv' }];
            input.value = 'selected';
            input.dispatch('change', { target: input });
            return settle(10).then(function () {
              var chips = __page.byId['chips'].querySelectorAll('.nm').map(function (n) { return n.textContent; });
              __page.byId['text'].value = 'Compare all three files';
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                return { uploads: __page.requests('/api/upload'), selected: input.value, chips: chips,
                         sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
              });
            });
            """);

        JsonElement upload = Assert.Single(result.GetProperty("uploads").EnumerateArray());
        Assert.Equal("POST", upload.GetProperty("method").GetString());
        JsonElement[] parts = upload.GetProperty("parts").EnumerateArray().ToArray();
        Assert.Equal(new[] { "photo.png", "notes.txt", "table.csv" },
            parts.Select(part => part.GetProperty("fileName").GetString()));
        Assert.All(parts, part => Assert.Equal("file", part.GetProperty("name").GetString()));
        Assert.Equal(string.Empty, result.GetProperty("selected").GetString());
        Assert.Equal(new[] { "photo.png", "notes.txt", "table.csv" }, Strings(result, "chips"));
        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        JsonElement message = sent.GetProperty("messages")[0];
        Assert.Equal(new[] { "first.png" }, Strings(message, "imagePaths"));
        Assert.Equal(new[] { "second.txt", "third.csv" }, Strings(message, "textFilePaths"));
        Assert.Equal(new[] { "notes.txt", "table.csv" }, Strings(message, "textFileNames"));
        Assert.Contains("SECOND-FILE-EVIDENCE", message.GetProperty("content").GetString(), StringComparison.Ordinal);
        Assert.Equal(new[] { "first.png", "second.txt", "third.csv" },
            message.GetProperty("attachments").EnumerateArray().Select(a => a.GetProperty("file").GetString()));
        Assert.True(message.GetProperty("attachments")[2].GetProperty("fileBacked").GetBoolean());
    }

    [Fact]
    public void SendingDuringQueuedUploadsKeepsTheDraftUntilEverySelectionIsAttached()
    {
        JsonElement result = Run("""
            var uploadGates = [], nextReply = 0;
            var replies = [
              { ok: true, files: [
                { ok: true, file: 'one.txt', fileName: 'one.txt', mediaType: 'text', textContent: 'ONE' },
                { ok: true, file: 'two.txt', fileName: 'two.txt', mediaType: 'text', textContent: 'TWO' }
              ] },
              { ok: true, file: 'three.txt', fileName: 'three.txt', mediaType: 'text', textContent: 'THREE' }
            ];
            R['/api/upload'] = function () { return replies[nextReply++]; };
            var originalFetch = fetch;
            fetch = function (url, init) {
              if (url !== '/api/upload') return originalFetch(url, init);
              return new Promise(function (resolve) {
                uploadGates.push(function () { originalFetch(url, init).then(resolve); });
              });
            };
            """, """
            var input = __page.byId['file-input'];
            input.files = [{ name: 'one.txt' }, { name: 'two.txt' }];
            input.dispatch('change', { target: input });
            input.files = [{ name: 'three.txt' }];
            input.dispatch('change', { target: input });
            __page.byId['text'].value = 'Use every selected file';
            __page.byId['send'].dispatch('click');
            var before = { sent: __page.requests('/api/chat').length, draft: __page.byId['text'].value };
            return settle(10).then(function () {
              before.started = uploadGates.length;
              uploadGates[0]();
              return settle(10);
            }).then(function () {
              __page.byId['send'].dispatch('click');
              var between = { sent: __page.requests('/api/chat').length, count: window.TensorAgent.attachmentCount() };
              uploadGates[1]();
              return settle(10).then(function () {
                __page.byId['send'].dispatch('click');
                return settle(10).then(function () {
                  return { before: before, between: between, notices: __page.notices(),
                           sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
                });
              });
            });
            """);

        Assert.Equal(0, result.GetProperty("before").GetProperty("sent").GetInt32());
        Assert.Equal("Use every selected file", result.GetProperty("before").GetProperty("draft").GetString());
        Assert.Equal(1, result.GetProperty("before").GetProperty("started").GetInt32());
        Assert.Equal(0, result.GetProperty("between").GetProperty("sent").GetInt32());
        Assert.Equal(2, result.GetProperty("between").GetProperty("count").GetInt32());
        Assert.Contains(Strings(result, "notices"), text => text.Contains("wait for file uploads", StringComparison.Ordinal));
        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "one.txt", "two.txt", "three.txt" }, Strings(sent.GetProperty("messages")[0], "textFilePaths"));
    }

    [Fact]
    public void OneSelectedFileAcceptsTheExistingUploadResponse()
    {
        JsonElement result = Run("""
            R['/api/upload'] = { ok: true, file: 'one.txt', fileName: 'notes.txt', mediaType: 'text',
                                 textContent: 'evidence', warning: 'Read the warning' };
            """, """
            var input = __page.byId['file-input'];
            input.files = [{ name: 'notes.txt' }];
            input.dispatch('change', { target: input });
            return settle(10).then(function () {
              return { uploads: __page.requests('/api/upload'), count: window.TensorAgent.attachmentCount(),
                       notices: __page.notices(), errors: __page.errorNotices() };
            });
            """);

        Assert.Single(result.GetProperty("uploads").EnumerateArray());
        Assert.Equal(1, result.GetProperty("count").GetInt32());
        Assert.Contains("Read the warning", Strings(result, "notices"));
        Assert.Empty(Strings(result, "errors"));
    }

    [Theory]
    [InlineData("{ __status: 413, body: { error: 'File too large' } }", "File too large")]
    [InlineData("{ __reject: 'Load failed' }", "Load failed")]
    [InlineData("{ ok: true, file: 'only.txt' }", "did not return every uploaded file")]
    [InlineData("{ ok: true, files: [{ ok: true, file: 'one.txt' }, { ok: false, error: 'bad file' }] }", "did not return every uploaded file")]
    public void FailedOrIncompleteBatchNeverAttachesAPartialSelection(string reply, string expectedError)
    {
        JsonElement result = Run("R['/api/upload'] = " + reply + ";", """
            window.TensorAgent.addAttachment({ ok: true, file: 'existing.txt', fileName: 'existing.txt', mediaType: 'text' });
            var input = __page.byId['file-input'];
            input.files = [{ name: 'one.txt' }, { name: 'two.txt' }];
            input.dispatch('change', { target: input });
            return settle(10).then(function () {
              return { uploads: __page.requests('/api/upload').length, count: window.TensorAgent.attachmentCount(),
                       chips: __page.byId['chips'].querySelectorAll('.nm').map(function (n) { return n.textContent; }),
                       errors: __page.errorNotices() };
            });
            """);

        Assert.Equal(1, result.GetProperty("uploads").GetInt32());
        Assert.Equal(1, result.GetProperty("count").GetInt32());
        Assert.Equal(new[] { "existing.txt" }, Strings(result, "chips"));
        Assert.Contains(expectedError, Assert.Single(Strings(result, "errors")), StringComparison.Ordinal);
    }

    [Fact]
    public void AnImageStaysInTheComposerWhenTheLoadedModelHasNoVisionProjector()
    {
        JsonElement result = Run("""
            var modelReads = 0;
            R['/api/models'] = function () {
              modelReads++;
              return modelReads === 1
                ? { loaded: 'Qwen3.5-9B-IQ4_XS.gguf', architecture: 'qwen35',
                    loadedBackend: 'ggml_metal', visionReady: true }
                : { loaded: 'Qwen3.5-9B-IQ4_XS.gguf', architecture: 'qwen35',
                    loadedBackend: 'ggml_metal', loadedMmProj: 'stale-or-wrong.gguf', visionReady: false };
            };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'chart.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'What does this chart show?';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              var action = __page.byId['chat'].querySelector('.notice-action');
              if (action) action.dispatch('click');
              return {
                sent: __page.requests('/api/chat').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history(),
                notice: __page.byId['chat'].textContent,
                routes: __page.requests('/api/agent/events').map(function (c) { return c.body; })
              };
            });
            """);

        Assert.Equal(0, result.GetProperty("sent").GetInt32());
        Assert.Equal("What does this chart show?", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Equal(0, result.GetProperty("history").GetArrayLength());
        Assert.Contains(result.GetProperty("routes").EnumerateArray(), route =>
            route.TryGetProperty("type", out JsonElement type) && type.GetString() == "open-route" &&
            route.TryGetProperty("route", out JsonElement name) && name.GetString() == "models");
    }

    [Fact]
    public void ATextOnlyModelCanHandAnImageFileToASelectedHostSkill()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: ['documents'],
                                         skillsEnabled: true, maxTokens: 2048 };
            R['/api/models'] = { loaded: 'text-only.gguf', architecture: 'llama',
                                 loadedBackend: 'ggml_metal', visionReady: false,
                                 acceptsVisionProjector: false };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'photo.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'Put this photo into a PDF.';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "documents" }, Strings(request, "skills"));
        Assert.Equal(new[] { "a1.png" }, Strings(request.GetProperty("messages")[0], "imagePaths"));
    }

    [Fact]
    public void AServerVisionRefusalRestoresTheExactImageDraft()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: ['summarize'],
                                         skillsEnabled: true, maxTokens: 2048 };
            R['/api/models'] = { loaded: 'text-only.gguf', architecture: 'llama',
                                 loadedBackend: 'ggml_metal', visionReady: false,
                                 acceptsVisionProjector: false };
            R['/api/chat'] = { __status: 400,
                body: { code: 'vision_not_ready', error: 'This skill cannot read the image file.' } };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'photo.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'Summarize this image.';
            __page.byId['send'].dispatch('click');
            return settle(15).then(function () {
              return {
                sent: __page.requests('/api/chat').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history()
              };
            });
            """);

        Assert.Equal(1, result.GetProperty("sent").GetInt32());
        Assert.Equal("Summarize this image.", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
    }

    [Fact]
    public void AnImageDraftIsNotSentWhenTheModelUnloadsDuringCapabilityRefresh()
    {
        JsonElement result = Run("""
            var modelReads = 0;
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: ['documents'],
                                         skillsEnabled: true, maxTokens: 2048 };
            R['/api/models'] = function () {
              modelReads++;
              return modelReads === 1
                ? { loaded: 'text-only.gguf', architecture: 'llama', visionReady: false,
                    acceptsVisionProjector: false }
                : { loaded: null, architecture: null, visionReady: false,
                    acceptsVisionProjector: false };
            };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'photo.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'Put this into a PDF.';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return {
                sent: __page.requests('/api/chat').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history()
              };
            });
            """);

        Assert.Equal(0, result.GetProperty("sent").GetInt32());
        Assert.Equal("Put this into a PDF.", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
    }

    [Fact]
    public void AFileBackedCsvKeepsItsPathAndChipWithoutPuttingRowsInThePrompt()
    {
        JsonElement result = Run(string.Empty, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a5.csv', fileName: 'responses.csv',
                                               mediaType: 'text', url: '/uploads/a5.csv', fileBacked: true });
            __page.byId['text'].value = 'Please analyze this form.';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            """);

        JsonElement message = result.GetProperty("sent")[0].GetProperty("messages")[0];
        Assert.Equal("Please analyze this form.", message.GetProperty("content").GetString());
        Assert.DoesNotContain("[File:", message.GetProperty("content").GetString(), StringComparison.Ordinal);
        Assert.Equal(new[] { "a5.csv" }, Strings(message, "textFilePaths"));
        Assert.Equal(new[] { "responses.csv" }, Strings(message, "textFileNames"));

        JsonElement attachment = Assert.Single(message.GetProperty("attachments").EnumerateArray());
        Assert.Equal("a5.csv", attachment.GetProperty("file").GetString());
        Assert.Equal("responses.csv", attachment.GetProperty("fileName").GetString());
        Assert.Equal("text", attachment.GetProperty("mediaType").GetString());
        Assert.True(attachment.GetProperty("fileBacked").GetBoolean());
        Assert.False(attachment.TryGetProperty("textContent", out _));
    }

    /// <summary>
    /// The second question about a picture is asked with the picture.
    ///
    /// <para>
    /// The history was mapped to <c>{role, content}</c> on its way into the body, so
    /// turn two of "what is in this photo / and what colour is the car" reached the
    /// model with no photo attached to turn one. It answers anyway, which is what made
    /// this expensive to notice.
    /// </para>
    /// </summary>
    [Fact]
    public void AFollowUpQuestionStillCarriesTheEarlierTurnsPicture()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [{ token: 'A cat.' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'cat.png', mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'What is this?';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              __page.byId['text'].value = 'And what colour?';
              __page.byId['send'].dispatch('click');
              return settle(20);
            }).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement sent = result.GetProperty("sent");
        Assert.Equal(2, sent.GetArrayLength());
        JsonElement messages = sent[1].GetProperty("messages");
        Assert.Equal(3, messages.GetArrayLength());
        Assert.Equal(new[] { "a1.png" }, Strings(messages[0], "imagePaths"));
        Assert.Equal("A cat.", messages[1].GetProperty("content").GetString());
        Assert.Equal("And what colour?", messages[2].GetProperty("content").GetString());
    }

    /// <summary>
    /// A file a turn produced is a link in the transcript AND a line in the history,
    /// so the next message does not delete it from the saved chat.
    /// </summary>
    [Fact]
    public void AFileATurnProducedBecomesALinkAndSurvivesInTheHistory()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [
              { skill_step: 'shell', skill: 'documents', detail: 'make_pdf.py', ok: true,
                files: [{ name: 'photo.pdf', bytes: 40960, url: '/api/code/artifacts/r/photo.pdf' }] },
              { tool_progress: 'finished', tool: 'shell', seconds: 3 },
              { token: 'Done.' },
              { done: true, truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'make a pdf';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { transcript: __page.transcript(), history: window.TensorAgent.history() };
            });
            """);

        JsonElement turns = result.GetProperty("transcript");
        JsonElement answer = turns[turns.GetArrayLength() - 1];
        Assert.Contains(answer.GetProperty("media").EnumerateArray(),
            m => m.GetProperty("src").GetString() == "/api/code/artifacts/r/photo.pdf"
                 && m.GetProperty("text").GetString()!.Contains("photo.pdf", StringComparison.Ordinal));

        JsonElement history = result.GetProperty("history");
        JsonElement assistant = history[history.GetArrayLength() - 1];
        Assert.Equal("photo.pdf", assistant.GetProperty("artifacts")[0].GetProperty("name").GetString());
    }

    [Fact]
    public void AGuardedArtifactAppearsOnlyWhenItsDedicatedVerifiedFrameArrives()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [
              { skill_step: 'skills_run', skill: 'documents', detail: 'scripts/make_pptx.py',
                ok: true, files: null },
              { tool_progress: 'finished', tool: 'skills_run', seconds: 3 },
              { artifact_verified: true,
                files: [{ name: 'apple-report.pptx', bytes: 15569,
                          url: '/api/code/artifacts/verified/apple-report.pptx' }] },
              { token: 'Done.' },
              { done: true, truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'make the Apple report';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { transcript: __page.transcript(), history: window.TensorAgent.history() };
            });
            """);

        JsonElement turns = result.GetProperty("transcript");
        JsonElement answer = turns[turns.GetArrayLength() - 1];
        Assert.Contains(answer.GetProperty("media").EnumerateArray(),
            item => item.GetProperty("src").GetString()
                == "/api/code/artifacts/verified/apple-report.pptx");

        JsonElement history = result.GetProperty("history");
        JsonElement assistant = history[history.GetArrayLength() - 1];
        JsonElement artifact = Assert.Single(assistant.GetProperty("artifacts").EnumerateArray());
        Assert.Equal("apple-report.pptx", artifact.GetProperty("name").GetString());
        Assert.Equal(15569, artifact.GetProperty("bytes").GetInt32());
    }

    // =====================================================================================
    // reopening a saved chat
    // =====================================================================================

    /// <summary>
    /// The whole bug, from the user's side: a saved chat with media in it, reopened.
    ///
    /// <para>
    /// The transcript here is what the host really stores — the same property names,
    /// because <c>StoredMessage</c> is the page's own message shape — and what is
    /// asserted is that each kind of attachment comes back as the thing it is: a
    /// picture as a picture, a recording as something playable, a clip as a clip, a
    /// document as a link that opens it. A row of paperclip labels would satisfy a
    /// weaker test and would not be the file coming back.
    /// </para>
    /// </summary>
    /// <summary>
    /// A file the app uploads reaches the composer, however many lines it has.
    ///
    /// <para>
    /// The regression this exists for: the app used to build
    /// <c>window.TensorAgent.addAttachment({…json…})</c> as a JavaScript SOURCE string
    /// and hand it to MAUI, which wraps every script as
    /// <c>try{JSON.stringify(eval('&lt;script&gt;'))}catch(e){'null'};</c>. The script
    /// therefore became the body of a single-quoted literal, and JSON's <c>\n</c> was
    /// read as that literal's escape — so a text file with two lines produced a real
    /// newline inside an unterminated string, eval threw, MAUI's own catch returned the
    /// STRING "null", and the app discarded it. Upload succeeded, nothing attached,
    /// nothing said. An apostrophe in a file name closed the literal and did the same.
    /// </para>
    /// <para>
    /// So the payload here carries every character that used to break it: newlines, a
    /// double quote, an apostrophe, a backslash and a non-ASCII character — and it
    /// arrives base64'd, which is the fix.
    /// </para>
    /// </summary>
    [Fact]
    public void AMultiLineFileTheAppUploadsReachesTheComposerIntact()
    {
        string json = JsonSerializer.Serialize(new
        {
            ok = true,
            file = "a1.txt",
            fileName = "Bob's \"notes\".txt",
            mediaType = "text",
            url = "/uploads/a1.txt",
            textContent = "line one\nline two\ttabbed\n\"quoted\" and O'Brien \\ backslash\nnaïve café\n",
        });
        string payload = Convert.ToBase64String(Encoding.UTF8.GetBytes(json));

        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [{ token: 'Read it.' }, { done: true, truncated: false }] };
            """, $$"""
            var answer = window.TensorAgent.__fromHost('addAttachment', '{{payload}}');
            __page.byId['text'].value = 'What is in it?';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return {
                answer: answer,
                sent: __page.requests('/api/chat').map(function (c) { return c.body; })
              };
            });
            """);

        Assert.Equal("ok", result.GetProperty("answer").GetString());

        JsonElement body = result.GetProperty("sent")[0];
        JsonElement messages = body.GetProperty("messages");
        string content = messages[messages.GetArrayLength() - 1].GetProperty("content").GetString()!;

        // The file arrived whole: the envelope, every line, and the characters that used
        // to close the literal early.
        Assert.Contains("[File: Bob's \"notes\".txt]", content, StringComparison.Ordinal);
        Assert.Contains("line one\nline two", content, StringComparison.Ordinal);
        Assert.Contains("O'Brien \\ backslash", content, StringComparison.Ordinal);
        Assert.Contains("naïve café", content, StringComparison.Ordinal);
        Assert.Contains("[End of file]", content, StringComparison.Ordinal);
    }

    // =====================================================================================
    // content shared into the app
    // =====================================================================================

    [Fact]
    public void AShareWaitsForLaunchThenPreservesTheDraftAndAppliesEveryValidPart()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'Keep this draft  ';
            R['/api/agent/share/claim'] = function () {
              return { share: {
                id: 'share-1',
                text: 'What can you tell me about this?\n\nShared text:\n\u0022\u0022\u0022\nこんにちは 🌍\nsecond line\n\u0022\u0022\u0022',
                attachments: [
                  { ok: true, file: 'shared.png', fileName: '旅行.png', mediaType: 'image', url: '/uploads/shared.png' },
                  { ok: true, file: 'shared.txt', fileName: 'notes.txt', mediaType: 'text', url: '/uploads/shared.txt', textContent: 'café\nline two' },
                  { ok: false, error: 'broken.mov was not attached.' }
                ],
                notices: ['Only part of the page text fit.'],
                title: 'Travel note',
                newChat: true
              } };
            };
            """, """
            return {
              text: __page.byId['text'].value,
              attachments: window.TensorAgent.attachmentCount(),
              chips: __page.byId['chips'].querySelectorAll('.chip').map(function (c) { return c.textContent; }),
              notices: __page.byId['chat'].textContent,
              sent: __page.requests('/api/chat').length,
              calls: __page.calls.map(function (c) {
                return { path: c.path, method: c.method, type: c.body && c.body.type };
              })
            };
            """);

        const string shared = "What can you tell me about this?\n\nShared text:\n\"\"\"\nこんにちは 🌍\nsecond line\n\"\"\"";
        Assert.Equal("Keep this draft " + shared, result.GetProperty("text").GetString());
        Assert.Equal(2, result.GetProperty("attachments").GetInt32());
        Assert.Equal(new[] { "↗Travel note✕", "旅行.png✕", "📄notes.txt✕" },
            result.GetProperty("chips").EnumerateArray().Select(c => c.GetString()).ToArray());
        Assert.Contains("Only part of the page text fit.", result.GetProperty("notices").GetString(), StringComparison.Ordinal);
        Assert.Contains("broken.mov was not attached.", result.GetProperty("notices").GetString(), StringComparison.Ordinal);
        Assert.Equal(0, result.GetProperty("sent").GetInt32());

        var calls = result.GetProperty("calls").EnumerateArray().ToList();
        int launchSession = calls.FindIndex(c => c.GetProperty("path").GetString() == "/api/sessions");
        int ready = calls.FindIndex(c =>
            c.GetProperty("path").GetString() == "/api/agent/events" &&
            c.TryGetProperty("type", out JsonElement type) && type.GetString() == "ready");
        int claim = calls.FindIndex(c => c.GetProperty("path").GetString() == "/api/agent/share/claim");
        int shareSession = calls.FindIndex(claim + 1, c => c.GetProperty("path").GetString() == "/api/sessions");
        Assert.True(launchSession >= 0 && launchSession < ready && ready < claim && claim < shareSession,
            "startup must open its launch chat, announce ready, claim, then await the share's new chat");
        Assert.Equal("POST", calls[claim].GetProperty("method").GetString());
        Assert.DoesNotContain(calls, c => c.GetProperty("path").GetString() == "/api/agent/share/ack");
    }

    [Fact]
    public void RepeatedClaimsAndConcurrentNudgesDoNotDuplicateAnAppliedShare()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'base';
            R['/api/agent/share/claim'] = function () {
              return { share: {
                id: 'share-retry', text: 'shared once', newChat: false,
                attachments: [{ ok: true, file: 'one.pdf', fileName: 'one.pdf', mediaType: 'text', url: '/uploads/one.pdf' }]
              } };
            };
            """, """
            var claimsBefore = __page.requests('/api/agent/share/claim').length;
            var hostAnswer = window.TensorAgent.__fromHost('takeShare', 'e30=');
            document.dispatch('visibilitychange', {});
            var immediateClaims = __page.requests('/api/agent/share/claim').length - claimsBefore;
            return settle(20).then(function () {
              return {
                hostAnswer: hostAnswer,
                immediateClaims: immediateClaims,
                claims: __page.requests('/api/agent/share/claim').length - claimsBefore,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                sharedChips: __page.byId['chips'].querySelectorAll('.shared').length,
                sent: __page.requests('/api/chat').length
              };
            });
            """);

        Assert.Equal("ok", result.GetProperty("hostAnswer").GetString());
        Assert.Equal(1, result.GetProperty("immediateClaims").GetInt32());
        Assert.InRange(result.GetProperty("claims").GetInt32(), 1, 2);
        Assert.Equal("base shared once", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Equal(1, result.GetProperty("sharedChips").GetInt32());
        Assert.Equal(0, result.GetProperty("sent").GetInt32());
    }

    [Fact]
    public void AShareIsNotAppliedUntilItsRequestedNewChatOpensSuccessfully()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'draft';
            var opens = 0;
            R['/api/sessions?conversation=new'] = function () {
              opens++;
              if (opens === 2) return { __status: 500, body: { error: 'not yet' } };
              return { sessionId: 's' + opens, conversationId: 'c' + opens, messages: [], think: false, skills: [] };
            };
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-new-chat', text: 'the shared text', attachments: [], newChat: true } };
            };
            """, """
            var before = {
              text: __page.byId['text'].value,
              sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
            };
            window.TensorAgent.takeShare();
            return settle(20).then(function () {
              return {
                before: before,
                after: {
                  text: __page.byId['text'].value,
                  attachments: window.TensorAgent.attachmentCount(),
                  sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
                }
              };
            });
            """);

        Assert.Equal("draft", result.GetProperty("before").GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("before").GetProperty("sharedChips").GetInt32());
        Assert.Equal("draft the shared text", result.GetProperty("after").GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("after").GetProperty("attachments").GetInt32());
        Assert.Equal(1, result.GetProperty("after").GetProperty("sharedChips").GetInt32());
    }

    [Fact]
    public void AShareNeverAutoSendsEvenIfALegacyPayloadRequestsIt()
    {
        JsonElement result = Run("""
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-send', text: 'Send this shared text', attachments: [], newChat: false, autoSend: true } };
            };
            R['/api/chat'] = { __sse: [{ token: 'Done.' }, { done: true, truncated: false }] };
            """, """
            return {
              sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
              text: __page.byId['text'].value,
              sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
            };
            """);

        Assert.Empty(result.GetProperty("sent").EnumerateArray());
        Assert.Equal("Send this shared text", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("sharedChips").GetInt32());
    }

    [Fact]
    public void NavigationPreservesSharedTextAttachmentAndIdUntilTheyAreSent()
    {
        JsonElement result = Run("""
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-nav', text: 'shared across chats', title: 'Shared article', newChat: false,
                attachments: [{ ok: true, file: 'article.png', fileName: 'article.png', mediaType: 'image', url: '/uploads/article.png' }] } };
            };
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'user', content: 'an older question' }], think: false, skills: [] };
            R['/api/chat'] = { __sse: [{ token: 'Done.' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(10).then(function () {
              var preserved = {
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
              };
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                return {
                  preserved: preserved,
                  sent: __page.requests('/api/chat').map(function (c) { return c.body; })
                };
              });
            });
            """);

        JsonElement preserved = result.GetProperty("preserved");
        Assert.Equal("shared across chats", preserved.GetProperty("text").GetString());
        Assert.Equal(1, preserved.GetProperty("attachments").GetInt32());
        Assert.Equal(1, preserved.GetProperty("sharedChips").GetInt32());

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "share-nav" }, Strings(request, "shareIds"));
        JsonElement message = request.GetProperty("messages")[1];
        Assert.Equal(new[] { "article.png" }, Strings(message, "stillImagePaths"));
        Assert.Contains("shared across chats", message.GetProperty("content").GetString()!, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitDiscardRemovesOnlyTheTrackedSharedDraft()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'keep me';
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-discard', text: 'remove me', title: 'Disposable', newChat: false,
                attachments: [{ ok: true, file: 'discard.png', fileName: 'discard.png', mediaType: 'image', url: '/uploads/discard.png' }] } };
            };
            R['/api/agent/share/discard'] = function () { return { ok: true }; };
            """, """
            __page.byId['text'].value += ' typed later';
            var shared = __page.byId['chips'].querySelector('.shared');
            shared.querySelector('button').dispatch('click');
            return settle(10).then(function () {
              return {
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                sharedChips: __page.byId['chips'].querySelectorAll('.shared').length,
                discards: __page.requests('/api/agent/share/discard').map(function (c) { return c.body.id; }),
                sent: __page.requests('/api/chat').length
              };
            });
            """);

        Assert.Equal("keep me typed later", result.GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("attachments").GetInt32());
        Assert.Equal(0, result.GetProperty("sharedChips").GetInt32());
        Assert.Equal(new[] { "share-discard" }, Strings(result, "discards"));
        Assert.Equal(0, result.GetProperty("sent").GetInt32());
    }

    [Fact]
    public void IndependentQueuedSharesUseDifferentFreshChatsAndNeverMerge()
    {
        JsonElement result = Run("""
            var offered = 0;
            var sessionOpens = 0;
            R['/api/sessions?conversation=new'] = function () {
              sessionOpens++;
              return { sessionId: 's' + sessionOpens, conversationId: 'c' + sessionOpens,
                messages: [], think: false, skills: [] };
            };
            var shares = [
              // Version-1 envelopes could opt out of a new chat. An envelope boundary
              // is authoritative now, so even these legacy values must stay separate.
              { id: 'share-a', text: 'first', title: 'First', newChat: false,
                attachments: [{ ok: true, file: 'a.txt', fileName: 'a.txt', mediaType: 'text',
                  textContent: 'first attachment', url: '/uploads/a.txt' }] },
              { id: 'share-b', text: 'second', title: 'Second', newChat: false,
                attachments: [{ ok: true, file: 'b.png', fileName: 'b.png', mediaType: 'image', url: '/uploads/b.png' }] }
            ];
            R['/api/agent/share/claim'] = function () { return { share: shares[offered] || null }; };
            R['/api/chat'] = function () {
              offered = 1;
              return { __sse: [{ token: 'Answer.' }, { done: true, truncated: false }] };
            };
            """, """
            var claimsBeforeSend = __page.requests('/api/agent/share/claim').length;
            __page.byId['send'].dispatch('click');
            var sent = __page.requests('/api/chat')[0].body;
            window.TensorAgent.takeShare();
            var whileGenerating = {
              generating: window.TensorAgent.isGenerating(),
              claims: __page.requests('/api/agent/share/claim').length,
              text: __page.byId['text'].value
            };
            return settle(20).then(function () {
              return {
                claimsBeforeSend: claimsBeforeSend,
                whileGenerating: whileGenerating,
                sent: sent,
                sessionOpens: sessionOpens,
                finalClaims: __page.requests('/api/agent/share/claim').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history(),
                chips: __page.byId['chips'].querySelectorAll('.shared').map(function (c) { return c.textContent; })
              };
            });
            """);

        JsonElement during = result.GetProperty("whileGenerating");
        Assert.True(during.GetProperty("generating").GetBoolean());
        Assert.Equal(result.GetProperty("claimsBeforeSend").GetInt32(), during.GetProperty("claims").GetInt32());
        Assert.Equal(string.Empty, during.GetProperty("text").GetString());
        Assert.Equal(new[] { "share-a" }, Strings(result.GetProperty("sent"), "shareIds"));
        Assert.Equal("s2", result.GetProperty("sent").GetProperty("sessionId").GetString());

        Assert.Equal(result.GetProperty("claimsBeforeSend").GetInt32() + 1,
            result.GetProperty("finalClaims").GetInt32());
        Assert.Equal(3, result.GetProperty("sessionOpens").GetInt32());
        Assert.Equal("second", result.GetProperty("text").GetString());
        Assert.DoesNotContain("first", result.GetProperty("text").GetString()!, StringComparison.Ordinal);
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.Equal(new[] { "↗Second✕" }, Strings(result, "chips"));
    }

    /// <summary>
    /// Coming back returns to the chat the page was in, not to the newest saved one.
    ///
    /// <para>
    /// The pairing that breaks the obvious implementation: <c>fresh</c> is the empty
    /// chat a launch just opened, and it is not in the conversation list at all — a
    /// chat with no messages is never listed. <c>yesterday</c> is. So a page that
    /// resumed "the most recent conversation" would take the user from the clean
    /// composer they were handed on launch into an old transcript, the first time
    /// WebKit reclaimed the content process behind a trip to the Models screen. Which
    /// is the thing this whole change exists to stop.
    /// </para>
    /// </summary>
    [Fact]
    public void ComingBackOpensTheChatThePageWasInAndNotTheNewestSavedOne()
    {
        JsonElement result = Run("""
            R['/api/agent/launch'] = { cold: false, conversation: 'fresh' };
            R['/api/agent/conversations'] = { conversations: [{ id: 'yesterday', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 4 }] };
            R['/api/sessions?conversation=fresh'] = {
              sessionId: 's1', conversationId: 'fresh', messages: [], think: false, skills: []
            };
            R['/api/sessions?conversation=yesterday'] = {
              sessionId: 's2', conversationId: 'yesterday', think: false, skills: [],
              messages: [{ role: 'user', content: 'yesterday' }, { role: 'assistant', content: 'indeed' }]
            };
            """, """
            return {
              opened: __page.requests('/api/sessions').map(function (c) { return c.url; }),
              shown: __page.transcript().map(function (t) { return t.text; }).join(' | ')
            };
            """);

        var opened = result.GetProperty("opened").EnumerateArray().Select(u => u.GetString()!).ToList();
        Assert.Contains("/api/sessions?conversation=fresh", opened);
        Assert.DoesNotContain("/api/sessions?conversation=yesterday", opened);
        Assert.DoesNotContain("indeed", result.GetProperty("shown").GetString()!, StringComparison.Ordinal);
    }

    /// <summary>
    /// Launching the app opens a clean chat, not yesterday's.
    ///
    /// <para>
    /// The page used to open the most recent conversation on every load, so the app
    /// always reopened mid-thought — reasonable on a desktop tab that stays open,
    /// wrong for something the user launches from a home screen. It now starts empty
    /// and leaves the previous chat one tap away in the menu.
    /// </para>
    /// <para>
    /// Only on a LAUNCH, which is the half this test exists to pin. The identical page
    /// load happens when WebKit kills the content process of a WebView whose view left
    /// the window, and that one must still come back to the chat the user was reading,
    /// possibly with an answer still being generated for it on the host side. The page
    /// cannot tell the two apart from inside, so it asks; every other test here runs
    /// with the <c>cold: false</c> default and exercises that path.
    /// </para>
    /// </summary>
    [Fact]
    public void LaunchingTheAppOpensAnEmptyChatRatherThanTheLastOne()
    {
        JsonElement result = Run("""
            R['/api/agent/launch'] = { cold: true };
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old chat', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [{ role: 'user', content: 'yesterday' }, { role: 'assistant', content: 'indeed' }]
            };
            """, """
            return {
              shown: __page.transcript().map(function (t) { return t.text; }).join(' | '),
              history: window.TensorAgent.history(),
              opened: __page.requests('/api/sessions').map(function (c) { return c.url; })
            };
            """);

        // The session that was opened is the whole behaviour: "new" is a fresh chat,
        // "saved" is yesterday's. Asserted on the request rather than on an empty
        // transcript, because a fresh chat is not empty on screen — it carries the
        // "New chat" placeholder, which is exactly what should be there.
        var opened = result.GetProperty("opened").EnumerateArray().Select(u => u.GetString()!).ToList();
        Assert.Contains("/api/sessions?conversation=new", opened);
        Assert.DoesNotContain("/api/sessions?conversation=saved", opened);

        // And nothing of yesterday's chat came with it — not on screen, and not in the
        // history the next message would be sent with, which is the half that would
        // quietly carry the old conversation into the new one's context.
        string shown = result.GetProperty("shown").GetString()!;
        Assert.DoesNotContain("yesterday", shown, StringComparison.Ordinal);
        Assert.DoesNotContain("indeed", shown, StringComparison.Ordinal);
        Assert.Equal(0, result.GetProperty("history").GetArrayLength());
    }

    [Fact]
    public void AReopenedChatShowsItsPicturesSoundsClipsAndDocumentsAgain()
    {
        JsonElement result = Run("""
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old chat', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user',
                  content: '[File: notes.md]\nBODY\n[End of file]\n\nLook at these',
                  imagePaths: ['a1.png', 'a3_0001.png'],
                  stillImagePaths: ['a1.png'],
                  videoFilePaths: ['a3.mp4'],
                  audioPaths: ['a2.wav'],
                  textFilePaths: ['a4.md'],
                  textFileNames: ['notes.md'],
                  isVideo: true,
                  attachments: [
                    { file: 'a1.png', fileName: 'cat.png', mediaType: 'image' },
                    { file: 'a2.wav', fileName: 'note.wav', mediaType: 'audio' },
                    { file: 'a3.mp4', fileName: 'clip.mp4', mediaType: 'video', frames: ['a3_0001.png'] },
                    { file: 'a4.md', fileName: 'notes.md', mediaType: 'text' },
                    { file: 'a5.heic', fileName: 'IMG_1.heic', mediaType: 'image', previewFile: 'a5-preview.png' }
                  ] },
                { role: 'assistant', content: 'Here it is.', thinking: 'hmm',
                  artifacts: [{ name: 'report.pdf', bytes: 2048, url: '/api/code/artifacts/r/report.pdf' }] }
              ]
            };
            """, """
            return { transcript: __page.transcript(), history: window.TensorAgent.history() };
            """);

        JsonElement turns = result.GetProperty("transcript");
        Assert.Equal(2, turns.GetArrayLength());

        JsonElement user = turns[0];
        var media = user.GetProperty("media").EnumerateArray()
            .Select(m => (Tag: m.GetProperty("tag").GetString()!, Src: m.GetProperty("src").GetString()!))
            .ToList();

        Assert.Contains(media, m => m.Tag == "IMG" && m.Src == "/uploads/a1.png");
        Assert.Contains(media, m => m.Tag == "AUDIO" && m.Src == "/uploads/a2.wav");
        Assert.Contains(media, m => m.Tag == "VIDEO" && m.Src == "/uploads/a3.mp4");
        Assert.Contains(media, m => m.Tag == "A" && m.Src == "/uploads/a4.md");
        // A HEIC is shown through the PNG the host wrote beside it: no browser
        // renders HEIC, so pointing at the original is a broken image with a 200.
        Assert.Contains(media, m => m.Tag == "IMG" && m.Src == "/uploads/a5-preview.png");

        // The user's own sentence, not the file they attached, in the bubble. The
        // page renders Markdown into innerHTML, so that is where the words are.
        string bubble = user.GetProperty("html").GetString()!;
        Assert.Contains("Look at these", bubble, StringComparison.Ordinal);
        Assert.DoesNotContain("End of file", bubble, StringComparison.Ordinal);

        // The file the turn produced is a link again, not a sentence about a link.
        Assert.Contains(turns[1].GetProperty("media").EnumerateArray(),
            m => m.GetProperty("src").GetString() == "/api/code/artifacts/r/report.pdf");

        // And the history the next request is built from kept every path, so a
        // follow-up question is asked with the same attachments the user can see.
        JsonElement history = result.GetProperty("history");
        Assert.Equal(2, history.GetArrayLength());
        Assert.Equal(new[] { "a1.png", "a3_0001.png" }, Strings(history[0], "imagePaths"));
        Assert.Equal(new[] { "a2.wav" }, Strings(history[0], "audioPaths"));
        Assert.Equal("hmm", history[1].GetProperty("thinking").GetString());
        Assert.Equal("report.pdf", history[1].GetProperty("artifacts")[0].GetProperty("name").GetString());
    }

    /// <summary>
    /// Reopening a chat and then saying something else does not erase what the host
    /// saved about it.
    ///
    /// <para>
    /// The host writes the whole <c>messages</c> array it is sent over the stored
    /// copy, on the reasonable ground that the page is the authority on what the
    /// conversation contains. That makes any field the page forgets a field the next
    /// message DELETES from disk — so the answer's reasoning and the PDF it produced
    /// survive being read and are then thrown away by the follow-up question. This is
    /// the round trip, in one test.
    /// </para>
    /// </summary>
    [Fact]
    public void TheNextMessageDoesNotWipeTheAttachmentsAndFilesAlreadySaved()
    {
        JsonElement result = Run("""
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'Look', imagePaths: ['a1.png'], stillImagePaths: ['a1.png'],
                  attachments: [{ file: 'a1.png', fileName: 'cat.png', mediaType: 'image' }] },
                { role: 'assistant', content: 'A cat.', thinking: 'hmm',
                  artifacts: [{ name: 'report.pdf', bytes: 2048, url: '/api/code/artifacts/r/report.pdf' }] }
              ]
            };
            """, """
            __page.byId['text'].value = 'Anything else?';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement messages = result.GetProperty("sent")[0].GetProperty("messages");
        Assert.Equal(3, messages.GetArrayLength());
        Assert.Equal(new[] { "a1.png" }, Strings(messages[0], "imagePaths"));
        Assert.Equal("cat.png", messages[0].GetProperty("attachments")[0].GetProperty("fileName").GetString());
        Assert.Equal("hmm", messages[1].GetProperty("thinking").GetString());
        Assert.Equal("report.pdf", messages[1].GetProperty("artifacts")[0].GetProperty("name").GetString());
    }

    // =====================================================================================
    // the skills switch
    // =====================================================================================

    /// <summary>
    /// Off means off: the setting is saved, the selection is dropped, and the next
    /// message names no skill.
    /// </summary>
    [Fact]
    public void TurningSkillsOffSavesTheSettingAndSendsNoSkills()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = function (call) {
              return call.method === 'POST' ? call.body
                : { thinkByDefault: false, defaultSkills: ['documents'], skillsEnabled: true, maxTokens: 2048 };
            };
            """, """
            var before = [];
            __page.byId['text'].value = 'first';
            __page.byId['send'].dispatch('click');
            before = __page.requests('/api/chat').map(function (c) { return c.body; });

            __page.byId['skills-master'].checked = false;
            __page.byId['skills-master'].dispatch('change');
            return settle(20).then(function () {
              __page.byId['text'].value = 'second';
              __page.byId['send'].dispatch('click');
              return {
                before: before,
                after: __page.requests('/api/chat').map(function (c) { return c.body; }),
                saved: __page.requests('/api/agent/settings').filter(function (c) { return c.method === 'POST'; })
                         .map(function (c) { return c.body; }),
                chips: __page.byId['skillchips'].children.length,
                listClass: __page.byId['skills-list'].className,
              };
            });
            """);

        // With it on, the chat request names the default skill.
        Assert.Equal(new[] { "documents" }, Strings(result.GetProperty("before")[0], "skills"));

        // Saved as a setting, not merely held in the page: it is a preference, and it
        // is what the host reads to build no skill plan at all.
        JsonElement saved = result.GetProperty("saved");
        Assert.True(saved.GetArrayLength() >= 1, "turning the switch off saved nothing");
        Assert.False(saved[saved.GetArrayLength() - 1].GetProperty("skillsEnabled").GetBoolean());

        JsonElement after = result.GetProperty("after");
        Assert.Equal(2, after.GetArrayLength());
        Assert.False(after[1].TryGetProperty("skills", out _),
            "the message after the switch went off still named a skill");
        Assert.False(after[1].GetProperty("skills_discovery").GetBoolean());
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
        Assert.Contains("off", result.GetProperty("listClass").GetString()!, StringComparison.Ordinal);
    }

    /// <summary>
    /// A chat saved when skills were on does not switch them back on for itself.
    /// The setting is the wider fact; a stored selection is a memory of one chat.
    /// </summary>
    [Fact]
    public void ASavedChatDoesNotReviveSkillsThatWereTurnedOff()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: [], skillsEnabled: false, maxTokens: 2048 };
            R['/api/skills'] = { enabled: false, installable: true, skills: [{ name: 'documents', description: 'd' }] };
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 1 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: ['documents'],
              messages: [{ role: 'user', content: 'hello' }]
            };
            """, """
            __page.byId['text'].value = 'again';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                     chips: __page.byId['skillchips'].children.length };
            """);

        Assert.Equal(1, result.GetProperty("sent").GetArrayLength());
        Assert.False(result.GetProperty("sent")[0].TryGetProperty("skills", out _),
            "a saved chat re-selected a skill after the feature was switched off");
        Assert.False(result.GetProperty("sent")[0].GetProperty("skills_discovery").GetBoolean());
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
    }

    [Fact]
    public void DeselectingEverySkillSendsAnExplicitEmptySelection()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = {
              thinkByDefault: false, defaultSkills: ['documents'], skillsEnabled: true, maxTokens: 2048
            };
            """, """
            window.TensorAgent.setSkills([]);
            __page.byId['text'].value = 'search something and create a PowerPoint report';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.True(request.TryGetProperty("skills", out JsonElement skills));
        Assert.Equal(JsonValueKind.Array, skills.ValueKind);
        Assert.Empty(skills.EnumerateArray());
        Assert.False(request.GetProperty("skills_discovery").GetBoolean());
    }

    [Fact]
    public void RoutedNetworkPreflightRestoresThePromptAndOffersTheSettingInline()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = function (call) {
              return call.method === 'POST' ? call.body
                : { thinkByDefault: false, defaultSkills: [], skillsEnabled: true,
                    allowNetwork: false, maxTokens: 2048 };
            };
            R['/api/chat'] = { __status: 503,
              body: { code: 'network_disabled',
                      error: 'network access is disabled by the user for this research workflow' } };
            """, """
            var prompt = '搜索apple M6的信息，并对比M5芯片，然后生成pptx报告';
            __page.byId['text'].value = prompt;
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var action = __page.byId['chat'].querySelector('.notice-action');
              var label = action && action.textContent;
              if (action) action.dispatch('click');
              return settle(10).then(function () {
                return {
                  prompt: __page.byId['text'].value,
                  history: window.TensorAgent.history(),
                  label: label,
                  saved: __page.requests('/api/agent/settings')
                    .filter(function (c) { return c.method === 'POST'; })
                    .map(function (c) { return c.body; })
                };
              });
            });
            """);

        Assert.Equal("搜索apple M6的信息，并对比M5芯片，然后生成pptx报告",
            result.GetProperty("prompt").GetString());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.Equal("Turn on Network", result.GetProperty("label").GetString());
        JsonElement saved = Assert.Single(result.GetProperty("saved").EnumerateArray());
        Assert.True(saved.GetProperty("allowNetwork").GetBoolean());
    }

    [Fact]
    public void RoutedSetupPreflightRestoresThePromptAndAttachmentsAndOpensSettings()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __status: 503,
              body: { code: 'routed_workflow_unavailable',
                      error: 'This routed workflow cannot start: skills_run is unavailable.' } };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a5.md', fileName: 'notes.md',
                                               mediaType: 'text', url: '/uploads/a5.md',
                                               textContent: 'existing notes' });
            var prompt = '搜索apple M6的信息，并对比M5芯片，然后生成pptx报告';
            __page.byId['text'].value = prompt;
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var action = __page.byId['chat'].querySelector('.notice-action');
              var label = action && action.textContent;
              if (action) action.dispatch('click');
              return settle(10).then(function () {
                return {
                  prompt: __page.byId['text'].value,
                  attachments: window.TensorAgent.attachmentCount(),
                  history: window.TensorAgent.history(),
                  label: label,
                  routes: __page.requests('/api/agent/events').map(function (c) { return c.body; })
                };
              });
            });
            """);

        Assert.Equal("搜索apple M6的信息，并对比M5芯片，然后生成pptx报告",
            result.GetProperty("prompt").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.Equal("Open Settings", result.GetProperty("label").GetString());
        Assert.Contains(result.GetProperty("routes").EnumerateArray(), route =>
            route.TryGetProperty("type", out JsonElement type) && type.GetString() == "open-route"
            && route.TryGetProperty("route", out JsonElement name) && name.GetString() == "settings");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReopenedChatPreservesWhetherAnEmptySkillSelectionWasExplicit(bool explicitSelection)
    {
        string explicitLiteral = explicitSelection ? "true" : "false";
        JsonElement result = Run($$"""
            R['/api/agent/conversations'] = {
              conversations: [{ id: 'saved', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 1 }]
            };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              skillsExplicit: {{explicitLiteral}},
              messages: [{ role: 'user', content: 'hello' }]
            };
            """, """
            __page.byId['text'].value = 'search Apple M6 and compare M5, then make a pptx';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        if (explicitSelection)
        {
            Assert.True(request.TryGetProperty("skills", out JsonElement skills));
            Assert.Empty(skills.EnumerateArray());
        }
        else
        {
            Assert.False(request.TryGetProperty("skills", out _),
                "an untouched saved chat disabled host-side skill discovery");
        }
    }

    // =====================================================================================
    // the file the model made
    // =====================================================================================

    /// <summary>
    /// Tapping a generated file asks the APP to open it instead of navigating.
    ///
    /// <para>
    /// Navigating is what used to happen, and it is the reported bug: the route serves
    /// its files as attachments (program-written content must never render in the origin
    /// holding the launch token) and a WKWebView with no download delegate drops an
    /// attachment silently -- so what the user saw was the route's own 404 body,
    /// <c>{"error":"not found"}</c>. Both anchors are covered here, because they are
    /// built by different code: the file card this page renders, and the markdown link
    /// the model copies into its answer.
    /// </para>
    /// </summary>
    [Fact]
    public void TappingAGeneratedFileAsksTheAppToOpenItRatherThanNavigating()
    {
        JsonElement result = Run("""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [
              { skill_step: 'shell', skill: 'documents', detail: 'make_pdf.py', ok: true,
                files: [{ name: 'photo.pdf', bytes: 2048, url: '/api/code/artifacts/run1/photo.pdf' }] },
              { tool_progress: 'finished', tool: 'shell', seconds: 3 },
              { token: 'Here is your PDF: [photo.pdf](/api/code/artifacts/run1/photo.pdf)' },
              { done: true, truncated: false }
            ] };
            """, """
            window.TensorAgent.nativeReady();
            __page.byId['text'].value = 'turn this photo into a pdf';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { links: __page.transcript().reduce(function (all, t) { return all.concat(t.media); }, [])
                .filter(function (m) { return m.tag === 'A'; })
                .map(function (m) { return m.src; }) };
            });
            """);

        // Both links exist and both point at the route.
        string[] links = Strings(result, "links");
        Assert.Contains("/api/code/artifacts/run1/photo.pdf", links);
    }

    /// <summary>
    /// The click itself, on the file card the page rendered: claimed by the page and
    /// handed to the app, with nothing navigated.
    /// </summary>
    [Fact]
    public void TheClickOnAGeneratedFileIsHandedToTheApp()
    {
        JsonElement result = Run("""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [
              { skill_step: 'shell', skill: 'documents', detail: 'make_pdf.py', ok: true,
                files: [{ name: 'photo.pdf', bytes: 2048, url: '/api/code/artifacts/run1/photo%20one.pdf' }] },
              { tool_progress: 'finished', tool: 'shell', seconds: 3 },
              { token: 'Done.' },
              { done: true, truncated: false }
            ] };
            """, """
            window.TensorAgent.nativeReady();
            __page.byId['text'].value = 'turn this photo into a pdf';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var anchor = null;
              (function walk(node) {
                node.children.forEach(function (c) {
                  if (c.tagName === 'A' && String(c.href || '').indexOf('/api/code/artifacts/') === 0) anchor = c;
                  walk(c);
                });
              })(__page.byId['chat']);
              var prevented = false;
              document.dispatch('click', { target: anchor, button: 0, defaultPrevented: false,
                                           preventDefault: function () { prevented = true; } });
              return { found: !!anchor, prevented: prevented,
                       asked: __page.requests('/api/agent/events').map(function (c) { return c.body; }) };
            });
            """);

        Assert.True(result.GetProperty("found").GetBoolean(), "the file card rendered no link to tap");
        Assert.True(result.GetProperty("prevented").GetBoolean(),
            "the tap navigated instead of being handed to the app");
        JsonElement[] asked = result.GetProperty("asked").EnumerateArray()
            .Where(e => e.TryGetProperty("type", out JsonElement t) && t.GetString() == "open-file")
            .ToArray();
        JsonElement open = Assert.Single(asked);
        // The URL is passed through as the page holds it -- percent-encoded -- because
        // the app decodes it one segment at a time against the artifact store.
        Assert.Equal("/api/code/artifacts/run1/photo%20one.pdf", open.GetProperty("url").GetString());
    }

    // =====================================================================================
    // the harness itself
    // =====================================================================================
    //
    // What follows tests PageDom.js, not the page: that the fake network can fail the
    // way WebKit's does, that a stream can die or go quiet, and that aborting is real.
    // A test of the page's behaviour under those conditions is only as good as the
    // model of the conditions, so the model is checked on its own first.

    [Fact]
    public void HarnessARouteThatRejectsMakesFetchRejectWithATypeError()
    {
        JsonElement result = Run("""
            R['/api/harness/down'] = { __reject: 'Load failed' };
            R['/api/harness/default'] = function () { return { __reject: true }; };
            R['/api/harness/throws'] = function () { throw new Error('route exploded'); };
            var tries = 0;
            R['/api/harness/flaky'] = function () { return ++tries === 1 ? { __reject: 'Load failed' } : { ok: true, tries: tries }; };
            """, """
            function outcome(p) {
              return p.then(
                function (res) { return res.json().then(function (body) { return { resolved: true, body: body }; }); },
                function (e) { return { resolved: false, name: e && e.name, message: e && e.message, typeError: e instanceof TypeError }; });
            }
            var threwSynchronously = false, flakyPromise;
            try { flakyPromise = fetch('/api/harness/throws'); } catch (e) { threwSynchronously = true; }
            return Promise.all([
              outcome(fetch('/api/harness/down')),
              outcome(fetch('/api/harness/default')),
              outcome(flakyPromise || Promise.reject(new Error('never made'))),
              outcome(fetch('/api/harness/flaky')),
              outcome(fetch('/api/harness/flaky')),
            ]).then(function (all) {
              return { threwSynchronously: threwSynchronously, down: all[0], byDefault: all[1], throws: all[2],
                       flakyFirst: all[3], flakySecond: all[4],
                       recorded: __page.requests('/api/harness/flaky').length };
            });
            """);

        Assert.False(result.GetProperty("threwSynchronously").GetBoolean(), "a route that throws must reject fetch, not throw out of it");

        JsonElement down = result.GetProperty("down");
        Assert.False(down.GetProperty("resolved").GetBoolean());
        Assert.Equal("TypeError", down.GetProperty("name").GetString());
        Assert.Equal("Load failed", down.GetProperty("message").GetString());
        Assert.True(down.GetProperty("typeError").GetBoolean());

        JsonElement byDefault = result.GetProperty("byDefault");
        Assert.Equal("TypeError", byDefault.GetProperty("name").GetString());
        Assert.Equal("Load failed", byDefault.GetProperty("message").GetString());

        JsonElement throws = result.GetProperty("throws");
        Assert.False(throws.GetProperty("resolved").GetBoolean());
        Assert.Equal("route exploded", throws.GetProperty("message").GetString());

        // A route function decides per call, so "down once, then back" is one counter.
        Assert.False(result.GetProperty("flakyFirst").GetProperty("resolved").GetBoolean());
        Assert.True(result.GetProperty("flakySecond").GetProperty("resolved").GetBoolean());
        Assert.Equal(2, result.GetProperty("flakySecond").GetProperty("body").GetProperty("tries").GetInt32());
        Assert.Equal(2, result.GetProperty("recorded").GetInt32());
    }

    [Fact]
    public void HarnessAHangingStreamReadRejectsWithAbortErrorWhenItsSignalFires()
    {
        JsonElement result = Run("""
            R['/api/harness/quiet'] = { __sse: [{ token: 'one' }], __then: 'hang' };
            """, """
            var ctrl = new AbortController();
            var fired = 0;
            ctrl.signal.addEventListener('abort', function () { fired++; });
            var abortedBefore = ctrl.signal.aborted;
            return fetch('/api/harness/quiet', { signal: ctrl.signal }).then(function (res) {
              var reader = res.body.getReader();
              return reader.read().then(function (first) {
                var settled = null;
                var second = reader.read().then(
                  function (r) { settled = { resolved: true, done: r.done }; },
                  function (e) { settled = { resolved: false, name: e && e.name }; });
                return wait(30).then(function () {
                  var stillPending = settled === null;
                  ctrl.abort();
                  return second.then(function () {
                    return reader.read().then(
                      function () { return { resolved: true }; },
                      function (e) { return { resolved: false, name: e && e.name }; });
                  }).then(function (third) {
                    return { abortedBefore: abortedBefore, abortedAfter: ctrl.signal.aborted, fired: fired,
                             first: first, stillPending: stillPending, second: settled, third: third };
                  });
                });
              });
            });
            """);

        Assert.False(result.GetProperty("abortedBefore").GetBoolean());
        Assert.True(result.GetProperty("abortedAfter").GetBoolean());
        Assert.Equal(1, result.GetProperty("fired").GetInt32());

        JsonElement first = result.GetProperty("first");
        Assert.False(first.GetProperty("done").GetBoolean());
        Assert.Equal("data: {\"token\":\"one\"}\n", first.GetProperty("value").GetString());

        Assert.True(result.GetProperty("stillPending").GetBoolean(), "a hanging read settled on its own");
        JsonElement second = result.GetProperty("second");
        Assert.False(second.GetProperty("resolved").GetBoolean());
        Assert.Equal("AbortError", second.GetProperty("name").GetString());
        // And the reader stays dead: every read after the abort is refused the same way.
        Assert.Equal("AbortError", result.GetProperty("third").GetProperty("name").GetString());
    }

    [Fact]
    public void HarnessAStreamCanDropAfterItsFramesAndTakeRealTimePerRead()
    {
        JsonElement result = Run("""
            R['/api/harness/drops'] = { __sse: [{ token: 'a' }, { token: 'b' }], __then: 'reject', __delay: 20 };
            R['/api/harness/plain'] = { fine: true };
            """, """
            var began = Date.now(), frames = [];
            var dead = new AbortController(); dead.abort();
            var preAborted = fetch('/api/harness/plain', { signal: dead.signal }).then(
              function () { return { resolved: true }; },
              function (e) { return { resolved: false, name: e && e.name }; });
            return fetch('/api/harness/drops').then(function (res) {
              var reader = res.body.getReader();
              function pump() {
                return reader.read().then(function (r) {
                  if (r.done) return { how: 'ended' };
                  frames.push(r.value);
                  return pump();
                }, function (e) { return { how: 'rejected', name: e && e.name, message: e && e.message }; });
              }
              return pump();
            }).then(function (end) {
              return preAborted.then(function (pre) {
                var n = __page.element('div'); n.className = 'notice error'; n.textContent = 'It broke.';
                var b = __page.element('button'); b.className = 'notice-action'; b.textContent = 'Retry';
                n.appendChild(b);
                __page.byId['chat'].appendChild(n);
                var plain = __page.element('div'); plain.className = 'notice'; plain.textContent = 'Just so you know.';
                __page.byId['chat'].appendChild(plain);
                return { elapsed: Date.now() - began, frames: frames, end: end, preAborted: pre,
                         errorNotices: __page.errorNotices(), notices: __page.notices(),
                         recordedDead: __page.requests('/api/harness/plain').length };
              });
            });
            """);

        Assert.Equal(new[] { "data: {\"token\":\"a\"}\n", "data: {\"token\":\"b\"}\n" }, Strings(result, "frames"));
        JsonElement end = result.GetProperty("end");
        Assert.Equal("rejected", end.GetProperty("how").GetString());
        Assert.Equal("TypeError", end.GetProperty("name").GetString());
        Assert.Equal("Load failed", end.GetProperty("message").GetString());
        // Three reads (two frames and the drop) at 20 ms each: the delay is real time.
        Assert.True(result.GetProperty("elapsed").GetInt32() >= 55,
            "the delayed reads came back in " + result.GetProperty("elapsed").GetInt32() + " ms");

        JsonElement pre = result.GetProperty("preAborted");
        Assert.False(pre.GetProperty("resolved").GetBoolean());
        Assert.Equal("AbortError", pre.GetProperty("name").GetString());
        Assert.Equal(1, result.GetProperty("recordedDead").GetInt32());

        Assert.Equal(new[] { "It broke.Retry" }, Strings(result, "errorNotices"));
        Assert.Equal(new[] { "It broke.Retry", "Just so you know." }, Strings(result, "notices"));
    }

    [Fact]
    public void ASendLostBeforeTheTurnIdIsNeverAnsweredWithAnEarlierTurnsAnswer()
    {
        // The host keeps a finished turn for an hour, so "what is this conversation
        // generating" answers with the PREVIOUS question's turn long after it ended.
        // A page recovering a request whose turn it never learned must not take that
        // for its own, or the old answer appears under the new question and both are
        // saved.
        JsonElement result = Run($$"""
            var sends = 0;
            R['/api/chat'] = function () {
              sends++;
              return sends === 1
                ? { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Paris.' }, { done: true, truncated: false }] } }
                : { __reject: 'Load failed' };
            };
            R['/api/agent/turns'] = { turn: { id: 't1', running: false } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Paris.' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'capital of France?';
            __page.byId['send'].dispatch('click');
            return wait(200).then(function () {
              __page.byId['text'].value = 'and of Spain?';
              __page.byId['send'].dispatch('click');
              return wait(400);
            }).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                text: __page.byId['text'].value,
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
              };
            });
            """);

        // The first question and its answer, and nothing else: the second question is
        // back in the composer because the host never took it.
        var turns = Bubbles(result.GetProperty("transcript"));
        Assert.Equal(2, turns.Count);
        Assert.Equal("user", turns[0].GetProperty("role").GetString());
        Assert.Contains("Paris.", turns[1].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("capital of France?", history[0].GetProperty("content").GetString());
        Assert.Equal("and of Spain?", result.GetProperty("text").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        // Never re-read: the page had already read t1 to its end.
        Assert.Equal(0, result.GetProperty("attaches").GetInt32());
    }

    [Fact]
    public void StoppingDuringARecoveryDisarmsIt()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __reject: 'Load failed' };
            R['/api/agent/turns'] = { turn: null };
            R['/api/agent/turns/t9/stop'] = { stopped: true };
            """, """
            window.TensorAgent.__testing.resumeDelays([250]);
            __page.byId['text'].value = 'a question';
            __page.byId['send'].dispatch('click');
            return wait(60).then(function () {
              var pendingBefore = window.TensorAgent.__testing.recovery().pending;
              window.TensorAgent.stop();
              var pendingAfter = window.TensorAgent.__testing.recovery().pending;
              __page.byId['text'].value = 'something else entirely';
              return wait(500).then(function () {
                return {
                  pendingBefore: pendingBefore,
                  pendingAfter: pendingAfter,
                  text: __page.byId['text'].value,
                  generating: window.TensorAgent.isGenerating(),
                  lookups: __page.requests('/api/agent/turns').length,
                };
              });
            });
            """);

        Assert.True(result.GetProperty("pendingBefore").GetBoolean(), "the recovery was never armed");
        Assert.False(result.GetProperty("pendingAfter").GetBoolean(), "stopping left the recovery timer armed");
        // The composer is what the user typed after stopping, not that plus a message
        // the recovery decided to hand back minutes later.
        Assert.Equal("something else entirely", result.GetProperty("text").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    [Fact]
    public void AResumeDuringASendsPrefillLeavesTheRequestAlone()
    {
        // The host answers a chat request's headers only with its first frame, so for
        // the whole of a prefill there is nothing to deliver and nothing to trust by.
        // Aborting the request there loses the turn id AND the acknowledgement of any
        // shared draft it named, which makes every later send a 409.
        JsonElement result = Run($$"""
            var shareTaken = false;
            R['/api/agent/share/claim'] = function () {
              return shareTaken ? { share: null }
                : { share: { id: 'share-1', text: 'look at this', newChat: false, attachments: [] } };
            };
            R['/api/chat'] = function () {
              shareTaken = true;
              return { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Well' }, { done: true, truncated: false }], __delay: 400 } };
            };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            """, """
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(120).then(function () {
              var shared = __page.byId['chips'].querySelectorAll('.shared').length;
              __page.byId['send'].dispatch('click');
              window.TensorAgent.resumeTurn();
              return wait(60).then(function () {
                document.visibilityState = 'hidden';
                document.dispatch('visibilitychange', {});
                document.visibilityState = 'visible';
                document.dispatch('visibilitychange', {});
                window.TensorAgent.resumeTurn();
                return wait(1800).then(function () {
                  return {
                    sharedBefore: shared,
                    sends: __page.requests('/api/chat').length,
                    sharedAfter: __page.byId['chips'].querySelectorAll('.shared').length,
                    lookups: __page.requests('/api/agent/turns').length,
                    transcript: __page.transcript(),
                    generating: window.TensorAgent.isGenerating(),
                    kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
                  };
                });
              });
            });
            """);

        Assert.Equal(1, result.GetProperty("sharedBefore").GetInt32());
        Assert.Equal(1, result.GetProperty("sends").GetInt32());
        // The request was answered, so the share was consumed and its chip is gone.
        Assert.Equal(0, result.GetProperty("sharedAfter").GetInt32());
        Assert.Equal(0, result.GetProperty("lookups").GetInt32());
        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Well", assistant[0].GetProperty("html").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        // The resume looked at the request, saw one that had not been answered yet, and
        // left it alone rather than superseding it.
        Assert.Contains("resume-skip", result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()));
    }

    [Fact]
    public void AStreamThatGoesSilentIsReplacedByTheWatchdogWithoutAnyEvent()
    {
        // Nothing fires when a connection dies without a FIN: no error, no visibility
        // change, no host nudge. The host writes a keep-alive every five seconds, so
        // silence is the signal.
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.streamTrustMs(50);
            window.TensorAgent.__testing.watchdogMs(60);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(700).then(function () {
              return {
                transcript: __page.transcript(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
                kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.Contains("watchdog", result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()));
    }

    [Fact]
    public void AnErrorInATurnIsSaidOnceHoweverOftenTheTurnIsReplayed()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ error: 'The tool could not run.' }, { token: 'Sorry' }], __then: 'reject' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ error: 'The tool could not run.' }, { token: 'Sorry' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'run it';
            __page.byId['send'].dispatch('click');
            return wait(400).then(function () {
              return { errors: __page.errorNotices(), generating: window.TensorAgent.isGenerating() };
            });
            """);

        var errors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Single(errors, e => e!.Contains("The tool could not run.", StringComparison.Ordinal));
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    [Fact]
    public void AChatThatCannotBeReopenedKeepsWhatIsOnTheScreen()
    {
        JsonElement result = Run("""
            var sessions = 0;
            R['/api/sessions?conversation=new'] = function () {
              sessions++;
              return sessions === 1
                ? { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] }
                : { __reject: 'Load failed' };
            };
            R['/api/chat'] = { __sse: [{ token: 'An answer.' }, { done: true, truncated: false }] };
            """, """
            __page.byId['text'].value = 'a question';
            __page.byId['send'].dispatch('click');
            return wait(200).then(function () {
              return window.TensorAgent.openConversation(null);
            }).then(function () {
              return wait(400);
            }).then(function () {
              return { transcript: __page.transcript().map(function (t) { return t.html; }), errors: __page.errorNotices() };
            });
            """);

        string shown = string.Join(" | ", result.GetProperty("transcript").EnumerateArray().Select(t => t.GetString()));
        Assert.Contains("An answer.", shown, StringComparison.Ordinal);
        var reopenErrors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Contains(reopenErrors, e => e!.Contains("Could not open that chat", StringComparison.Ordinal));
    }

    private static string[] Strings(JsonElement parent, string name) =>
        parent.TryGetProperty(name, out JsonElement list) && list.ValueKind == JsonValueKind.Array
            ? list.EnumerateArray().Select(e => e.GetString()!).ToArray()
            : Array.Empty<string>();

    // =====================================================================================
    // coming back to a stream the page could not read
    // =====================================================================================
    //
    // The report these guard: "switch to another app while the model is answering, come
    // back later, see 'Could not open the shared item: Load failed', and nothing works".
    // Behind it were four separate things the page did wrong once its stream was gone:
    // trusted a stream object that would never deliver again, retried a lookup once and
    // gave up, mistook a stream that ended without the host's `done` for a finished
    // answer, and let a POST that failed at the transport level surface as a share error.

    private const string TurnHeader = "'X-TensorAgent-Turn': 't1'";

    /// <summary>
    /// The transcript without the empty-state placeholder: the fake DOM registers ids
    /// lazily, so the page's <c>$('empty').remove()</c> removes a different element and
    /// the placeholder stays in the chat's children for the life of the test.
    /// </summary>
    private static List<JsonElement> Bubbles(JsonElement transcript) => transcript.EnumerateArray()
        .Where(t => !(t.GetProperty("html").GetString() ?? string.Empty).StartsWith("<h1>TensorAgent</h1>", StringComparison.Ordinal))
        .ToList();

    [Fact]
    public void AStaleStreamIsReplacedWhenThePageBecomesVisibleAgain()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.streamTrustMs(0);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var during = { generating: window.TensorAgent.isGenerating(), lookups: __page.requests('/api/agent/turns').length };
              document.visibilityState = 'hidden';
              document.dispatch('visibilitychange', {});
              document.visibilityState = 'visible';
              document.dispatch('visibilitychange', {});
              return wait(500).then(function () {
                return {
                  during: during,
                  transcript: __page.transcript(),
                  history: window.TensorAgent.history(),
                  generating: window.TensorAgent.isGenerating(),
                  attaches: __page.requests('/api/agent/turns/t1').length,
                  errors: __page.errorNotices(),
                  diag: JSON.parse(window.TensorAgent.diagnostics()),
                };
              });
            });
            """);

        Assert.True(result.GetProperty("during").GetProperty("generating").GetBoolean(), "the answer was streaming before the page was hidden");
        Assert.Equal(0, result.GetProperty("during").GetProperty("lookups").GetInt32());
        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hello", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean(), "the page still said it was working after the replayed answer finished");
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        var kinds = result.GetProperty("diag").GetProperty("events").EnumerateArray().Select(e => e.GetProperty("k").GetString()).ToList();
        Assert.Contains("visibility", kinds);
        Assert.Contains("supersede", kinds);
        Assert.Contains("attach", kinds);
        Assert.Contains("finish", kinds);
        Assert.False(result.GetProperty("diag").GetProperty("attached").GetBoolean());
    }

    [Fact]
    public void AStreamThatIsStillDeliveringIsLeftAloneByANudge()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var nudged = window.TensorAgent.resumeTurn();
              return settle(20).then(function () {
                return {
                  nudged: nudged,
                  lookups: __page.requests('/api/agent/turns').length,
                  generating: window.TensorAgent.isGenerating(),
                  delivering: JSON.parse(window.TensorAgent.diagnostics()).delivering,
                };
              });
            });
            """);

        Assert.True(result.GetProperty("nudged").GetBoolean());
        Assert.Equal(0, result.GetProperty("lookups").GetInt32());
        Assert.True(result.GetProperty("generating").GetBoolean());
        Assert.True(result.GetProperty("delivering").GetBoolean());
    }

    [Fact]
    public void AStreamThatRejectsMidAnswerIsTakenUpAgainWithBackoffNotOnce()
    {
        JsonElement result = Run($$"""
            var lookups = 0;
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'reject' } };
            R['/api/agent/turns'] = function () {
              lookups++;
              return lookups < 3 ? { __reject: 'Load failed' } : { turn: { id: 't1', running: true } };
            };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10, 20, 30, 40, 50]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(600).then(function () {
              return {
                lookups: __page.requests('/api/agent/turns').length,
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices(),
                kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
              };
            });
            """);

        Assert.True(result.GetProperty("lookups").GetInt32() >= 3, "the page gave up before the third lookup");
        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        Assert.Equal(2, result.GetProperty("history").GetArrayLength());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        var kinds = result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()).ToList();
        Assert.Contains("stream-failed", kinds);
        Assert.Contains("resume-retry", kinds);
        Assert.Contains("resume-lookup-failed", kinds);
    }

    [Fact]
    public void AStreamThatEndsWithoutADoneFrameIsTakenUpAgainWhenTheTurnIsStillRunning()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }] } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
                kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        // ONE assistant entry, carrying the whole answer -- not a fragment and then the whole.
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hello", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.Contains("eof-without-done", result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()));
    }

    [Fact]
    public void AStreamThatEndsWithoutADoneFrameIsReadAgainWhenTheTurnFinishedMeanwhile()
    {
        // The connection died as the answer was ending. The host still has the turn --
        // finished turns are kept for an hour -- so its replay, which ends with the
        // frame that says the turn is over, is the whole answer; the fragment this
        // reader was left holding is not.
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }] } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: false } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo there' }, { done: true, truncated: false }] };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello there", assistant[0].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hello there", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
    }

    [Fact]
    public void AStreamThatEndsWithoutADoneFrameFinishesWithWhatItHasWhenTheTurnIsGone()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }] } };
            R['/api/agent/turns'] = { turn: null };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hel", assistant[0].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hel", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(0, result.GetProperty("attaches").GetInt32());
    }

    [Fact]
    public void ASendThatLosesItsStreamBeforeTheTurnIdAttachesToTheTurnTheHostStarted()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __reject: 'Load failed' };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices(),
                text: __page.byId['text'].value,
              };
            });
            """);

        var turns = Bubbles(result.GetProperty("transcript"));
        Assert.Equal(2, turns.Count);
        Assert.Equal("user", turns[0].GetProperty("role").GetString());
        Assert.Contains("Hello", turns[1].GetProperty("html").GetString());
        Assert.Equal(2, result.GetProperty("history").GetArrayLength());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        Assert.Equal("", result.GetProperty("text").GetString());
    }

    [Fact]
    public void ASendThatNeverReachedTheHostGoesBackIntoTheComposer()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __reject: 'Load failed' };
            R['/api/agent/turns'] = { turn: null };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'cat.png', mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'what is this';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
              };
            });
            """);

        Assert.Empty(Bubbles(result.GetProperty("transcript")));
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal("what is this", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
    }

    [Fact]
    public void WhenTheHostCannotBeReachedForLongEnoughThePageHandsTheScreenBackAndSaysSo()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'reject' } };
            R['/api/agent/turns'] = { __reject: 'Load failed' };
            """, """
            window.TensorAgent.__testing.resumeDelays([5, 5]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices(),
                lookups: __page.requests('/api/agent/turns').length,
              };
            });
            """);

        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(2, result.GetProperty("lookups").GetInt32());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hel", history[1].GetProperty("content").GetString());
        var errors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Contains(errors, e => e!.Contains("Lost the connection", StringComparison.Ordinal));
        Assert.DoesNotContain(errors, e => e!.Contains("Load failed", StringComparison.Ordinal));
    }

    [Fact]
    public void AShareClaimThatFailsAtTheTransportIsRetriedOnceAndNotReported()
    {
        JsonElement result = Run("""
            var claims = 0;
            R['/api/agent/share/claim'] = function () {
              claims++;
              return claims === 1 ? { __reject: 'Load failed' } : { share: null };
            };
            """, """
            var claimsBefore = __page.requests('/api/agent/share/claim').length;
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(500).then(function () {
              return {
                claims: __page.requests('/api/agent/share/claim').length - claimsBefore,
                errors: __page.errorNotices(),
              };
            });
            """);

        Assert.Equal(2, result.GetProperty("claims").GetInt32());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
    }

    [Fact]
    public void AShareClaimTheHostRefusesIsReportedWithoutARetry()
    {
        JsonElement result = Run("""
            R['/api/agent/share/claim'] = { __status: 500, body: { error: 'broken' } };
            """, """
            var claimsBefore = __page.requests('/api/agent/share/claim').length;
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(500).then(function () {
              return {
                claims: __page.requests('/api/agent/share/claim').length - claimsBefore,
                errors: __page.errorNotices(),
              };
            });
            """);

        Assert.Equal(1, result.GetProperty("claims").GetInt32());
        var errors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Contains(errors, e => e!.Contains("Could not open the shared item", StringComparison.Ordinal) && e.Contains("HTTP 500", StringComparison.Ordinal));
    }

    [Fact]
    public void ALongAnswerOfThousandsOfFramesRendersWholeAndOnce()
    {
        // A page re-attaching to a long answer is handed every frame so far at once;
        // the answer is painted per chunk rather than per frame, and must still come
        // out complete, in one bubble.
        JsonElement result = Run("""
            var frames = [];
            for (var i = 0; i < 2000; i++) frames.push({ token: 'w' + i + ' ' });
            frames.push({ done: true, truncated: false });
            R['/api/chat'] = { __sse: frames };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var assistant = __page.transcript().filter(function (t) { return t.role === 'assistant' && t.html.indexOf('<h1>TensorAgent</h1>') !== 0; });
              return { bubbles: assistant.length, words: (assistant[0] && assistant[0].html || '').split('w').length - 1, generating: window.TensorAgent.isGenerating() };
            });
            """);

        Assert.Equal(1, result.GetProperty("bubbles").GetInt32());
        Assert.Equal(2000, result.GetProperty("words").GetInt32());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

}
