// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Matrix;

/// <summary>
/// One feature / prompt type exercised by the matrix. Captures both
/// the metadata for filtering and the CLI flag template that drives
/// TensorSharp.Cli for this feature.
/// </summary>
public sealed record FeatureSpec(
    string Id,
    string DisplayName,
    FeatureKind Kind,
    string? PromptFile,
    string? MediaFile,
    string? ToolsFile,
    string? MultiTurnFile,
    int PrefillTokens,
    int DecodeTokens,
    int MaxTokens,
    bool EnableThinking,
    bool RequiresImage,
    bool RequiresAudio,
    bool RequiresVideo,
    bool RequiresTools)
{
    /// <summary>
    /// Substrings the assistant output must contain for the cell to pass
    /// correctness. Case-insensitive. Empty means "no semantic check" — only
    /// applicable to synthetic benchmark modes that don't generate real text.
    /// </summary>
    public IReadOnlyList<string> ExpectedContains { get; init; } = Array.Empty<string>();
}

public enum FeatureKind
{
    SyntheticPrefill,
    SyntheticDecode,
    Text,
    UploadedText,
    Image,
    Audio,
    Video,
    Tools,
    Thinking,
    MultiTurn,
}

public static class FeatureCatalog
{
    public const int DefaultMaxTokens = 64;

    public static readonly FeatureSpec SyntheticPrefill512 = new(
        Id: "pp512",
        DisplayName: "Synthetic prefill (512 tok)",
        Kind: FeatureKind.SyntheticPrefill,
        PromptFile: null,
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 512,
        DecodeTokens: 0,
        MaxTokens: 0,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false);

    public static readonly FeatureSpec SyntheticPrefill2048 = SyntheticPrefill512 with
    {
        Id = "pp2048",
        DisplayName = "Synthetic prefill (2048 tok)",
        PrefillTokens = 2048,
    };

    public static readonly FeatureSpec SyntheticDecode128 = new(
        Id: "tg128",
        DisplayName: "Synthetic decode (128 tok after 32 prefill)",
        Kind: FeatureKind.SyntheticDecode,
        PromptFile: null,
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 32,
        DecodeTokens: 128,
        MaxTokens: 0,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false);

    public static readonly FeatureSpec ShortText = new(
        Id: "short_text",
        DisplayName: "Short text prompt (single turn)",
        Kind: FeatureKind.Text,
        PromptFile: "prompts/short_text.txt",
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: DefaultMaxTokens,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // The prompt asks why the sky is blue. Any correct answer must
        // mention "blue" and at least gesture at scattering/wavelength.
        ExpectedContains = new[] { "blue" },
    };

    public static readonly FeatureSpec LongText = ShortText with
    {
        Id = "long_text",
        DisplayName = "Long text prompt (~1k tokens, single turn)",
        PromptFile = "prompts/long_text.txt",
        // The prompt asks for a paragraph of up to six sentences, which does not
        // fit in DefaultMaxTokens — and a model that reasons before answering
        // (gemma-4-E4B, gpt-oss) spends the whole budget on the preamble and gets
        // cut off before writing a single sentence of the summary. That reads as
        // a correctness failure when the model is fine; give the answer room.
        // 512, not 256: gpt-oss reasons ALWAYS, and its MLX greedy chain was
        // measured spending all 256 tokens in the analysis channel — the parser
        // correctly hides that channel, so the cell saw 0 output chars.
        MaxTokens = 512,
        // Summary of a paged-KV-cache report. Any correct summary names
        // either the data structure or the technique.
        ExpectedContains = new[] { "paged" },
    };

    public static readonly FeatureSpec UploadedText = ShortText with
    {
        Id = "uploaded_text",
        DisplayName = "Uploaded text file",
        PromptFile = "prompts/upload_text.txt",
        Kind = FeatureKind.UploadedText,
        // Same budget argument as LongText: this asks for a two-bullet analysis,
        // and a model that reasons first (gemma-4-E4B was measured narrating its
        // plan for the whole 64-token default) never reaches the bullets. The
        // assertion is about the ANSWER, so the answer needs room. gpt-oss was
        // still enumerating log lines in its analysis channel at 256.
        MaxTokens = 512,
        // Log analysis; bullet (1) asks for the error and its timestamp - asked
        // FIRST so a budget-bounded reasoner emits it before anything else. The
        // original bullet order asked "which event repeats most often" first,
        // and the log had a genuine 9-9 tie between "request" and "generation
        // done": gpt-oss and Muse-Glimmer both burned their entire budget
        // agonizing over the tie ("They tie. But maybe...") and never reached
        // the error bullet. The question now has one unambiguous answer per
        // bullet (error at 08:01:12; alice with 4 requests).
        ExpectedContains = new[] { "08:01:12" },
    };

    public static readonly FeatureSpec MultiTurn = new(
        Id: "multi_turn",
        DisplayName: "Multi-turn chat (KV reuse)",
        Kind: FeatureKind.MultiTurn,
        PromptFile: null,
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: "multi_turn/three_turn.jsonl",
        PrefillTokens: 0,
        DecodeTokens: 0,
        // Per TURN, and the assertion spans turns 2 and 3, so the default 64 is
        // the tightest budget in the catalog relative to what it asks for. On
        // Muse-Glimmer (which opens every turn with a reasoning channel) it
        // produced the bare fragment "Your favourite colour" and read as a
        // KV-reuse failure; the reuse was in fact fine (measured 79.3% prefix
        // reuse on the same backend), the answer simply never arrived.
        MaxTokens: 192,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // Turn 1 establishes 'Alex' / 'teal'; turn 2 asks for the name back,
        // turn 3 asks for the colour back. If KV reuse is correct the model
        // must surface both facts somewhere in the combined turn-2 + turn-3
        // assistant output.
        ExpectedContains = new[] { "alex", "teal" },
    };

    public static readonly FeatureSpec Tools = new(
        Id: "tools",
        DisplayName: "Function / tool calling",
        Kind: FeatureKind.Tools,
        PromptFile: "prompts/tools_question.txt",
        MediaFile: null,
        ToolsFile: "tools/weather_tools.json",
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        // gpt-oss deliberates in its analysis channel before emitting the call;
        // at 128 its MLX chain produced 0 visible chars. The assertion needs the
        // call itself, so the deliberation must fit too.
        MaxTokens: 384,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: true)
    {
        // The prompt asks about Tokyo weather and exposes one tool,
        // get_current_weather. The model must name the tool and the city.
        ExpectedContains = new[] { "get_current_weather", "tokyo" },
    };

    public static readonly FeatureSpec Thinking = new(
        Id: "thinking",
        DisplayName: "Thinking / reasoning mode",
        Kind: FeatureKind.Thinking,
        PromptFile: "prompts/thinking_question.txt",
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        // Raised twice, both times from measurement: at 256 gemma-4-E4B was
        // still laying out the algebra, and at 512 it was one line short —
        // cut off at "t_meet = 170 km / 150 km/h" with the meeting time never
        // stated (byte-identical truncation on Metal and MLX, so it is purely
        // budget). This cell deliberately asks a model to think out loud in
        // LaTeX; 1024 is what that actually takes.
        MaxTokens: 1024,
        EnableThinking: true,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // Two-train word problem.
        //   A leaves at 09:00 from station A, 60 km/h, toward B (200 km).
        //   B leaves at 09:30 from station B, 90 km/h, toward A.
        // By 09:30 A has covered 30 km, leaving 170 km. Closing speed
        // 60 + 90 = 150 km/h, so 170/150 h = 1 h 8 min from 09:30 => 10:38.
        ExpectedContains = new[] { "10:38" },
    };

    public static readonly FeatureSpec Image = new(
        Id: "image",
        DisplayName: "Image prompt",
        Kind: FeatureKind.Image,
        PromptFile: "prompts/image_question.txt",
        MediaFile: "media/image.png",
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        // Media cells get the same headroom as the long-form text cells: the
        // question is one sentence, but a reasoning model narrates before it
        // answers and would otherwise be cut off mid-preamble.
        MaxTokens: 192,
        EnableThinking: false,
        RequiresImage: true,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // Default media is the TensorSharp banner (eng/make-test-media.sh), whose
        // rendered title is legible to every VLM in the matrix, so a single
        // keyword is a sound "did the image reach the model at all" probe. Swap
        // the file and this expectation together — see README.
        ExpectedContains = new[] { "tensorsharp" },
    };

    public static readonly FeatureSpec Audio = new(
        Id: "audio",
        DisplayName: "Audio prompt",
        Kind: FeatureKind.Audio,
        PromptFile: "prompts/audio_question.txt",
        MediaFile: "media/sample.wav",
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: 192,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: true,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // eng/make-test-media.sh speaks "The quick brown fox jumps over the lazy
        // dog near the river bank" into sample.wav. One content word is enough to
        // separate a real transcription from "the audio is silent or
        // unintelligible", which is what the prompt invites when the clip never
        // reached the encoder.
        ExpectedContains = new[] { "fox" },
    };

    public static readonly FeatureSpec Video = new(
        Id: "video",
        DisplayName: "Video prompt",
        Kind: FeatureKind.Video,
        PromptFile: "prompts/video_question.txt",
        MediaFile: "media/sample.mp4",
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: 192,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: true,
        RequiresTools: false)
    {
        // eng/make-test-media.sh renders a slow zoom on the repo banner. The
        // first fixture here was an abstract red square panning across white;
        // it reached the model correctly (frames extracted, 3 x 48 vision
        // tokens injected) yet Gemma 4 still answered "no actual video content
        // has been provided", so the cell measured legibility rather than
        // plumbing. The banner keys off the same title the image cell does,
        // which every VLM in the matrix reads.
        ExpectedContains = new[] { "tensorsharp" },
    };

    public static readonly IReadOnlyList<FeatureSpec> All = new[]
    {
        SyntheticPrefill512,
        SyntheticPrefill2048,
        SyntheticDecode128,
        ShortText,
        LongText,
        UploadedText,
        MultiTurn,
        Tools,
        Thinking,
        Image,
        Audio,
        Video,
    };

    public static FeatureSpec? FindById(string id)
    {
        foreach (FeatureSpec f in All)
        {
            if (string.Equals(f.Id, id, StringComparison.OrdinalIgnoreCase))
            {
                return f;
            }
        }
        return null;
    }
}
