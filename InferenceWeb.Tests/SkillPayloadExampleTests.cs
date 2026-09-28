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
using System.Text;
using TensorSharp.Runtime;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Renders a skills request through the real production path and writes the result to
/// <c>TS_SKILL_PAYLOAD_OUT</c>, so what a model actually receives can be read rather
/// than inferred.
///
/// <para>
/// This is documentation that cannot go stale: it calls <see cref="SkillPrompt"/>,
/// <see cref="SkillTools"/> and the per-family <see cref="ChatTemplate"/> renderers
/// directly, so the day a renderer changes, the captured example changes with it. It is
/// skipped unless the environment variable is set, because its output is a file rather
/// than an assertion.
/// </para>
/// </summary>
public class SkillPayloadExampleTests : IDisposable
{
    private readonly string _baseDir;

    public SkillPayloadExampleTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-payload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static string OutPath => Environment.GetEnvironmentVariable("TS_SKILL_PAYLOAD_OUT");

    private void WriteSkill(string name, string description, string body, params (string, string)[] extras)
    {
        string dir = Path.Combine(_baseDir, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            $"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n");
        foreach ((string path, string content) in extras)
        {
            string full = Path.Combine(dir, path);
            Directory.CreateDirectory(Path.GetDirectoryName(full));
            File.WriteAllText(full, content);
        }
    }

    /// <summary>Every family that has its own renderer, with the one TensorSharp calls for it.</summary>
    private static readonly (string Arch, Func<List<ChatMessage>, List<ToolFunction>, string> Render)[] Families =
    {
        ("gemma4",   (m, t) => ChatTemplate.RenderGemma4(m, true, tools: t)),
        ("gpt-oss",  (m, t) => ChatTemplate.RenderHarmony(m, true, tools: t)),
        ("qwen35",   (m, t) => ChatTemplate.RenderQwen35(m, true, enableThinking: true, tools: t)),
        ("nemotron_h_moe", (m, t) => ChatTemplate.RenderNemotron(m, true, tools: t)),
        ("glm-dsa",  (m, t) => ChatTemplate.RenderGlmDsa(m, true, tools: t)),
        ("deepseek4", (m, t) => ChatTemplate.RenderDeepSeek4(m, true, tools: t)),
        ("mistral3", (m, t) => ChatTemplate.RenderMistral3(m, true)),
    };

    [Fact]
    public void CaptureExamplePayloads()
    {
        string outPath = OutPath;
        if (string.IsNullOrEmpty(outPath))
            return;                                   // not a capture run

        WriteSkill("acme-invoice",
            "Formats invoice numbers for ACME Corp. Use whenever the user asks how to number, format or validate an ACME invoice, purchase order or credit note.",
            "# ACME invoice numbering\n\nEvery ACME invoice number has exactly this shape:\n\n    ACME-<region>-<year><sequence>-<checkdigit>\n\nSee `reference.md` for the regional codes.",
            ("reference.md", "Regional codes: EMA (Europe), APC (Asia-Pacific), NAM (North America).\n"),
            ("scripts/check.py", "print('ok')\n"));
        WriteSkill("xlsx", "Create and edit Excel spreadsheets.", "Use openpyxl.");

        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        IReadOnlyList<Skill> all = registry.Skills;
        IReadOnlyList<Skill> selected = all.Where(s => s.Id == "acme-invoice").ToList();

        var sb = new StringBuilder();

        void Section(string title)
        {
            sb.Append("\n\n");
            sb.Append(new string('=', 78)).Append('\n');
            sb.Append(title).Append('\n');
            sb.Append(new string('=', 78)).Append("\n\n");
        }

        // ---- 1. the injected system block, selected-skill form -----------------
        SkillPlan withTools = SkillPrompt.Plan(selected, Array.Empty<Skill>(),
            new SkillPromptOptions { ContextTokens = 32768, ToolsAvailable = true });
        Section("1. SYSTEM BLOCK injected for  \"skills\": [\"acme-invoice\"]   (tool-capable family)");
        sb.Append(withTools.Instructions);

        // ---- 2. discovery form: catalogue only ---------------------------------
        SkillPlan discovery = SkillPrompt.Plan(Array.Empty<Skill>(), all,
            new SkillPromptOptions { ContextTokens = 32768, ToolsAvailable = true });
        Section("2. SYSTEM BLOCK injected for  \"skills_discovery\": true  (catalogue only, nothing selected)");
        sb.Append(discovery.Instructions);

        // ---- 3. the same selection on a family that cannot carry tools ----------
        SkillPlan noTools = SkillPrompt.Plan(selected, Array.Empty<Skill>(),
            new SkillPromptOptions { ContextTokens = 32768, ToolsAvailable = false });
        Section("3. SYSTEM BLOCK for a family that renders NO tool declarations (mistral3)");
        sb.Append(noTools.Instructions);

        // ---- 4. the synthesized tools ------------------------------------------
        List<ToolFunction> tools = SkillTools.BuiltIn(allowScripts: true);
        Section("4. SYNTHESIZED TOOLS (the ToolFunction objects spliced into the request)");
        foreach (ToolFunction t in tools)
        {
            sb.Append("- ").Append(t.Name).Append(": ").Append(t.Description).Append('\n');
            if (t.Parameters != null)
                foreach (KeyValuePair<string, ToolParameter> p in t.Parameters)
                    sb.Append("    ").Append(p.Key).Append(" (").Append(p.Value.Type).Append("): ")
                      .Append(p.Value.Description).Append('\n');
            sb.Append('\n');
        }

        // ---- 5. how each family writes those tools into the prompt --------------
        var convo = new List<ChatMessage>
        {
            new() { Role = "system", Content = withTools.Instructions.TrimEnd('\n') },
            new() { Role = "user", Content = "Which region code is Asia-Pacific? Check the reference file." },
        };

        foreach ((string arch, Func<List<ChatMessage>, List<ToolFunction>, string> render) in Families)
        {
            SkillModelCapabilities caps = SkillCapabilities.For(arch);
            Section($"5.{arch}  — rendered prompt   (declares tools: {caps.ToolsRendered}, renders tool results: {caps.ToolResultsRendered})");
            string prompt;
            try { prompt = render(convo, caps.ToolsRendered ? tools : null); }
            catch (Exception ex) { prompt = "<renderer threw: " + ex.GetType().Name + ": " + ex.Message + ">"; }
            sb.Append(prompt);
        }

        // ---- 6. a full skills_read round trip -----------------------------------
        Section("6. THE TOOL ROUND TRIP — what the loop adds between rounds");
        var context = new SkillToolContext(selected.ToList());
        var call = new ToolCall
        {
            Name = "skills_read",
            Arguments = new Dictionary<string, object> { ["skill"] = "acme-invoice", ["path"] = "reference.md" },
        };
        SkillToolResult result = SkillTools.Execute(call, context);
        sb.Append("assistant turn (tool call the model emitted, never forwarded to the client):\n");
        sb.Append("  name      = ").Append(call.Name).Append('\n');
        sb.Append("  arguments = ").Append(string.Join(", ", call.Arguments.Select(kv => kv.Key + "=" + kv.Value))).Append('\n');
        sb.Append("\ntool turn (what TensorSharp answers with, appended in process):\n");
        sb.Append("  ok      = ").Append(result.Ok).Append('\n');
        sb.Append("  content =\n");
        foreach (string line in (result.Content ?? string.Empty).Split('\n'))
            sb.Append("    | ").Append(line).Append('\n');

        File.WriteAllText(outPath, sb.ToString());
    }
}
