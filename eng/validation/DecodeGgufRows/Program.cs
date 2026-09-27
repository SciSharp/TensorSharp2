// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length != 3)
    throw new ArgumentException("Usage: DecodeGgufRows model.gguf rows.json decoded-rows.json (metadata only)");
if (File.Exists(args[2]))
    throw new IOException($"Refusing to overwrite {args[2]}");
using var gguf = new GgufFile(args[0]);
var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
var rows = JsonNode.Parse(File.ReadAllText(args[1]))!.AsArray();
foreach (JsonNode? row in rows)
{
    var tokens = row!["Tokens"]!.AsArray().Select(value => value!.GetValue<int>()).ToList();
    row["DecodedText"] = tokenizer.Decode(tokens);
}
File.WriteAllText(args[2], rows.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Decoded {rows.Count} rows from GGUF vocabulary; no model weights loaded.");
