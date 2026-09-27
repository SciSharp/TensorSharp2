using InferenceWeb.Tests;

if (args.Length is < 2 or > 3 || args[0] != "--write-glm" ||
    (args.Length == 3 && args[2] != "--quantize"))
{
    Console.Error.WriteLine("Usage: dotnet run --project eng/validation/ParallelPlacementFixture -- --write-glm /workspace/tiny-glm-dsa.gguf [--quantize]");
    return 2;
}

string path = Path.GetFullPath(args[1]);
Directory.CreateDirectory(Path.GetDirectoryName(path)!);
if (File.Exists(path))
{
    Console.Error.WriteLine($"Output already exists: {path}");
    return 2;
}

GlmDsaSyntheticModelBuilder.Write(path, quantize: args.Length == 3);
Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new
{
    path,
    architecture = "glm-dsa",
    bytes = new FileInfo(path).Length,
    synthetic = true,
    purpose = "Deterministic end-to-end mode and numerical validation; not language-quality or production-throughput evidence."
}));
return 0;
