using System.Text.Json;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="ToolFunction.ParseList"/> has to accept the shapes people actually
/// write. The CLI's <c>--tools</c> flag used to deserialize the file straight
/// into <see cref="ToolFunction"/>, so a definition in the JSON Schema shape —
/// which is what anyone copying a tool out of an API request has — killed the
/// process with an unhandled JsonException on the schema's own
/// <c>"type": "object"</c>.
/// </summary>
public class ToolFunctionParseTests
{
    private const string JsonSchemaShape = """
    [
      {
        "name": "get_current_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string", "description": "The city name." },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
          },
          "required": ["city"]
        }
      }
    ]
    """;

    private const string FlatShape = """
    [
      {
        "name": "get_current_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
          "city": { "type": "string", "description": "The city name." },
          "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["city"]
      }
    ]
    """;

    private const string OpenAiWrapperShape = """
    [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": { "type": "string", "description": "The city name." },
              "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["city"]
          }
        }
      }
    ]
    """;

    private static void AssertWeatherTool(ToolFunction fn)
    {
        Assert.Equal("get_current_weather", fn.Name);
        Assert.Equal("Get the current weather in a given city.", fn.Description);
        Assert.Equal(2, fn.Parameters.Count);
        Assert.Equal("string", fn.Parameters["city"].Type);
        Assert.Equal("The city name.", fn.Parameters["city"].Description);
        Assert.Equal(new[] { "celsius", "fahrenheit" }, fn.Parameters["unit"].Enum);
        Assert.Equal(new[] { "city" }, fn.Required);
        // The schema's own keywords must not leak in as parameters.
        Assert.False(fn.Parameters.ContainsKey("type"));
        Assert.False(fn.Parameters.ContainsKey("properties"));
        Assert.False(fn.Parameters.ContainsKey("required"));
    }

    [Theory]
    [InlineData(nameof(JsonSchemaShape))]
    [InlineData(nameof(FlatShape))]
    [InlineData(nameof(OpenAiWrapperShape))]
    public void ParsesEveryAcceptedShapeIdentically(string which)
    {
        string json = which switch
        {
            nameof(JsonSchemaShape) => JsonSchemaShape,
            nameof(FlatShape) => FlatShape,
            _ => OpenAiWrapperShape,
        };

        var tools = ToolFunction.ParseList(json);
        Assert.Single(tools);
        AssertWeatherTool(tools[0]);
    }

    [Fact]
    public void AcceptsASingleObjectAndAToolsEnvelope()
    {
        Assert.Single(ToolFunction.ParseList(JsonSchemaShape.Trim().TrimStart('[').TrimEnd(']')));
        Assert.Single(ToolFunction.ParseList($$"""{"tools": {{JsonSchemaShape}} }"""));
    }

    [Fact]
    public void EmptyInputIsAnEmptyList()
    {
        Assert.Empty(ToolFunction.ParseList(""));
        Assert.Empty(ToolFunction.ParseList("   "));
        Assert.Empty(ToolFunction.ParseList("[]"));
    }

    [Fact]
    public void ToolWithoutParametersIsStillUsable()
    {
        var tools = ToolFunction.ParseList("""[{"name": "now", "description": "Current time."}]""");
        Assert.Single(tools);
        Assert.Equal("now", tools[0].Name);
        Assert.Empty(tools[0].Parameters);
        Assert.Empty(tools[0].Required);
    }

    /// <summary>
    /// A tool's <c>parameters</c> is a JSON Schema document, so <c>enum</c> may
    /// hold values of any type and <c>type</c> may be a union. Both reach the
    /// model as prompt text, so the spelling matters: a boolean has to arrive as
    /// JSON's <c>true</c> rather than .NET's <c>True</c>, and a nullable field
    /// has to keep a type name the renderers recognise instead of degrading to
    /// <c>any</c>. Same rule as the server's tool parser (issue #142).
    /// </summary>
    [Fact]
    public void JsonSchemaValuesThatAreNotStringsKeepTheirJsonSpelling()
    {
        var tools = ToolFunction.ParseList("""
        [{
          "name": "configure",
          "parameters": {
            "type": "object",
            "properties": {
              "level":   { "type": "integer", "enum": [0, 1, 2] },
              "persist": { "type": "boolean", "enum": [true, false] },
              "scope":   { "type": ["string", "null"] },
              "choice":  { "type": "string", "enum": ["a", null] }
            }
          }
        }]
        """);

        var fn = Assert.Single(tools);
        Assert.Equal(new[] { "0", "1", "2" }, fn.Parameters["level"].Enum);
        Assert.Equal(new[] { "true", "false" }, fn.Parameters["persist"].Enum);
        Assert.Equal("string", fn.Parameters["scope"].Type);
        Assert.Equal(new[] { "a", "null" }, fn.Parameters["choice"].Enum);
    }

    /// <summary>
    /// Every failure mode has to surface as a <see cref="JsonException"/> (or a
    /// subclass of it, e.g. JsonReaderException) so the CLI's single catch turns
    /// a bad file into a one-line error instead of an unhandled crash.
    /// </summary>
    [Fact]
    public void MalformedInputThrowsJsonExceptionRatherThanSomethingElse()
    {
        Assert.ThrowsAny<JsonException>(() => ToolFunction.ParseList("not json"));
        Assert.ThrowsAny<JsonException>(() => ToolFunction.ParseList("[1, 2, 3]"));
        Assert.ThrowsAny<JsonException>(() => ToolFunction.ParseList("\"a string\""));
    }
}
