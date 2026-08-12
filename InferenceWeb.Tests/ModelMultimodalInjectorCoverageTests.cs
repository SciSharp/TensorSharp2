// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;

namespace InferenceWeb.Tests;

/// <summary>
/// Structural guard for <see cref="ModelMultimodalInjector"/>.
///
/// The injector routes every stage of the multimodal hand-off through a
/// <c>switch (_model)</c> over concrete model types. Adding a new vision model means
/// touching FOUR of those switches, and missing one fails SILENTLY: the encoder still
/// runs, the prompt still grows by N image rows, but nothing ever calls the model's
/// <c>SetVisionEmbeddings</c>, so those rows keep the embedding of the filler token and
/// the model answers as if it never saw the image.
///
/// That is exactly what happened to Muse-Glimmer: <c>LoadProjectors</c> and
/// <c>ProcessMuseGlimmerHistory</c> were both wired up, but neither
/// <c>QueuePreparedVisionEmbeddings</c> nor <c>QueuePreparedVisionEmbeddingsForSlice</c>
/// had a <c>MuseGlimmerModel</c> case. Asked "这是什么图片" about a banner, the model
/// replied that it had been given "a huge block of characters, all 'o' repeated".
///
/// Rather than re-test that end to end (which needs a 10 GB model plus a 2 GB mmproj),
/// this reads the IL of the four switches and asserts they cover the same set of model
/// types. It is a fast, model-free tripwire for the whole class of bug.
/// </summary>
public class ModelMultimodalInjectorCoverageTests
{
    private static readonly BindingFlags AnyMethod =
        BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    /// <summary>
    /// Concrete <see cref="ModelBase"/>-derived types a method's IL performs an
    /// <c>isinst</c>/<c>castclass</c> against — i.e. the model types its
    /// <c>switch (_model)</c> handles.
    /// </summary>
    private static HashSet<Type> ModelTypesReferencedBy(string methodName)
    {
        MethodInfo method = typeof(ModelMultimodalInjector).GetMethod(methodName, AnyMethod)
            ?? throw new InvalidOperationException(
                $"ModelMultimodalInjector.{methodName} not found — did the injector get refactored?");

        byte[] il = method.GetMethodBody()?.GetILAsByteArray()
            ?? throw new InvalidOperationException($"{methodName} has no IL body.");

        Module module = method.Module;
        var found = new HashSet<Type>();

        // isinst = 0x75, castclass = 0x74; both are followed by a 4-byte metadata token.
        // Scanning for the opcode byte alone would produce false positives from operand
        // bytes, so every candidate token is resolved and simply ignored when it does not
        // name a ModelBase subclass.
        for (int i = 0; i + 4 < il.Length; i++)
        {
            if (il[i] != 0x75 && il[i] != 0x74)
                continue;
            int token = BitConverter.ToInt32(il, i + 1);
            Type? resolved;
            try
            {
                resolved = module.ResolveType(token);
            }
            catch
            {
                continue;
            }
            if (resolved != null && typeof(ModelBase).IsAssignableFrom(resolved) && !resolved.IsAbstract)
                found.Add(resolved);
        }

        return found;
    }

    /// <summary>
    /// Every model that gets a projector loaded must also be able to receive the
    /// embeddings that projector produces, through BOTH queue paths: the whole-prompt
    /// one and the per-prefill-slice one the continuous-batching executor uses.
    /// </summary>
    [Theory]
    [InlineData("QueuePreparedVisionEmbeddings")]
    [InlineData("QueuePreparedVisionEmbeddingsForSlice")]
    public void EveryModelWithAProjector_CanAlsoReceiveItsVisionEmbeddings(string queueMethod)
    {
        HashSet<Type> loaders = ModelTypesReferencedBy("LoadProjectors");
        HashSet<Type> receivers = ModelTypesReferencedBy(queueMethod);

        Assert.NotEmpty(loaders);

        // A model may legitimately load an AUDIO-only projector, so only require a
        // vision receiver for models that actually declare a vision encoder hook.
        var missing = loaders
            .Where(t => t.GetMethod("SetVisionEmbeddings", AnyMethod) != null)
            .Where(t => !receivers.Contains(t))
            .Select(t => t.Name)
            .OrderBy(n => n)
            .ToList();

        Assert.True(missing.Count == 0,
            $"ModelMultimodalInjector.{queueMethod} has no case for: {string.Join(", ", missing)}. " +
            "Those models will run their vision encoder and then silently drop the embeddings, " +
            "leaving the image rows as filler-token embeddings.");
    }

    /// <summary>Muse-Glimmer specifically — the model this guard was written for.</summary>
    [Fact]
    public void MuseGlimmer_IsCoveredByEveryVisionStage()
    {
        foreach (string stage in new[]
                 {
                     "LoadProjectors",
                     "ProcessPromptTokens",
                     "QueuePreparedVisionEmbeddings",
                     "QueuePreparedVisionEmbeddingsForSlice",
                 })
        {
            Assert.True(ModelTypesReferencedBy(stage).Contains(typeof(MuseGlimmerModel)),
                $"ModelMultimodalInjector.{stage} does not handle MuseGlimmerModel.");
        }
    }
}
