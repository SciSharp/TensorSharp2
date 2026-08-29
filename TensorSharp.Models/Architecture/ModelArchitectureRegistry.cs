// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// The set of architecture plug-ins this process can load, keyed by every
    /// <c>general.architecture</c> string that selects them.
    ///
    /// Built-ins register themselves through <see cref="BuiltInArchitectures"/> on
    /// first use. <see cref="Register"/> is public so a host application can add an
    /// architecture without forking TensorSharp.
    /// </summary>
    public static class ModelArchitectureRegistry
    {
        /// <summary>
        /// Architecture assumed when a GGUF declares none and no detector claims it.
        /// Historically Qwen 3's block, which is also what unlabelled community
        /// conversions of that family ship.
        /// </summary>
        public const string DefaultArchitectureId = "qwen3";

        private static readonly object Gate = new();
        private static readonly Dictionary<string, ModelArchitectureDescriptor> ByAlias =
            new(StringComparer.OrdinalIgnoreCase);
        private static readonly List<ModelArchitectureDescriptor> Ordered = new();
        private static bool _builtInsLoaded;

        /// <summary>Add an architecture plug-in. Idempotent for an identical descriptor
        /// instance; conflicting aliases are rejected loudly rather than silently
        /// shadowing an existing family.</summary>
        public static void Register(ModelArchitectureDescriptor descriptor)
        {
            ArgumentNullException.ThrowIfNull(descriptor);
            descriptor.Validate();

            lock (Gate)
            {
                foreach (string alias in descriptor.Aliases)
                {
                    if (ByAlias.TryGetValue(alias, out var existing) && !ReferenceEquals(existing, descriptor))
                    {
                        throw new InvalidOperationException(
                            $"Architecture alias '{alias}' is already registered by '{existing.Id}'; " +
                            $"'{descriptor.Id}' cannot claim it too.");
                    }
                }

                if (Ordered.Contains(descriptor))
                    return;

                foreach (string alias in descriptor.Aliases)
                    ByAlias[alias] = descriptor;
                Ordered.Add(descriptor);
            }
        }

        /// <summary>All registered architectures, in registration order.</summary>
        public static IReadOnlyList<ModelArchitectureDescriptor> All
        {
            get
            {
                EnsureBuiltIns();
                lock (Gate) return Ordered.ToArray();
            }
        }

        /// <summary>Every architecture id a model file may declare.</summary>
        public static IReadOnlyCollection<string> KnownAliases
        {
            get
            {
                EnsureBuiltIns();
                lock (Gate) return ByAlias.Keys.ToArray();
            }
        }

        public static bool TryGet(string architecture, out ModelArchitectureDescriptor descriptor)
        {
            EnsureBuiltIns();
            lock (Gate)
                return ByAlias.TryGetValue(architecture ?? string.Empty, out descriptor);
        }

        /// <summary>
        /// Pick the plug-in for a model file. <paramref name="architecture"/> is the
        /// GGUF's <c>general.architecture</c>, which may be null: a file with no
        /// metadata at all is offered to every registered detector first, and only
        /// then falls back to <see cref="DefaultArchitectureId"/>.
        /// </summary>
        public static ModelArchitectureDescriptor Resolve(string architecture, GgufFile probe)
        {
            EnsureBuiltIns();

            if (!string.IsNullOrEmpty(architecture))
            {
                if (TryGet(architecture, out var byName))
                    return byName;
                throw new NotSupportedException(
                    $"Unsupported architecture: {architecture}. Registered: " +
                    string.Join(", ", All.Select(d => d.Id).OrderBy(x => x, StringComparer.Ordinal)) + ".");
            }

            if (probe != null)
            {
                foreach (var candidate in All)
                {
                    if (candidate.DetectFromTensors != null && candidate.DetectFromTensors(probe))
                        return candidate;
                }
            }

            if (TryGet(DefaultArchitectureId, out var fallback))
                return fallback;

            throw new NotSupportedException(
                $"The model declares no architecture and the default '{DefaultArchitectureId}' is not registered.");
        }

        /// <summary>
        /// Locate the mmproj companion for a loaded model when the operator did not
        /// name one, using the architecture's own <see
        /// cref="ModelArchitectureDescriptor.ProjectorFileHints"/>. Returns null when
        /// the family declares no hints or nothing beside the model matches.
        ///
        /// Hosts call this only after deciding the model can actually use a projector
        /// (it implements a multimodal capability), so this never has to know which
        /// families are multimodal.
        /// </summary>
        public static string FindCompanionProjector(string architecture, string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath) || !TryGet(architecture, out var descriptor))
                return null;
            if (descriptor.ProjectorFileHints.Count == 0)
                return null;

            string directory = System.IO.Path.GetDirectoryName(modelPath);
            if (string.IsNullOrEmpty(directory) || !System.IO.Directory.Exists(directory))
                return null;

            foreach (string hint in descriptor.ProjectorFileHints)
            {
                if (!hint.Contains('*'))
                {
                    string exact = System.IO.Path.Combine(directory, hint);
                    if (System.IO.File.Exists(exact))
                        return exact;
                    continue;
                }

                // Ordered so the pick is stable when several companions sit side by side.
                string[] matches = System.IO.Directory.GetFiles(directory, hint);
                if (matches.Length > 0)
                {
                    Array.Sort(matches, StringComparer.OrdinalIgnoreCase);
                    return matches[0];
                }
            }

            return null;
        }

        private static void EnsureBuiltIns()
        {
            lock (Gate)
            {
                if (_builtInsLoaded)
                    return;
                _builtInsLoaded = true;
            }

            // Outside the lock: registration re-enters Register(), which takes it.
            BuiltInArchitectures.RegisterAll();
        }
    }
}
