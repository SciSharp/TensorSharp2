using System;
using System.IO;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Resolves a client-supplied upload reference to a full path inside the
    /// upload directory. The canonical form is the bare server filename that
    /// <c>/api/upload</c> returns as <c>file</c>; an absolute path is still
    /// accepted for compatibility with older clients, but must resolve inside
    /// the upload directory — traversal and a sibling directory that merely
    /// shares the root's name are both rejected.
    /// </summary>
    internal static class UploadFileReference
    {
        public static bool TryResolve(string uploadRoot, string reference, out string fullPath)
        {
            fullPath = null;
            if (string.IsNullOrWhiteSpace(reference))
                return false;

            string root = Path.GetFullPath(uploadRoot);

            // A name with no directory component joins under the root with no
            // escape route, so it needs no further containment check.
            if (IsBareFileName(reference))
            {
                fullPath = Path.Combine(root, reference);
                return true;
            }

            string full;
            try
            {
                full = Path.GetFullPath(reference);
            }
            catch (Exception ex) when (ex is ArgumentException or PathTooLongException or NotSupportedException)
            {
                return false;
            }

            if (!IsInsideDirectory(root, full))
                return false;

            fullPath = full;
            return true;
        }

        private static bool IsBareFileName(string reference)
        {
            return reference.IndexOf('/') < 0
                && reference.IndexOf('\\') < 0
                && reference != "."
                && reference != ".."
                && !Path.IsPathRooted(reference);
        }

        /// <summary>
        /// True when <paramref name="path"/> resolves inside <paramref name="root"/>.
        /// Built on <see cref="Path.GetRelativePath(string, string)"/> rather than a
        /// string prefix comparison so traversal and a sibling directory sharing the
        /// root's name both land outside. Both arguments must already be full paths.
        /// </summary>
        private static bool IsInsideDirectory(string root, string path)
        {
            string relative = Path.GetRelativePath(root, path);
            return relative != ".."
                && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
                && !Path.IsPathRooted(relative);
        }
    }
}
