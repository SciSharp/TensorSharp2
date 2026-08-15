using System;
using System.IO;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Confinement check for client-supplied file paths that must resolve
    /// inside the upload directory.
    /// </summary>
    internal static class UploadPathGuard
    {
        /// <summary>
        /// True when <paramref name="path"/> resolves inside <paramref name="root"/>.
        /// Built on <see cref="Path.GetRelativePath(string, string)"/> rather than a
        /// string prefix comparison, so a sibling directory that merely shares the
        /// root's name (<c>uploads-evil</c>) and traversal out of the root are both
        /// rejected. Both arguments must already be full paths.
        /// </summary>
        public static bool IsInsideDirectory(string root, string path)
        {
            string relative = Path.GetRelativePath(root, path);
            return relative != ".."
                && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
                && !Path.IsPathRooted(relative);
        }
    }
}
