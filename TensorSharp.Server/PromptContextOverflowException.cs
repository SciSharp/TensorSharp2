using System;

namespace TensorSharp.Server
{
    /// <summary>
    /// The prompt, plus its generation reserve, does not fit the model's
    /// context window — a client input error the HTTP layer maps to 400.
    /// Derives from <see cref="InvalidOperationException"/> so callers that
    /// catch the broader type still handle it.
    /// </summary>
    public sealed class PromptContextOverflowException : InvalidOperationException
    {
        public PromptContextOverflowException(string message) : base(message)
        {
        }
    }
}
