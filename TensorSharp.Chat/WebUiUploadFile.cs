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
using System.IO;

namespace TensorSharp.Chat;

/// <summary>
/// A transport-neutral upload part. Its stream is opened only after every part is
/// validated and the batch's storage is reserved; the upload service disposes it.
/// </summary>
public sealed record WebUiUploadFile(string FileName, long Length, Func<Stream> OpenReadStream);
