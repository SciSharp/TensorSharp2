// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace tsg_glm {

// Gather a TP rank's strip from each checkpoint row. Reading one strip at a
// time caused millions of small reads on network storage. Bounded source
// slabs amortize the reads; each worker reuses its scratch allocation.
inline bool read_file_strips(FILE * file, uint64_t offset, size_t stride, size_t strip,
                             size_t bytes, void * output, std::vector<uint8_t> & scratch,
                             size_t slab_bytes = 1024 * 1024)
{
    if (!file || !output || strip == 0 || stride < strip || bytes == 0 || bytes % strip != 0)
        return false;
    const size_t rows = bytes / strip;
    const uint64_t max_offset = uint64_t(std::numeric_limits<int64_t>::max());
    if (offset > max_offset || strip > max_offset - offset ||
        rows - 1 > (max_offset - offset - strip) / stride)
        return false;
    slab_bytes = std::max(slab_bytes, strip);
    const size_t rows_per_slab = std::min(rows, 1 + (slab_bytes - strip) / stride);
    const size_t capacity = (rows_per_slab - 1) * stride + strip;
    if (scratch.size() < capacity) scratch.resize(capacity);
    auto * dest = static_cast<uint8_t *>(output);
    for (size_t row = 0; row < rows;)
    {
        const size_t count = std::min(rows_per_slab, rows - row);
        const size_t span = (count - 1) * stride + strip;
        const uint64_t start = offset + uint64_t(row) * stride;
#if defined(_WIN32)
        if (_fseeki64(file, (int64_t) start, SEEK_SET) != 0) return false;
#else
        if (fseeko(file, (off_t) start, SEEK_SET) != 0) return false;
#endif
        // The last slab ends at the final strip, not at the end of its source
        // row: a slice of the final tensor can end exactly at checkpoint EOF.
        if (fread(scratch.data(), 1, span, file) != span) return false;
        for (size_t i = 0; i < count; ++i)
            std::memcpy(dest + (row + i) * strip, scratch.data() + i * stride, strip);
        row += count;
    }
    return true;
}

} // namespace tsg_glm
