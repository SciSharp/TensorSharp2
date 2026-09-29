#!/usr/bin/env bash
# Actual two-GPU numerical coverage; unavailable GPU counts are not passes.
set -euo pipefail
repo="${1:-/workspace/TensorSharp-validation}"
cd "$repo"
export PATH="/workspace/dotnet:/usr/local/cuda/bin:$PATH"
export DOTNET_ROOT=/workspace/dotnet TENSORSHARP_GGML_NO_UPDATE=1
export TS_TEST_GLM_CUDA=1 TS_TEST_GGML_BACKEND=cuda OMP_NUM_THREADS=2
out="$repo/artifacts/parallel-validation/gates"
mkdir -p "$out"
dotnet build InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
    -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:CudaArch=compute_120 \
    > "$out/build.log" 2>&1
filter='FullyQualifiedName~GlmExplicitParallelismCudaTests|FullyQualifiedName~GlmDsaNativeBatchedDecodeLayerSplitTests.BatchedDecode_MatchesPerSequenceDecode_OneGpu|FullyQualifiedName~GlmDsaNativeBatchedDecodeLayerSplitTests.BatchedDecode_MatchesPerSequenceDecode_TwoGpus'
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build --no-restore \
    --filter "$filter" --logger 'console;verbosity=detailed' \
    --logger 'trx;LogFileName=gpu-tests.trx' --results-directory "$out" \
    > "$out/gpu-tests.log" 2>&1
ctest --test-dir TensorSharp.GGML.Native/build --output-on-failure -V \
    -R 'deepseek41-(tp-worker-lifetime|moe-tensor-parallel$|tp-failure-recovery|moe-tensor-parallel-cuda-2$|quantized-strip-projections-cuda)' \
    --output-junit "$out/native-tests.xml" > "$out/native-tests.log" 2>&1
git -C ExternalProjects/ggml status --porcelain --untracked-files=all > "$out/upstream-status.txt"
test ! -s "$out/upstream-status.txt"
