#!/usr/bin/env bash
# Provision an isolated, unmodified-upstream CUDA validation checkout.
set -euo pipefail
repo="${1:-/workspace/TensorSharp-validation}"
archive="${2:-/workspace/tensorsharp-source.tar.gz}"
ggml_revision="${3:-353b63b439f27ab2cc19dac97ab1681ba6d2d084}"
mkdir -p "$repo/artifacts/parallel-validation" /workspace/dotnet
tar -xzf "$archive" -C "$repo"
export PATH="/workspace/dotnet:/usr/local/cuda/bin:$PATH"
export DOTNET_ROOT=/workspace/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
if ! test -x /workspace/dotnet/dotnet; then
    curl -fsSL https://dot.net/v1/dotnet-install.sh -o /workspace/dotnet-install.sh
    bash /workspace/dotnet-install.sh --channel 10.0 --install-dir /workspace/dotnet
fi
if ! test -d "$repo/ExternalProjects/ggml/.git"; then
    mkdir -p "$repo/ExternalProjects/ggml"
    git -C "$repo/ExternalProjects/ggml" init
    git -C "$repo/ExternalProjects/ggml" remote add origin https://github.com/ggml-org/ggml.git
    git -C "$repo/ExternalProjects/ggml" fetch --depth 1 origin "$ggml_revision"
    git -C "$repo/ExternalProjects/ggml" checkout --detach FETCH_HEAD
fi
test "$(git -C "$repo/ExternalProjects/ggml" rev-parse HEAD)" = "$ggml_revision"
test -z "$(git -C "$repo/ExternalProjects/ggml" status --porcelain --untracked-files=all)"
{
    date -u -Iseconds
    uname -a
    dotnet --info
    nvcc --version
    nvidia-smi
    nvidia-smi topo -m
    git -C "$repo/ExternalProjects/ggml" rev-parse HEAD
    git -C "$repo/ExternalProjects/ggml" status --porcelain --untracked-files=all
} > "$repo/artifacts/parallel-validation/environment.txt"
cmake -S "$repo/TensorSharp.GGML.Native" -B "$repo/TensorSharp.GGML.Native/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
    -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF \
    -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build "$repo/TensorSharp.GGML.Native/build" -j 16
test -z "$(git -C "$repo/ExternalProjects/ggml" status --porcelain --untracked-files=all)"
cd "$repo"
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release \
    -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:CudaArch=compute_120
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
    -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:CudaArch=compute_120
