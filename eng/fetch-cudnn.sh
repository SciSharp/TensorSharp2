#!/usr/bin/env bash
# Provision the cuDNN runtime used by the Wan / Qwen-Image VAE convolutions on the
# CUDA backends into ExternalProjects/cudnn. The Linux twin of fetch-cudnn.ps1.
#
# ggml lowers conv2d to im2col + mul_mat, and on a full-resolution VAE decode that
# lowering — not the GEMM — is the dominant cost, because a 3x3 convolution
# materialises 9x its input before any math happens. cuDNN convolves directly, so
# tsg_cuda_cudnn_conv.cu replaces the lowering (and the band tiling that exists only
# to bound the im2col scratch) with one library call on ggml's own device buffers.
#
# cuDNN is NOT part of the CUDA Toolkit, so this fetches the official public
# redistributable from developer.download.nvidia.com — no account or licence click
# needed for the redist channel. Only include/ and lib/ are kept: the build needs the
# headers and the runtime dlopen()s the libraries.
#
# Environment:
#   TENSORSHARP_CUDNN=OFF|0|false  skip entirely (the VAE stays on im2col+GEMM)
#   TENSORSHARP_CUDNN_VERSION      override the pinned cuDNN version
#   TENSORSHARP_CUDNN_CUDA_MAJOR   override the CUDA major (default: probe nvcc, else 12)
#   TS_CUDNN_DIR / CUDNN_DIR       use an existing cuDNN install and download nothing
#   TENSORSHARP_GGML_NO_UPDATE     keep whatever is already provisioned
#
# Always exits 0: cuDNN is an optional accelerator, so an offline or proxied machine
# must still get a working build.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUDNN_DIR="${REPO_ROOT}/ExternalProjects/cudnn"
MARKER="${CUDNN_DIR}/.complete"

# Pinned rather than "newest available" on purpose: a convolution library silently
# changing under a released build is worse than being a few releases behind.
DEFAULT_VERSION="9.25.0.15"

step() { echo "cudnn: $*"; }

is_off() {
    case "$(printf '%s' "${TENSORSHARP_CUDNN:-}" | tr '[:upper:]' '[:lower:]')" in
        off|0|false|no) return 0 ;;
        *) return 1 ;;
    esac
}

is_true() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|on|true|yes) return 0 ;;
        *) return 1 ;;
    esac
}

cuda_major() {
    if [ -n "${TENSORSHARP_CUDNN_CUDA_MAJOR:-}" ]; then printf '%s' "${TENSORSHARP_CUDNN_CUDA_MAJOR}"; return; fi
    local nvcc=""
    if command -v nvcc >/dev/null 2>&1; then nvcc="$(command -v nvcc)"
    elif [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"; fi
    if [ -n "$nvcc" ]; then
        local v
        v="$("$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"
        if [ -n "$v" ]; then printf '%s' "$v"; return; fi
    fi
    printf '12'
}

if is_off; then
    step "TENSORSHARP_CUDNN is off; the VAE convolutions stay on ggml im2col+GEMM."
    exit 0
fi

# An existing cuDNN wins: nothing to download, and the user's own version is the one
# they meant to build against.
for d in "${TS_CUDNN_DIR:-}" "${CUDNN_DIR_ENV:-}" "${CUDA_HOME:-}" /usr /usr/local/cuda; do
    [ -n "$d" ] || continue
    if [ -f "$d/include/cudnn.h" ]; then
        step "using the cuDNN already installed at $d"
        exit 0
    fi
done

VERSION="${TENSORSHARP_CUDNN_VERSION:-$DEFAULT_VERSION}"
MAJOR="$(cuda_major)"
STAMP="${VERSION}-cuda${MAJOR}"

if [ -f "$MARKER" ]; then
    HAVE="$(tr -d '[:space:]' < "$MARKER")"
    if [ "$HAVE" = "$STAMP" ]; then step "already provisioned ($STAMP)"; exit 0; fi
    if is_true "${TENSORSHARP_GGML_NO_UPDATE:-}"; then
        step "TENSORSHARP_GGML_NO_UPDATE set; keeping the provisioned $HAVE"
        exit 0
    fi
    step "replacing provisioned $HAVE with $STAMP"
    rm -rf "$CUDNN_DIR"
fi

case "$(uname -m)" in
    aarch64|arm64) PLAT="linux-aarch64" ;;
    ppc64le)       PLAT="linux-ppc64le" ;;
    *)             PLAT="linux-x86_64" ;;
esac
ARCHIVE="cudnn-${PLAT}-${VERSION}_cuda${MAJOR}-archive.tar.xz"
URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/${PLAT}/${ARCHIVE}"

provision() {
    mkdir -p "$CUDNN_DIR" || return 1
    local tmp="${CUDNN_DIR}/${ARCHIVE}.part"
    rm -f "$tmp"
    step "downloading $URL"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 30 -o "$tmp" "$URL" || return 1
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$tmp" "$URL" || return 1
    else
        step "neither curl nor wget is available"
        return 1
    fi
    step "extracting"
    # Strip the archive's top-level directory and keep only what is used.
    tar -xJf "$tmp" -C "$CUDNN_DIR" --strip-components=1 \
        --wildcards 'cudnn-*/include/*' 'cudnn-*/lib/*' || return 1
    rm -f "$tmp"
    [ -f "${CUDNN_DIR}/include/cudnn.h" ] || { step "extraction finished but include/cudnn.h is missing"; return 1; }
    printf '%s' "$STAMP" > "$MARKER"
    step "ready at $CUDNN_DIR ($STAMP, $(du -sh "$CUDNN_DIR" 2>/dev/null | cut -f1))"
    step "note: the VAE uses it only with TS_VAE_CUDNN_CONV=1 - it wins on short clips"
    step "      and loses badly on long ones (see fast_conv_enabled in ggml_ops_core.cpp)."
    step "      TENSORSHARP_CUDNN=OFF skips this download entirely."
    return 0
}

if ! provision; then
    step "not provisioned; the VAE convolutions will use ggml im2col+GEMM."
    step "set TENSORSHARP_CUDNN=OFF to stop trying, or TS_CUDNN_DIR to point at an install."
    rm -f "${CUDNN_DIR}"/*.part 2>/dev/null || true
fi
exit 0
