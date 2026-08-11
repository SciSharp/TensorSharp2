#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source_dir="${script_dir}/Native"
build_dir="${source_dir}/build"
install_dir="${source_dir}/dist"
configuration="${CONFIGURATION:-Release}"
jobs="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
arch="${CMAKE_OSX_ARCHITECTURES:-arm64}"

# MLX compiles Metal shaders, so this build needs a working `xcrun metal` --
# which a machine with only the Command Line Tools does not have, and which
# Xcode 16+ ships as a separately downloadable component. Provision both and
# build against the Xcode the helper resolves rather than whatever xcode-select
# happens to point at, so no `sudo xcode-select -s` is required. See
# eng/ensure-metal-toolchain.sh for the full story.
metal_setup="${repo_root}/eng/ensure-metal-toolchain.sh"
if [[ -f "${metal_setup}" ]]; then
    resolved_developer_dir="$(bash "${metal_setup}" --print-developer-dir)"
    if [[ -n "${resolved_developer_dir}" ]]; then
        export DEVELOPER_DIR="${resolved_developer_dir}"
    fi
fi

# A build tree configured under a different toolchain has the old SDK frozen
# into CMakeCache.txt -- METAL_LIB, CMAKE_TAPI and friends keep pointing into
# /Library/Developer/CommandLineTools even after Xcode becomes available, and
# the Metal steps keep failing. Detect the change and reconfigure from scratch.
# Only generated output is removed: the fetched _deps/*-src checkouts (~130 MB
# of mlx, mlx-c, fmt, ...) and their -subbuild stamps are kept so this costs a
# reconfigure, not a re-clone.
toolchain_stamp="${build_dir}/.tensorsharp-toolchain"
current_stamp="developer_dir=${DEVELOPER_DIR:-<default>}
sdk=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || echo '<unknown>')
arch=${arch}"

if [[ -f "${build_dir}/CMakeCache.txt" ]] && \
   [[ "$(cat "${toolchain_stamp}" 2>/dev/null || true)" != "${current_stamp}" ]]; then
    echo "mlx-native: build toolchain changed since the last configure; discarding the stale CMake cache in ${build_dir}"
    find "${build_dir}" -mindepth 1 -maxdepth 1 ! -name '_deps' -exec rm -rf {} +
    if [[ -d "${build_dir}/_deps" ]]; then
        find "${build_dir}/_deps" -mindepth 1 -maxdepth 1 -name '*-build' -exec rm -rf {} +
    fi
fi

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE="${configuration}" \
  -DCMAKE_INSTALL_PREFIX="${install_dir}" \
  -DCMAKE_OSX_ARCHITECTURES="${arch}"

# Written only after a successful configure, so an interrupted run re-cleans
# rather than trusting a half-configured tree.
printf '%s\n' "${current_stamp}" > "${toolchain_stamp}"

cmake --build "${build_dir}" --config "${configuration}" --target install --parallel "${jobs}"
