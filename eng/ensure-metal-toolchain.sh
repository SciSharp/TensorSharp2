#!/usr/bin/env bash
# Provision the Metal shader compiler that the MLX native backend needs at
# build time (TensorSharp.Backends.MLX/build-native-macos.sh -> mlx-c -> mlx).
#
# MLX preprocesses its Metal kernel headers during the build --
# mlx/backend/metal/make_compiled_preamble.sh shells out to
# `xcrun -sdk macosx metal` -- so a *working* Metal compiler is mandatory. Two
# independent things break it on an otherwise healthy developer machine, and
# this script fixes both:
#
#   1. The active developer directory is the Command Line Tools.
#      Installing the CLT alone leaves `xcode-select -p` pointing at
#      /Library/Developer/CommandLineTools, which ships no Toolchains directory
#      and therefore no Metal compiler at all:
#          xcrun: error: unable to find utility "metal", not a developer tool
#          or in PATH
#      Fix: build with DEVELOPER_DIR pointing at a full Xcode.app.
#
#   2. Xcode no longer bundles the Metal compiler.
#      Since Xcode 16 the `metal` binary inside Xcode.app is only a stub; the
#      real toolchain is a separately downloadable component (~700 MB):
#          error: cannot execute tool 'metal' due to missing Metal Toolchain;
#          use: xcodebuild -downloadComponent MetalToolchain
#      Fix: run that download. It needs no sudo and installs into the system
#      asset store, so it is shared by every project and survives Xcode updates.
#
# Either failure reaches the user through MLX as the far less obvious:
#      error Metal compiler header resolution failed for .../reduce_utils.h
#
# This script deliberately does NOT run `sudo xcode-select -s`: repointing the
# machine-wide developer directory is a global side effect this build does not
# need. It resolves an Xcode and reports it instead, so the caller can export
# DEVELOPER_DIR for the build alone.
#
# Usage:
#   eng/ensure-metal-toolchain.sh
#       Provision the toolchain, logging to stdout.
#   eng/ensure-metal-toolchain.sh --print-developer-dir
#       Same, but print the resolved developer directory as the only stdout
#       line (all logging goes to stderr) so callers can capture it:
#           DEVELOPER_DIR="$(eng/ensure-metal-toolchain.sh -p)"
#
# Environment overrides:
#   TENSORSHARP_MLX_SKIP_METAL_SETUP  1/ON/true -> do nothing and exit 0, for
#                                     machines provisioned out of band.
#   TENSORSHARP_XCODE_DEVELOPER_DIR   Explicit <Xcode.app>/Contents/Developer to
#                                     use; tried before any autodetection.
#
# Written for the bash 3.2 that macOS ships as /bin/bash -- no mapfile, no
# associative arrays, no ${var,,}.
set -euo pipefail

print_developer_dir=0
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--print-developer-dir) print_developer_dir=1 ;;
        -h|--help)
            echo "usage: $(basename "$0") [--print-developer-dir]"
            exit 0
            ;;
        *)
            echo "metal-toolchain: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
    shift
done

# In --print-developer-dir mode stdout is reserved for the resolved path, so
# every message and every subprocess's chatter has to go to stderr instead.
log() {
    if [ "${print_developer_dir}" = "1" ]; then
        echo "metal-toolchain: $*" >&2
    else
        echo "metal-toolchain: $*"
    fi
}

warn() {
    echo "metal-toolchain: $*" >&2
}

die() {
    echo "metal-toolchain: $*" >&2
    exit 1
}

run_logged() {
    if [ "${print_developer_dir}" = "1" ]; then
        "$@" >&2
    else
        "$@"
    fi
}

is_truthy() {
    case "${1:-}" in
        1|ON|on|On|TRUE|true|True|YES|yes|Yes) return 0 ;;
        *) return 1 ;;
    esac
}

if [ "$(uname -s)" != "Darwin" ]; then
    log "not macOS; nothing to provision"
    exit 0
fi

if is_truthy "${TENSORSHARP_MLX_SKIP_METAL_SETUP:-}"; then
    log "TENSORSHARP_MLX_SKIP_METAL_SETUP is set; leaving the Metal toolchain alone"
    # Pass through whatever the environment already selected so the caller does
    # not silently fall back to a different developer directory.
    if [ "${print_developer_dir}" = "1" ] && [ -n "${DEVELOPER_DIR:-}" ]; then
        printf '%s\n' "${DEVELOPER_DIR}"
    fi
    exit 0
fi

# --- 1. Locate a full Xcode --------------------------------------------------
# The Command Line Tools have no usr/bin/xcodebuild and no Toolchains/ at all,
# so either check separates them from a real Xcode.app. The path test is the
# more future-proof of the two (a future Xcode could drop the `metal` stub once
# the compiler is fully asset-based), so it wins when it matches.
is_full_xcode() {
    local dir="${1:-}"
    [ -n "${dir}" ] || return 1
    [ -d "${dir}" ] || return 1
    [ -x "${dir}/usr/bin/xcodebuild" ] || return 1
    case "${dir}" in
        *.app/Contents/Developer) return 0 ;;
    esac
    [ -e "${dir}/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal" ]
}

# Read before this script exports DEVELOPER_DIR below: `xcode-select -p` honours
# that variable, so asking afterwards would just echo our own choice back.
active_dir="$(/usr/bin/xcode-select -p 2>/dev/null || true)"

candidates=""
add_candidate() {
    [ -n "${1:-}" ] || return 0
    candidates="${candidates}${1}
"
}

add_candidate "${TENSORSHARP_XCODE_DEVELOPER_DIR:-}"
add_candidate "${DEVELOPER_DIR:-}"
add_candidate "${active_dir}"
add_candidate "/Applications/Xcode.app/Contents/Developer"
# Side-by-side installs (Xcode-16.2.app, Xcode-beta.app, ...). Newest first so a
# machine keeping an old Xcode around does not pin the build to it. Read line by
# line rather than iterating word-split `ls` output, so a renamed bundle with a
# space in it ("Xcode 16.2.app") still resolves.
while IFS= read -r app; do
    add_candidate "${app}/Contents/Developer"
done <<EOF
$(ls -d /Applications/Xcode*.app 2>/dev/null | sort -r || true)
EOF
# Xcode installed outside /Applications (e.g. /Users/Shared, an external disk).
if command -v mdfind >/dev/null 2>&1; then
    while IFS= read -r app; do
        add_candidate "${app}/Contents/Developer"
    done <<EOF
$(mdfind "kMDItemCFBundleIdentifier == 'com.apple.dt.Xcode'" 2>/dev/null || true)
EOF
fi

developer_dir=""
while IFS= read -r candidate; do
    [ -n "${candidate}" ] || continue
    if is_full_xcode "${candidate}"; then
        developer_dir="${candidate}"
        break
    fi
done <<EOF
${candidates}
EOF

if [ -z "${developer_dir}" ]; then
    # Xcode itself cannot be fetched unattended -- both the App Store and
    # developer.apple.com require an authenticated Apple ID -- so this is where
    # automation has to hand back to the user.
    die "no full Xcode installation was found, and the Metal compiler ships only with Xcode.
    The Command Line Tools alone (${active_dir:-not configured}) cannot compile Metal shaders.
    Install Xcode with one of:
      * the Mac App Store: https://apps.apple.com/app/xcode/id497799835
      * https://developer.apple.com/download/all/?q=Xcode (Apple ID required)
      * 'brew install --cask xcodes' then 'xcodes install --latest'
    Then re-run the build, or point this script at an existing install with
      export TENSORSHARP_XCODE_DEVELOPER_DIR=/path/to/Xcode.app/Contents/Developer
    To build the rest of TensorSharp without the MLX backend, set TENSORSHARP_MLX_NATIVE_SKIP=true."
fi

export DEVELOPER_DIR="${developer_dir}"
xcodebuild="${developer_dir}/usr/bin/xcodebuild"

if [ "${active_dir}" != "${developer_dir}" ]; then
    log "xcode-select points at '${active_dir:-<unset>}', which has no Metal compiler; building with DEVELOPER_DIR=${developer_dir}"
    log "(optional, to make it the machine-wide default: sudo xcode-select -s '${developer_dir}')"
else
    log "using Xcode at ${developer_dir}"
fi

# Xcode's bundled packages install on first launch; until then xcrun can fail in
# confusing ways. Only warn -- the metal probe below is the real gate, and this
# check is advisory on machines where Xcode was provisioned by an installer.
if ! "${xcodebuild}" -checkFirstLaunchStatus >/dev/null 2>&1; then
    warn "Xcode's first-launch tasks have not run yet. If the build fails, run: sudo ${xcodebuild} -runFirstLaunch"
fi

# --- 2. Make sure `metal` actually runs --------------------------------------
# Probe with the same invocation MLX uses, so anything that satisfies this check
# satisfies make_compiled_preamble.sh too. Presence on disk is not enough: the
# Xcode 16+ `metal` binary exists but refuses to run until its asset is
# installed, which is exactly the case this script exists to repair.
metal_probe_output=""
metal_probe() {
    metal_probe_output="$(/usr/bin/xcrun -sdk macosx metal --version 2>&1)" && return 0
    return 1
}

if metal_probe; then
    log "Metal compiler ready: $(printf '%s\n' "${metal_probe_output}" | head -n 1)"
    if [ "${print_developer_dir}" = "1" ]; then
        printf '%s\n' "${developer_dir}"
    fi
    exit 0
fi

case "${metal_probe_output}" in
    *"license"*|*"License"*)
        die "the Xcode license has not been accepted yet. Run:
    sudo ${xcodebuild} -license accept
Then re-run the build. (xcrun said: ${metal_probe_output})"
        ;;
    *"Metal Toolchain"*|*"MetalToolchain"*|*"unable to find utility"*)
        : # A missing/stubbed compiler -- exactly what the download below fixes.
        ;;
    *)
        die "the Metal compiler is present but failed to run, and this is not a
missing-component failure that can be repaired automatically. xcrun said:
${metal_probe_output}"
        ;;
esac

component_status="$("${xcodebuild}" -showComponent MetalToolchain 2>/dev/null | sed -n 's/^Status: *//p' | head -n 1 || true)"
log "the Metal compiler is not installed (Xcode ships it as a downloadable component; status: ${component_status:-unknown})"
log "downloading it via 'xcodebuild -downloadComponent MetalToolchain' -- about 700 MB, one time, no sudo required..."

if ! run_logged "${xcodebuild}" -downloadComponent MetalToolchain; then
    die "downloading the Metal toolchain failed.
    Retry manually with:
      DEVELOPER_DIR='${developer_dir}' '${xcodebuild}' -downloadComponent MetalToolchain
    If this Xcode predates the downloadable Metal toolchain (Xcode 15 and
    earlier bundled it), the compiler is missing for another reason -- check
      DEVELOPER_DIR='${developer_dir}' xcrun -sdk macosx metal --version"
fi

if ! metal_probe; then
    die "the Metal toolchain reported a successful download but 'xcrun -sdk macosx metal --version' still fails:
${metal_probe_output}
    Inspect the component with:
      '${xcodebuild}' -showComponent MetalToolchain"
fi

log "Metal compiler installed: $(printf '%s\n' "${metal_probe_output}" | head -n 1)"

if [ "${print_developer_dir}" = "1" ]; then
    printf '%s\n' "${developer_dir}"
fi
