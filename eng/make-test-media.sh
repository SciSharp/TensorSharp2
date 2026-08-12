#!/usr/bin/env bash
# Generate the image / audio / video fixtures TensorSharp.TestMatrix's
# multimodal cells need.
#
# The media files are deliberately NOT committed (they are megabytes of binary
# that would bloat every clone), so the matrix ships the prompts and this
# script instead. Point the matrix at the output with `--media-dir`, or set
# `media_dir` in Defaults/matrix-config.json.
#
#   eng/make-test-media.sh ~/work/models/testmedia
#
# Produces, matching Matrix/FeatureCatalog.cs's MediaFile / ExpectedContains:
#   image.png   the repo banner            -> the model must say "tensorsharp"
#   sample.wav  spoken pangram, 16 kHz     -> the model must say "fox"
#   sample.mp4  slow zoom on the banner    -> the model must say "tensorsharp"
#
# Requirements:
#   image  none (copied from imgs/)
#   audio  macOS `say` + `afconvert`, else `espeak` + `ffmpeg`
#   video  `ffmpeg`
# Anything missing is reported and skipped; the matrix fails those cells with a
# clear "media file not found" rather than silently passing.
set -uo pipefail

OUT="${1:-}"
if [ -z "$OUT" ]; then
    echo "usage: $0 <output-dir>" >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"

SENTENCE="The quick brown fox jumps over the lazy dog near the river bank."
rc=0

# ---- image -----------------------------------------------------------------
if cp "$REPO_ROOT/imgs/banner_1.png" "$OUT/image.png" 2>/dev/null; then
    echo "image.png  <- imgs/banner_1.png"
else
    echo "image.png  SKIPPED (imgs/banner_1.png missing)" >&2
    rc=1
fi

# ---- audio -----------------------------------------------------------------
# 16 kHz mono PCM is what every audio encoder in the repo resamples to anyway,
# so emitting it directly keeps the fixture free of resampler differences.
if command -v say >/dev/null 2>&1 && command -v afconvert >/dev/null 2>&1; then
    say -o "$OUT/.say.aiff" "$SENTENCE" \
        && afconvert -f WAVE -d LEI16@16000 -c 1 "$OUT/.say.aiff" "$OUT/sample.wav" \
        && rm -f "$OUT/.say.aiff" \
        && echo "sample.wav <- say + afconvert"
elif command -v espeak >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
    espeak -w "$OUT/.espeak.wav" "$SENTENCE" \
        && ffmpeg -y -loglevel error -i "$OUT/.espeak.wav" -ar 16000 -ac 1 "$OUT/sample.wav" \
        && rm -f "$OUT/.espeak.wav" \
        && echo "sample.wav <- espeak + ffmpeg"
else
    echo "sample.wav SKIPPED (need macOS say+afconvert, or espeak+ffmpeg)" >&2
    rc=1
fi

# ---- video -----------------------------------------------------------------
# A slow zoom on the banner rather than an abstract synthetic shape. The first
# version of this fixture was a red square panning across a white field, which
# reached the model correctly (frames extracted, 3 x 48 vision tokens injected)
# and still failed the cell: Gemma 4 answered "no actual video content has been
# provided ... which appear to be corrupted". A fixture the model cannot
# describe cannot answer the question the cell asks — "did the video reach the
# model" — so it is legible content now, and the assertion keys off the banner
# title exactly as the image cell does.
#
# libx264 + yuv420p is the combination OpenCV's bundled decoder reads
# everywhere; the OpenCV build vendored with this repo decodes video but has no
# encoder, which is why this step needs ffmpeg.
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -loop 1 -i "$REPO_ROOT/imgs/banner_1.png" -t 3 -r 6 \
        -vf "scale=640:-2,zoompan=z='min(zoom+0.0012,1.12)':d=18:\
x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=640x424,format=yuv420p" \
        -c:v libx264 -preset veryfast "$OUT/sample.mp4" \
        && echo "sample.mp4 <- ffmpeg zoompan on the banner"
else
    echo "sample.mp4 SKIPPED (ffmpeg not on PATH)" >&2
    rc=1
fi

echo
ls -la "$OUT"
exit $rc
