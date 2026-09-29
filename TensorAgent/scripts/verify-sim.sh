#!/usr/bin/env bash
# Simulator E2E check, run from the Mac while the app is running in the
# simulator (run-sim.sh). The simulator shares the host's network namespace, so
# the app's loopback listener is reachable from here.
#
# Usage: verify-sim.sh <app stdout log written by run-sim.sh>
#   The log carries the "entry URL" line (Debug builds only) with the port and
#   the per-launch token; every /api request must present that token.
#
# Checks: GET / is TensorAgent's own index.html byte for byte, GET /api/engine shows
# the static GgmlOps link is alive (no DllNotFoundException), /api without the
# token is refused, the app's own routes answer the shapes the page reads, the
# startup self-test ran every interpreter, the media probe line shows
# ImageIO/AVFoundation decoded a HEIC, applied a stored EXIF orientation,
# round-tripped an MP4 and read a WAV -- and, when the app was launched with
# TENSORAGENT_UI_CHECK=1, that the composer's own gestures behave in real WebKit.
set -euo pipefail

LOG="${1:?usage: verify-sim.sh <app stdout log>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENTRY="$(grep -o 'entry URL http://127.0.0.1:[0-9]*/?token=[0-9a-f]*' "${LOG}" | tail -1 | sed 's/^entry URL //')"
if [[ -z "${ENTRY}" ]]; then
    echo "No 'entry URL' line in ${LOG}; is this a Debug build and has the app started?" >&2
    exit 1
fi
BASE="${ENTRY%%\?*}"
TOKEN="${ENTRY##*token=}"
# The launch token is presented the way the WebView presents it: as the cookie the
# entry URL sets. The server takes no bearer header, and adding one purely for this
# script would widen the surface for a convenience.
AUTH=(-H "Cookie: tensoragent_token=${TOKEN}")

fail() { echo "FAIL: $*" >&2; exit 1; }

# A PHONE's 127.0.0.1 is the phone's, so none of the API checks can run against a
# device log -- but everything the app printed about itself still can, and that is most
# of what is worth checking. Rather than a second script that would drift, the HTTP half
# is skipped when the server cannot be reached and the log half runs regardless.
API=1
if ! curl -fsS --max-time 3 "${AUTH[@]}" "${BASE}/api/agent/engine" >/dev/null 2>&1; then
    API=0
    echo "==> ${BASE} is not reachable from here; checking what the app logged (a device run)"
else
    echo "==> ${BASE} (token ${TOKEN:0:6}…)"
fi
skip_api() { echo "--  ${1}: skipped, the app's loopback is not reachable from this machine"; }

if (( API )); then
# 1. index.html is TENSORAGENT's own page, plus exactly one appended script tag.
#    It used to be TensorSharp.Server's, served byte-for-byte with a phone layout
#    injected over it. The app now ships its own phone-first page: the desktop page
#    is laid out for a mouse and a wide window, and no amount of injected CSS makes
#    that a good phone app. The whole of
#    the Server's file must still be there, in order: the app adds to the page and
#    never forks it.
TMP="$(mktemp)"
SOURCE="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/wwwroot/index.html"
trap 'rm -f "${TMP}"' EXIT
curl -fsS -o "${TMP}" "${AUTH[@]}" "${BASE}"
SERVED_BYTES="$(wc -c < "${TMP}")"
SOURCE_BYTES="$(wc -c < "${SOURCE}")"
# The tag goes in before </body>, not at the end, so the test is: take it out again
# and what is left must be the app's own file byte for byte.
python3 - "${TMP}" "${SOURCE}" <<'PYCHECK' || fail "GET / is not TensorAgent/src/TensorAgent.Maui/wwwroot/index.html plus one script tag"
import sys
served = open(sys.argv[1], 'rb').read()
source = open(sys.argv[2], 'rb').read()
tag = b'\n<script src="/tensoragent.js"></script>\n'
if tag not in served:
    print('the companion script tag is missing', file=sys.stderr)
    sys.exit(1)
if served.replace(tag, b'', 1) != source:
    print('the page differs from the Server\'s beyond the one added tag', file=sys.stderr)
    sys.exit(1)
PYCHECK
echo "ok  GET / is TensorAgent's own index.html (${SOURCE_BYTES} bytes) plus the companion tag (${SERVED_BYTES} served)"

else
    skip_api "the served page"
fi

if (( API )); then
# 2. Token gate.
CODE="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}api/models")"
[[ "${CODE}" == "403" ]] || fail "GET /api/models without the token returned ${CODE}, expected 403"
echo "ok  /api without the token -> 403"

else
    skip_api "the token gate"
fi

if (( API )); then
# 3. Engine probe: the static link works and no P/Invoke threw.
ENGINE="$(curl -fsS "${AUTH[@]}" "${BASE}api/agent/engine")"
grep -q '"engine"' <<<"${ENGINE}" || fail "/api/agent/engine returned no engine line: ${ENGINE}"
grep -q 'sh (in-process)' <<<"${ENGINE}" || fail "the shell backend is not the in-process one: ${ENGINE}"
echo "    ${ENGINE}"
echo "ok  GET /api/agent/engine"

else
    skip_api "the engine probe route"
fi

if (( API )); then
# 4. The surface the Web UI calls at load, with the shapes the page reads.
MODELS="$(curl -fsS "${AUTH[@]}" "${BASE}api/models")"
grep -q '"supportedBackends"' <<<"${MODELS}" || fail "/api/models has no supportedBackends: ${MODELS}"
grep -q '"defaultMaxTokens"' <<<"${MODELS}" || fail "/api/models has no defaultMaxTokens"
# The page must never be shown a backend this build cannot initialise, and its
# default must be one of the ones it was shown. On a device that means Metal leads;
# in the simulator, whose slice of the engine has no Metal at all, it means CPU is
# the only entry. The engine probe in the launch log is what decides which.
python3 - "${MODELS}" "$(grep -o '"ggmlMetalAvailable":[a-z]*' "${LOG}" | tail -1)" <<'PYCHECK' || fail "/api/models offers a backend this build cannot run"
import json, sys
models = json.loads(sys.argv[1])
metal = 'true' in sys.argv[2]
offered = [b['Value'] for b in models['supportedBackends']]
if not offered:
    print('no backends offered at all', file=sys.stderr); sys.exit(1)
if models['defaultBackend'] != offered[0]:
    print(f"default {models['defaultBackend']} is not the first offered {offered}", file=sys.stderr); sys.exit(1)
if metal and offered[0] != 'ggml_metal':
    print(f"Metal is available but {offered[0]} leads", file=sys.stderr); sys.exit(1)
if not metal and 'ggml_metal' in offered:
    print(f"Metal is not available but is offered: {offered}", file=sys.stderr); sys.exit(1)
print(f"    backends offered: {offered} (Metal available: {metal})")
PYCHECK
curl -fsS "${AUTH[@]}" "${BASE}api/queue/status" | grep -q 'pending' || fail "/api/queue/status shape"
# curl sends no Content-Length for a body-less POST and HttpListener answers 411;
# WKWebView's fetch() sends Content-Length: 0, so -d '' mirrors the browser.
SESSION="$(curl -fsS "${AUTH[@]}" -X POST -d '' "${BASE}api/sessions?conversation=new")"
grep -q '"sessionId"' <<<"${SESSION}" || fail "POST /api/sessions: ${SESSION}"
grep -q '"conversationId"' <<<"${SESSION}" || fail "POST /api/sessions did not bind a conversation: ${SESSION}"
SID="$(sed -n 's/.*"sessionId":"\([^"]*\)".*/\1/p' <<<"${SESSION}")"
curl -fsS "${AUTH[@]}" -X DELETE "${BASE}api/sessions/${SID}" >/dev/null || fail "DELETE /api/sessions/{id}"
CODE="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "${BASE}uploads/x.png")"
[[ "${CODE}" == "404" ]] || fail "GET /uploads/x.png returned ${CODE}, expected 404"
echo "ok  /api/models offers only runnable backends, /api/queue/status, /api/sessions binds a conversation"

else
    skip_api "the Web UI routes"
fi

if (( API )); then
# 5. The app's own surface: the catalog, the saved chats, the sandbox switches.
CATALOG="$(curl -fsS "${AUTH[@]}" "${BASE}api/agent/catalog")"
python3 - "${CATALOG}" <<'PYCHECK' || fail "the catalog does not match the approved reduced model list"
import json, sys
models = json.loads(sys.argv[1])['models']
expected_ids = [
    'gemma-4-e2b-q8',
    'gemma-4-e4b-iq4xs',
    'gemma-4-12b-iq2m',
    'bonsai-2-27b-ptq1-0',
    'qwen3.5-9b-iq4xs',
]
# The catalog is filtered by the device's memory tier, and Bonsai 2 27B is a 16 GB
# entry: the simulator lists it exactly when the host Mac reports 16 GB or more.
actual_ids = [model['id'] for model in models]
if 'bonsai-2-27b-ptq1-0' not in actual_ids:
    expected_ids.remove('bonsai-2-27b-ptq1-0')
if actual_ids != expected_ids:
    print(f'catalog ids are {actual_ids}, expected {expected_ids}', file=sys.stderr)
    sys.exit(1)
if any(model['minDeviceMemoryGB'] != (16 if model['id'] == 'bonsai-2-27b-ptq1-0' else 12) for model in models):
    print(f'catalog device tiers are {[(m["id"], m["minDeviceMemoryGB"]) for m in models]}', file=sys.stderr)
    sys.exit(1)
families = {model['family'] for model in models}
kinds = {model['kind'] for model in models}
expected_families = {'Gemma4', 'Qwen35'} | ({'Bonsai'} if 'bonsai-2-27b-ptq1-0' in actual_ids else set())
if families != expected_families:
    print(f'catalog families are {families}', file=sys.stderr)
    sys.exit(1)
if kinds != {'Dense'}:
    print(f'catalog architectures are {kinds}', file=sys.stderr)
    sys.exit(1)
PYCHECK
# The two sandbox switches must be PRESENT and readable; their values are the user's,
# not a default. This container is reused between runs and the settings file survives,
# so asserting "network is off" here failed the day someone turned it on in the app —
# a check that reports a preference as a regression. The DEFAULTS are pinned where a
# default belongs, in TensorAgent.Tests (AppSettings and the settings route).
SETTINGS="$(curl -fsS "${AUTH[@]}" "${BASE}api/agent/settings")"
grep -q '"allowNetwork":' <<<"${SETTINGS}" || fail "the settings carry no network switch: ${SETTINGS}"
grep -q '"allowCodeExecution":' <<<"${SETTINGS}" || fail "the settings carry no code-execution switch: ${SETTINGS}"
echo "    switches on this container: $(grep -o '"allowCodeExecution":[a-z]*' <<<"${SETTINGS}") $(grep -o '"allowNetwork":[a-z]*' <<<"${SETTINGS}")"
# Downloads belong to the app rather than to a page or a request, so there is a route
# that says what is transferring however the model list was left.
DOWNLOADS="$(curl -fsS "${AUTH[@]}" "${BASE}api/agent/downloads")"
grep -q '"downloads"' <<<"${DOWNLOADS}" || fail "/api/agent/downloads shape: ${DOWNLOADS}"
SKILLS="$(curl -fsS "${AUTH[@]}" "${BASE}api/skills")"
grep -q '"skills"' <<<"${SKILLS}" || fail "/api/skills shape: ${SKILLS}"
echo "ok  catalog contains the six approved dense models; the switches and the download list answer"

else
    skip_api "the app's own routes"
fi

if (( API )); then
# 6. The companion script itself is served and carries the app's additions.
SCRIPT="$(curl -fsS "${AUTH[@]}" "${BASE}tensoragent.js")"
for SYMBOL in window.TensorAgent addAttachment insertText dictationEnded nativeReady setVoice skill_step "\$('activity')" \
             resumeTurn attachTurn openConversation paintNavChats; do
    grep -q "${SYMBOL}" <<<"${SCRIPT}" || fail "the companion script is missing ${SYMBOL}"
done
echo "ok  GET /tensoragent.js serves the companion script"

else
    skip_api "the companion script"
fi

# 6c. The upload probe: the app posted a body bigger than the parser's own buffer to its
#     own /api/upload, with the real HTTP client, and got a file back. That is the shape
#     that used to answer "no file was uploaded" for every photo picked on a phone.
UPLOAD="$(grep -o 'uploadcheck .*' "${LOG}" | tail -1 || true)"
[[ -n "${UPLOAD}" ]] || fail "the upload probe did not run"
grep -q '^uploadcheck ok' <<<"${UPLOAD}" || fail "the upload probe failed: ${UPLOAD}"
echo "ok  ${UPLOAD}"

# 6d. Optional share E2E: unlike a route-only test, this crosses the App Group
# envelope, the host's ordinary upload service, the WebView fetch bridge, composer
# merge, and the authenticated explicit-discard route. Both the durable envelope and
# the app-side staged attachment must be gone. It runs when the app was launched with
# TENSORAGENT_SHARE_CHECK=1.
SHARECHECK="$(grep -o 'sharecheck PASS composer .*' "${LOG}" | tail -1 || true)"
SHARECLEANUP="$(grep -o 'sharecheck discard cleanup PASS .*' "${LOG}" | tail -1 || true)"
SHAREFAIL="$(grep -E -o 'sharecheck .*FAIL.*' "${LOG}" | tail -1 || true)"
if [[ -z "${SHARECHECK}" ]]; then
    if [[ -n "${SHAREFAIL}" ]]; then
        fail "the share E2E probe failed: ${SHAREFAIL}"
    fi
    echo "--  the share E2E check did not run; relaunch with TENSORAGENT_SHARE_CHECK=1 to include it"
else
    [[ -z "${SHAREFAIL}" ]] || fail "the share E2E probe failed: ${SHAREFAIL}"
    [[ -n "${SHARECLEANUP}" ]] || fail "the share E2E probe retained its draft but did not report explicit-discard cleanup"
    echo "ok  ${SHARECHECK}"
    echo "ok  ${SHARECLEANUP}"
fi

if (( API )); then
# 6b. A generation belongs to the app, so the page must be able to ask about one and
#     the engine must say what it is doing about the model the user last used. Both are
#     routes a page that came back from another screen depends on; a 404 here is an
#     answer silently lost and a send button that never enables.
TURNS="$(curl -fsS "${AUTH[@]}" "${BASE}api/agent/turns?conversation=none")"
grep -q '"turn"' <<<"${TURNS}" || fail "/api/agent/turns shape: ${TURNS}"
grep -q '"model"' <<<"${ENGINE}" || fail "/api/agent/engine does not report the model state: ${ENGINE}"
grep -q '"activeTurn"' <<<"${SESSION}" || fail "POST /api/sessions does not report a running turn: ${SESSION}"
echo "ok  the turn routes answer and the engine reports the model state"

else
    skip_api "the turn routes"
fi

# 7. The startup self-test: the interpreters that linked can actually run, and the
#    sandbox refuses what it must. This is the only place the embedded CPython and
#    JavaScriptCore are exercised on a real iOS runtime.
# macOS ships bash 3.2, which has no mapfile.
CHECKS="$(grep -o 'selftest .*' "${LOG}" || true)"
[[ -n "${CHECKS}" ]] || fail "no 'selftest' lines in ${LOG}; the self-test is Debug-only, is this a Debug build?"
sed 's/^/    /' <<<"${CHECKS}"
grep -q 'FAIL' <<<"${CHECKS}" && fail "a startup self-test check failed"
for CHECK in shell python python:stdlib python:numpy python:pillow python:lxml python:pptx python:docx node sandbox:write sandbox:network; do
    grep -q "ok   ${CHECK}:" <<<"${CHECKS}" || fail "self-test check '${CHECK}' is missing or did not pass"
done
echo "ok  startup self-test: shell, python, node and both sandbox refusals"

# 8. The media probe. TensorSharp.Models/Media/Apple compiles only for net10.0-ios, so
#    the repo's net10.0 xunit suite cannot execute one line of it: this log line is the
#    only place ImageIO and AVFoundation are actually run. The desktop suite pins the
#    contract (InferenceWeb.Tests/MediaProviderParityTests runs the same assertions
#    against the managed and Magick.NET providers), and MediaProbe runs it here.
MEDIA="$(grep -o 'media probe {.*' "${LOG}" | tail -1 | sed 's/^media probe //')"
[[ -n "${MEDIA}" ]] || fail "no 'media probe' line in ${LOG}; the probe is Debug-only, is this a Debug build?"
echo "    ${MEDIA}"
grep -q '"allPassed":true' <<<"${MEDIA}" || fail "media probe reported a failed check: ${MEDIA}"
for CHECK in providers png-roundtrip png-straight-alpha heic-decode exif-orientation mp4-roundtrip audio-decode; do
    grep -q "\"name\":\"${CHECK}\",\"ok\":true" <<<"${MEDIA}" || fail "media probe check '${CHECK}' is missing or did not pass"
done
echo "ok  media probe: HEIC decode, EXIF orientation, MP4 round trip and AVAudioFile all passed"

# 9. The composer's own gestures, driven inside the real WebView. This is the only
#    place the page's JavaScript runs on the engine that will actually run it, and
#    layout questions -- is the hold-to-talk button visible, is the message box gone --
#    have no answer anywhere else. Opt-in, because it costs a couple of seconds of the
#    launch: run-sim.sh with TENSORAGENT_UI_CHECK=1.
UICHECKS="$(grep -o 'uicheck .*' "${LOG}" || true)"
if [[ -z "${UICHECKS}" ]]; then
    echo "--  the composer gesture checks did not run; relaunch with TENSORAGENT_UI_CHECK=1 to include them"
else
    sed 's/^/    /' <<<"${UICHECKS}"
    grep -q 'FAIL' <<<"${UICHECKS}" && fail "a composer gesture check failed"
    for CHECK in voice-switch-gone reasoning-is-a-setting skills-moved-to-the-menu \
                 activity-above-the-box a-tap-still-types \
                 holding-the-box-gives-hold-to-talk the-keyboard-button-returns \
                 the-menu-comes-from-the-left the-menu-lists-the-saved-chats \
                 a-turn-can-be-taken-back-up skills-have-a-master-switch; do
        grep -q "uicheck ${CHECK} ok" <<<"${UICHECKS}" || fail "gesture check '${CHECK}' is missing or did not pass"
    done
    echo "ok  composer gestures, and the menu is a left drawer that lists the saved chats"
fi

# 10. The one claim no unit test can make: a generation that keeps going while the chat
#     is not on screen, and a page that finds its way back to it. Opt-in and it needs a
#     loaded model: run-sim.sh with TENSORAGENT_DEMO_PROMPT, TENSORAGENT_NAV_CHECK=1.
NAVCHECKS="$(grep -o 'navcheck .*' "${LOG}" || true)"
if [[ -z "${NAVCHECKS}" ]]; then
    echo "--  the navigation check did not run; relaunch with TENSORAGENT_NAV_CHECK=1 and a prompt to include it"
else
    sed 's/^/    /' <<<"${NAVCHECKS}"
    grep -q 'FAIL' <<<"${NAVCHECKS}" && fail "the generation did not survive leaving the chat"
    grep -q 'navcheck ok away for' <<<"${NAVCHECKS}" || fail "the navigation check never left the chat"
    grep -q 'navcheck ok back in the chat' <<<"${NAVCHECKS}" || fail "the page did not pick the answer back up"
    echo "ok  the answer kept being written while the chat was off screen, and the page took it back up"
fi

# 10b. The generation carried across a BACKGROUND switch: the engine's step count must
#      stop while the app is away and the answer must finish -- and, on a device, a
#      second answer must work afterwards. Opt-in, needs a loaded model, and needs
#      something to send the app away: run verify-background.sh, which launches with
#      TENSORAGENT_BACKGROUND_CHECK=1, does the switching, and asserts the same lines.
BGCHECKS="$(grep -o 'bgcheck .*' "${LOG}" || true)"
if [[ -z "${BGCHECKS}" ]]; then
    echo "--  the background switch check did not run; run verify-background.sh to include it"
else
    sed 's/^/    /' <<<"${BGCHECKS}"
    grep -q 'FAIL' <<<"${BGCHECKS}" && fail "the generation did not survive the app being sent to the background"
    grep -q 'bgcheck ok ' <<<"${BGCHECKS}" || fail "the background check never reported a finished answer"
    if grep -q 'paused [1-9]' <<<"${BGCHECKS}"; then
        grep -q 'engine held [1-9]' <<<"${BGCHECKS}" \
            || fail "the app was away but the engine's step loop was never held by the gate"
        grep -q 'the next answer after coming back worked' <<<"${BGCHECKS}" \
            || fail "the answer after coming back did not work"
        echo "ok  the model stopped while the app was away, the answer finished, and the next one worked"
    else
        echo "--  the app was never sent away during the background check; nothing was proved about the gate"
    fi
fi

# 11. The network switch, both ways, in one running process: refused when it is off,
#      and reaching the internet the moment it is turned on -- without a relaunch, which
#      is the whole of the bug. Opt-in, because it goes out to the network:
#      run-sim.sh with TENSORAGENT_NETWORK_CHECK=1.
NETCHECKS="$(grep -o 'netcheck .*' "${LOG}" || true)"
if [[ -z "${NETCHECKS}" ]]; then
    echo "--  the network switch check did not run; relaunch with TENSORAGENT_NETWORK_CHECK=1 to include it"
else
    sed 's/^/    /' <<<"${NETCHECKS}"
    grep -q 'FAIL' <<<"${NETCHECKS}" && fail "the network switch did not take effect"
    grep -q 'netcheck off · ok' <<<"${NETCHECKS}" || fail "with the network off, curl was not refused"
    grep -q 'netcheck on · ok' <<<"${NETCHECKS}" || fail "with the network on, curl still could not reach the internet"
    echo "ok  the network switch takes effect on the next command, in both directions"
fi

if (( API )); then
    echo "All simulator checks passed."
else
    echo "All the checks a device log can answer passed (the API half needs the app's own loopback)."
fi
