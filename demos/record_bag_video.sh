#!/usr/bin/env bash
# Record the camera window for one pass of a bag to a video (for slides and reports).
#
#   ./demos/record_bag_video.sh BAG [OUT.mp4] [replay options...]
#   ./demos/record_bag_video.sh data/bags/rosbag2_alpha_033_2026-11-14_18-38-21 \
#       data/videos/alpha033_11-14.mp4 --rate 6
#
# The viewer draws the window offscreen (--headless --record), so nothing pops up and
# the video doesn't depend on screen size. It starts first (YOLO load + warm-up), then
# the replay plays the bag once; when the replay ends the viewer is stopped and the
# video finished. Video time is wall time: at --rate 6 a ~1 fps bag shows ~6 fps.
# OpenCV here writes MPEG-4 Part 2; see the report in docs/ for converting to H.264.
set -euo pipefail
cd "$(dirname "$0")/.."
source env/activate_ros.sh

BAG="${1:?usage: $0 BAG [OUT.mp4] [replay options...]}"
shift
OUT="data/videos/$(basename "${BAG%.mcap}").mp4"
if (($#)) && [[ "$1" != -* ]]; then OUT="$1"; shift; fi
LOG="$(mktemp)"

python -m glide_bridge.ros.viewer --headless --record "$OUT" > "$LOG" 2>&1 &
VIEWER=$!
REPLAY=
cleanup() {
    if [[ -n "$REPLAY" ]]; then kill "$REPLAY" 2>/dev/null || true; fi
    kill -INT "$VIEWER" 2>/dev/null || true
    wait "$VIEWER" 2>/dev/null || true   # finishes the video file
    cat "$LOG"; rm -f "$LOG"
}
trap cleanup EXIT

# Wait until the viewer has subscribed (the model is loaded before that).
for _ in $(seq 120); do
    grep -q "subscribed to" "$LOG" && break
    kill -0 "$VIEWER" 2>/dev/null || { echo "viewer exited early"; exit 1; }
    sleep 0.5
done
sleep 1                                   # DDS discovery

python -m glide_bridge.ros.replay "$BAG" "$@" &
REPLAY=$!
wait "$REPLAY"
REPLAY=
sleep 1.5                                 # let the last frames reach the video
