#!/usr/bin/env bash
# Replay a bag onto the Glide camera topics and watch them in a matplotlib window:
# forward (up) camera left, down camera right; color on top, depth below. The bottom
# row is the affordance model: up color with the predicted heatmap (left) and its
# bird's-eye view from the up depth (right). Each panel updates at its own camera's rate.
#
#   ./demos/stream_demo.sh data/bags/rosbag2_alpha_033_2026-11-12_20-39-49_0.mcap
#   ./demos/stream_demo.sh                      # synthetic bag (up 6 fps, down 2 fps), made on first run
#   ./demos/stream_demo.sh BAG --rate 6                         # 6x speed
#   ./demos/stream_demo.sh BAG --streams up_color down_color    # color only (both nodes)
#
# Model checkpoint: newest data/runs/*/best.pt; pick one with AFFORDANCE_CKPT=path/best.pt,
# or AFFORDANCE=0 for the YOLO detection row instead.
#
# Close the window to stop. A timing report is saved to data/plots/stream_timing.png.
set -euo pipefail
cd "$(dirname "$0")/.."
source env/activate_ros.sh

BAG="${1:-data/bags/synthetic}"
shift || true
if [[ "$BAG" == data/bags/synthetic && ! -e "$BAG" ]]; then
    python demos/make_synthetic_bag.py "$BAG" --fps 6 --seconds 30
fi

# --streams applies to both nodes (what gets published and which panels to show);
# every other extra argument goes to the replay.
STREAMS=()
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == --streams ]]; then
        while ((i + 1 < ${#args[@]})) && [[ "${args[i + 1]}" != -* ]]; do
            STREAMS+=("${args[i + 1]}"); ((i += 1))
        done
    fi
done
VIEWER_ARGS=()
((${#STREAMS[@]})) && VIEWER_ARGS=(--streams "${STREAMS[@]}")
if [[ "${AFFORDANCE:-1}" != 0 ]]; then
    VIEWER_ARGS+=(--affordance ${AFFORDANCE_CKPT:+"$AFFORDANCE_CKPT"})
fi

python -m glide_bridge.ros.replay "$BAG" --loop "$@" &
REPLAY=$!
trap 'kill $REPLAY 2>/dev/null; wait $REPLAY 2>/dev/null' EXIT
python -m glide_bridge.ros.viewer --plot data/plots/stream_timing.png ${VIEWER_ARGS[@]+"${VIEWER_ARGS[@]}"}
