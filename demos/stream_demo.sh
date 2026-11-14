#!/usr/bin/env bash
# Replay a bag on /camera_raw and watch it in a matplotlib window.
#
#   ./demos/stream_demo.sh data/bags/rosbag2_alpha_033_2026-11-12_20-39-49_0.mcap
#   ./demos/stream_demo.sh                      # synthetic 6 fps bag, made on first run
#   ./demos/stream_demo.sh BAG --rate 2 --loop  # extra args go to the replay node
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

python -m glide_bridge.ros.replay "$BAG" --loop "$@" &
REPLAY=$!
trap 'kill $REPLAY 2>/dev/null; wait $REPLAY 2>/dev/null' EXIT
python -m glide_bridge.ros.viewer --plot data/plots/stream_timing.png
