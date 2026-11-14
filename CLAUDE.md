# glide-bridge: working rules

Project context: EC518 final project, "Where Can I Sit?" (instruction-conditioned
sittability affordances for the Glide mobility aid). This repo moves Glide camera and
telemetry data to a laptop and, via ROS 2, into the perception pipeline.

## Code quality

- **Type annotations on every function signature**, including `__init__`, private
  helpers, callbacks and `-> None`. We own all the code here: if you touch a function
  without annotations, add them.
- **Annotate variables whose type isn't obvious** from the right-hand side, especially
  array-like values: `img: np.ndarray = ...` vs `img: torch.Tensor = ...`, dtype/shape
  in a trailing comment when it matters (`# (H, W, 3) uint8 RGB`).
- **numpy-style docstrings** (`Parameters`, `Returns`, `Raises`, `Notes`, `Examples`
  sections). Put math in `Notes` with plain-text or LaTeX-ish formulas, e.g.
  `depth_m = raw_mm / 1000`, and state units (m, mm, s, ns) and frame conventions.
- Files keep `from __future__ import annotations` so annotations work on the robot's
  Python 3.11 and the laptop's 3.12.

## Testing preference

- For now, prefer **runnable, visual checks** over unit tests: scripts that stream
  frames live ("video"), save matplotlib plots (timing, histograms, overlays), or print
  a short summary you can eyeball. Put these under `demos/` or as `--plot`/`--snapshot`
  options on the tool itself, and say in the docstring how to run them.
- Unit tests (`tests/`, pytest) will be written later; don't add new ones unless asked.
  Keep the existing ones passing.

## Environments

- `uv` venv (`.venv`, Python 3.12): non-ROS tools (`view_cameras.py`, `record.py`,
  `glide_bridge` client code). No ROS needed.
- Conda env `glide_ros` (`env/ros_humble.yml`, ROS 2 Humble from RoboStack, Python
  3.11): everything under `glide_bridge/ros/` and the ROS scripts. Humble matches the
  robot. Activate with `source env/activate_ros.sh`.
- Never import `rclpy` from modules the uv env loads; ROS code lives in
  `glide_bridge/ros/` only.

## Getting data (bags from Google Drive)

- Use `./fetch_bags.sh` (rclone, binary at `~/.local/bin/rclone`, remote `glide_gdrive`,
  read-only scope) to download MCAP bags into `data/bags/`. Don't try the Google Drive
  connector's download tool for bags: it returns file content inline as base64, and
  bags are hundreds of MB.
  - `./fetch_bags.sh list` / `pull [name-substring]` / `get <Drive URL or file id>`.
  - Default source is the project Drive folder "Data"
    (id `1TNXE_c0k8CZVG0gO_6XvGgu1yogj4COo`); override with `GLIDE_DRIVE_FOLDER`.
  - If it says the remote is missing or the token expired, the user must run
    `./fetch_bags.sh setup` themselves (browser sign-in), e.g. `! ./fetch_bags.sh setup`.
- The Drive connector is still fine for metadata/search (finding new bag file ids).
- The `glide_gdrive` remote uses rclone's shared Google client_id, which rclone says is
  being retired during 2026. If downloads start failing with auth/quota errors, the fix
  is the user's own client_id (https://rclone.org/drive/#making-your-own-client-id),
  then `rclone config update glide_gdrive client_id=... client_secret=...` + `setup`.
- Known bag: `rosbag2_alpha_033_2026-11-12_20-39-49_0.mcap` (180 s indoor hallway,
  640x360). All cameras recorded at ~1 fps (`*_rfps` topics), and the up (forward)
  color camera dropped out (13 frames); the down camera has 172 frames.

## ROS conventions

- Camera topics are the **robot's own names** from `PRESETS["glide"]` in
  `glide_bridge/topics.py`: e.g. `up_color` (forward camera) on
  `/camera_up/color/image_raw` + `/camera_up/color/camera_info`, `down_color` on
  `/camera_down/...`. The topic identifies the camera. Sources (MCAP replay now, live
  Glide later) publish there, and consumers subscribe by stream name, so moving to the
  robot needs no remapping. Bag topics like `..._rfps` are republished without the suffix.
  (This replaced an earlier single `/camera_raw` topic.)
- Streams are independent: different rates and dropouts per camera are normal; never
  assume frames from two cameras arrive paired.
- Sensor QoS: best effort, keep last, small depth, so slow consumers drop frames
  instead of queueing them.
- Data files (`data/`, `*.mcap`) stay out of git.
