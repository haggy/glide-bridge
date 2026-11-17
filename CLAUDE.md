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
- `glide_ros` must keep **numpy < 2** (RoboStack Humble is built against 1.x). When
  adding pip packages there (e.g. ultralytics), pin `numpy<2` and any dep that would
  force numpy 2 (`opencv-python<4.12`); check with `pip install --dry-run` first.
- `glide_ros` pins the **pthreads build of OpenBLAS** (`libopenblas=*=*pthreads*`). The
  default OpenMP build loads conda's libomp with numpy, pip torch loads its own OpenMP,
  and the two segfault together (it shows up as the viewer crashing once detection
  runs). If a segfault appears with torch, check `OMP_NUM_THREADS=1` first: if that
  fixes it, it's duplicate OpenMP runtimes again.
- VS Code debugging of ROS code: `.vscode/launch.json` ("Camera window + bag replay")
  runs under `glide_ros` via `env/vscode_env.sh`, which exports the activated env to
  `.vscode/ros.env` + `.vscode/ros-python` (git-ignored, regenerated each launch).
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
- Layout: `data/bags/<recording>/<recording>_N.mcap` (rosbag2 splits long recordings;
  a bag is the folder). Pass the folder to the replay to play all parts in order, or one
  file to play just that part. `fetch_bags.sh` files downloads this way.
- Known bags (both 640x360, all cameras ~1 fps as `*_rfps` topics):
  - `rosbag2_alpha_033_2026-11-12_20-39-49` (1 file, 180 s indoor hallway): the up
    (forward) color camera dropped out (13 frames); the down camera has 172.
  - `rosbag2_alpha_033_2026-11-12_21-10-16`: only `_1` kept (13.7 s, 13 frames/camera,
    all cameras intact). `_0` was damaged (never finalized, mostly zero bytes) and was
    deleted locally. As of 2026-11-14 neither part is on Drive any more; the local `_1`
    is the only copy.
- Forward-camera frames missing from bags (2026-11-17 finding, still being investigated):
  both cameras run at 6 fps (`camera_info` ~5.7-6 Hz each). The on-robot reducer keeps a
  color/depth pair only when their stamps are within ~10 ms. The up camera's depth stamp
  often sits ~33 or ~65 ms behind color for long stretches, so those pairs are dropped.
  The down camera stays within ~1 ms. Plot: `data/plots/color_depth_sync.png`. The robot
  filesystem copy (`data/robot_fs/onboard_codes.zip`, from Jiatong, contains secrets) is
  from ~May 2026, before the July collection-config changes.
- Bags can come off the robot unfinalized. A file whose last 8 bytes aren't the MCAP
  magic (`89 4d 43 41 50 30 0d 0a`) is incomplete; check before debugging the replay.

## ROS conventions

- Camera topics are the **robot's own names** from `PRESETS["glide"]` in
  `glide_bridge/topics.py`: e.g. `up_color` (forward camera) on
  `/camera_up/color/image_raw` + `/camera_up/color/camera_info`, `down_color` on
  `/camera_down/...`. The topic identifies the camera. Sources (MCAP replay now, live
  Glide later) publish there, and consumers subscribe by stream name, so moving to the
  robot needs no remapping. Bag topics like `..._rfps` are republished without the suffix.
  (This replaced an earlier single `/camera_raw` topic.)
- Only one source should publish on the camera topics at a time. Two replays (or a
  replay plus the robot) on the same `ROS_DOMAIN_ID` interleave their frames in the
  viewer. Before debugging a "wrong images" report, check
  `ps -eo pid,args | grep glide_bridge.ros.replay` for leftover replays.
- Streams are independent: different rates and dropouts per camera are normal; never
  assume frames from two cameras arrive paired.
- Sensor QoS: best effort, keep last, small depth, so slow consumers drop frames
  instead of queueing them.
- Data files (`data/`, `*.mcap`) stay out of git.
