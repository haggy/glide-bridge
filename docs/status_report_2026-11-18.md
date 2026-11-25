# Status report: Glide data pipeline (2026-11-18)

EC518 "Where Can I Sit?". This report covers getting Glide camera data off the robot
and into a ROS 2 perception pipeline on the laptop, with object detection running on it.
It has three parts, each meant to become slides:

1. [Videos](#1-videos-of-the-camera-window): the camera window replaying each data bag.
2. [Pipeline](#2-pipeline-built-so-far): what is built and how the parts connect.
3. [Problems](#3-problems-encountered): solved and unsolved, with data capture first.

---

## 1. Videos of the camera window

Each video is the viewer window replaying one bag from start to end. All media files are
in `data/videos/`, which git ignores, so they exist only on this laptop. Copy them into the
slides directly.

| Video | Bag | Playback | Length | What it shows |
| --- | --- | --- | --- | --- |
| `alpha033_2026-11-14_18-38-21_x6.mp4` | 11-14, 116 s, hallway and elevators | 6x | 21 s | Newest capture. Down camera at a steady ~6 fps. The forward camera stalls for up to 16 s at a time. |
| `alpha033_2026-11-12_20-39-49_x6.mp4` | 11-12, 179 s, indoor hallway | 6x | 31 s | Worst case. The forward camera has 13 frames in 3 minutes, with a 51 s gap. |
| `alpha033_2026-11-12_21-10-16_x2.mp4` | 11-12, 13 s, lab | 2x | 8 s | The only bag where both cameras are complete. Shows YOLO detections on the forward camera (chair, laptop, mouse). |

Each video has a matching `*_still.png` frame for a slide thumbnail or for a backup if a
video won't play.

**Window layout.** The left column is the forward (up) camera and the right column is the
downward camera. The rows are:

- color;
- depth, colored from 0 to 5 m, with black where there is no reading;
- the color frame with YOLO boxes drawn on it (`ObjectDetection.apply_bounding_boxes`).

Each panel title shows the frame number, the frame rate and the latency from publish to
display. A panel updates only when its own camera delivers a frame. Because of that, the
forward camera's dropouts show up in the video as frozen panels.

**Playback speed.** The bags hold about 1 frame per second per camera (see
[3.1.2](#312-bags-are-1-fps-not-6-fps)). At 6x they play at roughly the cameras'
capture rate of 6 fps. Say on the slide that the footage is sped up.

**Format.** The files are MP4 containers with MPEG-4 Part 2 video (`mp4v`), 1280x1200 at
15 fps. That is the best codec this machine's OpenCV can write. PowerPoint on Windows and
Google Slides (which re-encodes videos uploaded to Drive) play it. Keynote and some Mac
players want H.264. If you have ffmpeg, convert with:

```bash
ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart out.mp4
```

**Re-recording.** Videos come from the new `--record` option on the viewer, which can draw
the window offscreen. To record a bag:

```bash
./demos/record_bag_video.sh data/bags/<recording> data/videos/<name>.mp4 --rate 6
```

**Other figures you could use:**

| Figure | What it shows |
| --- | --- |
| `data/plots/color_depth_sync.png` | Why the forward camera loses frames ([3.1.1](#311-forward-camera-loses-most-of-its-frames)) |
| `data/plots/detection_check.png` | Detector check against Ultralytics' reference boxes on `bus.jpg`. All 5 boxes match. |
| `data/plots/stream_timing.png` | Viewer timing report: intervals between frames and latency for each stream |

---

## 2. Pipeline built so far

### 2.1 Overview

```
 ROBOT (Glidance alpha 2 "alpha033", ROS 2 Humble, Yocto)
   camera_up / camera_down drivers, 6 fps color + depth + IR
        |
   image_fps_reducer (Glidance) -> *_rfps topics, ~1 fps
        |
   rosbag2 -> MCAP bag on /data   (kickstand long-press, or ./glide_robot.sh bag start)
        |
   copied off the robot by hand -> project Google Drive "Data" folder
        |
 LAPTOP
   ./fetch_bags.sh pull           rclone, read-only Drive access -> data/bags/<recording>/
        |
   glide_bridge.ros.replay        ROS 2 node: reads MCAP, publishes on the robot's own topic
        |                         names, keeps each camera's original timing (--rate N speeds up)
        |  Cyclone DDS, localhost (same RMW as the robot)
        v
   /camera_up/color/image_raw, /camera_up/depth/image_raw, /camera_down/... (+ camera_info)
        |
   glide_bridge.ros.viewer        ROS 2 node: one subscriber per stream
        |-- color frame --> ObjectDetection (YOLO26n, Ultralytics, CUDA) --> boxes row
        |
        +--> matplotlib window, timing report (--plot), snapshot, video (--record)
```

The repo also has a second path, from the initial commit, that skips ROS on the laptop:
`robot/glide_link_relay.py` on the robot sends topics over TCP port 8770 to `view_cameras.py`
and `record.py`. It isn't connected to the ROS pipeline yet
([3.1.6](#316-no-live-path-into-the-ros-pipeline-yet)).

### 2.2 Components

| Part | File(s) | What it does |
| --- | --- | --- |
| Bag download | `fetch_bags.sh` | `list` / `pull` / `get <url>` from the Drive "Data" folder with rclone. Puts split bags into one folder per recording. |
| Topic contract | `glide_bridge/topics.py`, `glide_bridge/ros/__init__.py` | Streams use the robot's own topic names (`up_color` is `/camera_up/color/image_raw` plus its `camera_info`). Consumers work unchanged when the source changes from replay to the live robot. |
| MCAP reader | `glide_bridge/ros/mcap_source.py`, `glide_bridge/mcap_utils.py` | Finds each stream in a bag (raw, `_rfps` or `/compressed` form) and reads split bags in order. |
| Replay node | `glide_bridge/ros/replay.py` | Publishes every camera at its recorded timing. Re-stamps each frame with its publish time so latency can be measured. Options: `--rate`, `--loop`, `--keep-stamps`, `--list`. |
| Image conversion | `glide_bridge/ros/image_msgs.py` | Converts between ROS `Image`/`CompressedImage` and numpy arrays. Handles rgb/bgr, 16-bit depth and row padding. |
| Viewer node | `glide_bridge/ros/viewer.py` | Grid of one column per camera and one row per image kind. Each panel updates on its own. Measures frame rate and latency per stream. Options: `--plot`, `--snapshot`, `--record`, `--headless`. |
| Object detection | `src/img_processing/obj_detection.py` | `ObjectDetection.apply_bounding_boxes(rgb) -> rgb`. YOLO26n (COCO classes) via Ultralytics 8.4. It runs on every color frame from both cameras. |
| Demos and checks | `demos/stream_demo.sh`, `demos/record_bag_video.sh`, `demos/check_detection.py`, `demos/make_synthetic_bag.py` | Replay plus window; video recording; detector check on a known image; a synthetic two-camera bag for testing without real data. |
| Environments | `env/ros_humble.yml`, `env/activate_ros.sh` | Conda env `glide_ros`: ROS 2 Humble from RoboStack, Python 3.11 (matches the robot), numpy < 2, Ultralytics. Non-ROS tools use a separate uv venv on Python 3.12. |
| Debugging | `.vscode/launch.json`, `env/vscode_env.sh` | "Camera window + bag replay" launch config, so you can set breakpoints in the detector. |

### 2.3 Measured performance

All numbers are from replaying on this laptop (RTX 4070 Ti).

**Transport.** For color frames, publish to receive takes 1–3 ms. The replay keeps the
recorded timing to within 7–11 ms (worst lag over a pass). A synthetic test bag (up at
6 fps, down at 2 fps) plays back at exactly 6.00 and 2.00 fps.

**Detection.** About 20 ms per frame, so the detection panels lag the color panels by
about 20–25 ms. The detector runs on the viewer's only ROS callback thread. That also
delays depth frames by about 20–30 ms. It is fine at these frame rates, but moving
detection to a worker thread is an easy fix if needed.

**Detector check.** On Ultralytics' `bus.jpg`, all 5 boxes match the reference boxes.

**Observation for the slides.** The off-the-shelf COCO model sometimes labels the robot's
own white caster as a "toilet" in the down camera (visible in the 11-14 video). It does find
chairs in the lab bag. This is a reason for the project's own sittability model, rather than
relying on COCO classes.

### 2.4 Not built yet

- A live robot source feeding the same ROS topics (see
  [3.1.6](#316-no-live-path-into-the-ros-pipeline-yet)).
- Segmentation and the sittability model itself (`src/img_processing/segmentation.py` is
  a placeholder).
- Use of depth and camera intrinsics in perception. Both are published, but nothing uses
  them yet.

---

## 3. Problems encountered

**Status key:**

- **OPEN**: unsolved.
- **WORKAROUND**: we can work around it, but the cause is on the robot.
- **FIXED**: solved on our side.

### 3.1 Data capture and the platform

#### 3.1.1 Forward camera loses most of its frames (OPEN, cause partly understood)

Frames that reached the bags:

| Bag | Length | Forward color/depth frames | Down color/depth frames | IR, both cameras | Longest forward gap |
| --- | --- | --- | --- | --- | --- |
| 11-12 20-39-49 | 179 s | 13 (7%) | 172 (96%) | ~180 (100%) | 51 s |
| 11-12 21-10-16 | 13 s | 13 (97%) | 13 (97%) | 14 | 1.4 s |
| 11-14 18-38-21 | 116 s | 39 (34%) | 108 (93%) | 116 (100%) | 16 s |

What we found:

- **The camera isn't slower.** Both cameras run at 6 fps in the data-collection config,
  and both `camera_info` streams arrive at about 5.7–6 Hz. The IR streams, which aren't
  paired with anything, are complete for both cameras.
- **The drops happen in the robot's frame reducer.** It keeps a color/depth pair only
  when their timestamps are within about 10 ms. The largest offset among kept pairs is
  9.74 ms.
- **On the forward camera, color and depth don't line up.** Its depth timestamp often
  sits about 33 ms or 65 ms behind color. That is one or two frame periods at the
  camera's internal rate. The offset stays at one level for tens of seconds at a time, so
  those pairs are rejected. The down camera stays within about 1 ms.
- **This explains the counts exactly.** Predicted kept fraction = 1/6 × (fraction of
  frames whose color and depth are within 10 ms):

  | Camera, bag | Predicted kept | Observed kept |
  | --- | --- | --- |
  | Down, 11-14 | 16.2% | 16.2% |
  | Forward, 11-14 | 5.95% | 5.9% |
  | Forward, 11-12 | 1.28% | 1.3% |

  Plot: `data/plots/color_depth_sync.png`.

What's still open:

- **Why only the forward camera's depth lags.** It could be a driver frame-sync setting,
  USB bandwidth, or a different firmware or config for that camera. Both cameras are set
  to `sync_mode: standalone`.
- **The current reducer can't be checked.** The copy of the robot filesystem we have is
  from about May 2026. That predates the July change to a timestamp-matched reducer
  skipping 5 frames out of 6.
- **Checks to run on the robot:**
  - `ros2 topic hz` and the color/depth stamps on the full-rate
    `/camera_up/{color,depth}/image_raw`;
  - the live `glide_alpha2_dc.yaml` and the reducer it runs.
- **Possible fixes:**
  - widen the reducer's pairing tolerance to about 70 ms (color and depth could then be up
    to 2 frames apart);
  - record the full-rate topics;
  - fix the camera sync at its source.

#### 3.1.2 Bags are ~1 fps, not 6 fps (OPEN, by design of the robot config)

- The cameras capture at 6 fps, but the bags record the reduced `_rfps` topics at about
  1 frame/s per camera.
- The documents disagree. The Data Capture Guide and our README say about 2 fps.
  Measured: 0.94–1.0 fps.
- **Consequence:** few training frames per minute of driving. Motion between frames is
  large (about 1 m at walking speed), so anything that needs consecutive frames won't
  work well.
- **Options:**
  - lower the reducer's skip count for collection runs;
  - record full-rate topics;
  - stream over Wi-Fi with `record.py`, up to the camera rate, at the cost of Wi-Fi drops.

#### 3.1.3 Damaged / unfinalized bag (WORKAROUND)

- `rosbag2_alpha_033_2026-11-12_21-10-16_0.mcap` came off the robot unfinalized. It is a
  412 MB file with no MCAP footer and zeros after the first 17 MB, about 7 s of data.
- That part was deleted. Only `_1` (13 s) remains, and it is no longer on Drive, so the
  local copy is the only one.
- **Likely cause:** the bag was copied while still being written, or recording stopped
  without a clean shutdown.
- **Check before using any bag:** the file must end with the MCAP magic bytes
  `89 4d 43 41 50 30 0d 0a`.
- **Still needed:** a clean stop-then-copy procedure. `./glide_robot.sh pull-bags` only
  copies finished bags.

#### 3.1.4 Documentation doesn't match the robot (OPEN, docs not yet corrected)

Our README and `topics.py` say both cameras run at 6 fps in every config. That is wrong:

| Config | Forward camera | Down camera |
| --- | --- | --- |
| Normal (`glide_alpha2.yaml`) | 6 fps | 15 fps |
| Data collection (`glide_alpha2_dc.yaml`) | 6 fps | 6 fps |

- The bag rate is documented as ~2 fps; it is ~1 fps (3.1.2).
- Nothing in the Glidance docs or the project Drive mentions the forward-camera dropout.
- We are holding the doc fixes until the investigation in 3.1.1 is finished.

#### 3.1.5 Robot software snapshot is old and contains secrets (OPEN)

- The robot filesystem copy (`onboard_codes.zip`, 2.3 GB) is from about May. It doesn't
  match what ran for the November captures.
- It contains:
  - root SSH keys;
  - a GCP service-account key;
  - device certificates;
  - shell history;
  - Glidance's proprietary `/shared` code.
- So it can't go into git or onto a widely shared Drive folder. It is stored locally in
  `data/robot_fs/` (git-ignored) and was not re-uploaded.
- **Needed:**
  - a current, secrets-free copy of `/shared/glide_launchers` and the reducer from the
    robot;
  - a decision on where the full image may be stored.

#### 3.1.6 No live path into the ROS pipeline yet (OPEN)

- The robot's DDS traffic never leaves the robot (Cyclone, localhost only, domain 33).
  There is no rosbridge on the robot.
- The laptop pipeline therefore runs only from bags today.
- The TCP relay (`glide_link_relay.py`) can stream the topics over Wi-Fi to `record.py` /
  `view_cameras.py`. It has not been connected to the ROS topics on the laptop.
- **Next step:** a small laptop node that republishes relay frames on
  `/camera_*/*/image_raw`. The viewer and detector then work live with no changes.

#### 3.1.7 Capture-day procedure overhead (WORKAROUND)

Each capture needs these manual steps:

- check which config booted (`./glide_robot.sh info`);
- stop Glidance's cloud uploader (`./glide_robot.sh uploader stop`) so it doesn't take
  the bags;
- start the bag (kickstand long-press);
- copy the bags off the robot to Drive by hand.

The robot is on different subnets in rooms 209 and 506.

Any of these steps can fail silently: wrong config, uploader running, or a bag copied
early (3.1.3).

### 3.2 Getting data from Drive to the laptop

| Problem | Status | Notes |
| --- | --- | --- |
| The Google Drive connector can't download bags | FIXED | It returns file contents inline, and bags are 37–412 MB. We use rclone with a read-only Drive login instead (`fetch_bags.sh`). |
| rclone's shared Google client ID is being retired in 2026 | OPEN (future risk) | If downloads start failing with auth or quota errors, register our own client ID (rclone docs: "Making your own client_id"). |
| Bags deleted from Drive | OPEN | 21-10-16 exists only on this laptop. Drive is the only shared store, so keep a backup. |

### 3.3 Laptop pipeline and tooling (solved)

| Problem | Fix |
| --- | --- |
| ROS 2 Humble can't be installed with pip, and the robot runs Python 3.11 | Separate conda env `glide_ros` (RoboStack Humble, 3.11). Non-ROS tools stay in the uv venv (3.12). `env/activate_ros.sh` switches cleanly. |
| A uv venv left active made 3.12 load 3.11 numpy (ImportError) | The activate script deactivates the venv and removes it from PATH. |
| `CONDA_BUILD: unbound variable` in scripts using `set -u` | `set -u` is relaxed during conda activation. |
| Conda's ROS hook set `ROS_LOCALHOST_ONLY=0` | Now forced to `1`, like the robot. |
| Installing Ultralytics pulled in numpy 2 and OpenCV 5, which break Humble | Pins: `numpy<2`, `opencv-python<4.12`, checked with `pip install --dry-run`. |
| Viewer segfaulted once torch was loaded (Mac) | Two OpenMP runtimes (conda OpenBLAS and torch). Fixed by pinning the pthreads build of OpenBLAS. |
| The conda CLI crashes while `glide_ros` is active | Run conda from a shell where `glide_ros` is not active. |
| Detector boxes in the wrong place: coordinates above 255 wrapped around, and colors were swapped | Box coordinates cast to `int32` instead of `uint8`. Frames converted RGB to BGR for Ultralytics. Verified with `demos/check_detection.py`. |
| "Every bag shows the same stream" | A leftover replay process was still publishing an old bag on the same topics. It was stopped. Before debugging, check with `ps -eo pid,args \| grep glide_bridge.ros.replay`. |
| Replay and viewer printed tracebacks on Ctrl-C | Both now shut down cleanly. |

---

## Suggested slide outline

1. **Goal and data source:** the Glide's two RGB-D cameras, and capturing bags on the robot.
2. **Pipeline diagram** (2.1). One line each on replay, the topic contract, the viewer and the detector.
3. **Demo video:** the 11-12 21-10-16 bag for detections, and the 11-14 bag for the dropouts.
4. **What works:** the numbers in 2.3.
5. **Data capture problems:** 3.1.1, with the timing plot and the kept-fraction table.
6. **Other open issues:** 3.1.2–3.1.6, one line each.
7. **Next steps:**
   - robot-side timestamp check and reducer settings;
   - full-rate or higher-rate collection;
   - live relay into ROS;
   - sittability model.
