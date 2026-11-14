# local_glide_bridge

Stream camera and telemetry data from the Glide robot (Glidance alpha 2,
`alpha033`) to this laptop, view it live, and record it for training. The same
client also works against a simulator.

```
 Glide robot (ROS 2 Humble, domain 33)              laptop (no ROS needed)
  /camera_up/*, /camera_down/*, /odom, ...
        |  Cyclone DDS, robot-local only
  glide_link_relay.py  (/data/glide_link)  ==TCP 8770 over Wi-Fi==>  view_cameras.py
        JPEG color, 16-bit PNG depth (mm),                            record.py
        PNG IR/labels, JSON for everything else                       glide_bridge.GlideLink
```

The robot runs Yocto Linux with no package manager and no rosbridge, and its DDS
traffic never leaves the robot. The relay is one Python file that uses only what
the robot already has (rclpy, OpenCV, numpy), runs alongside Glidance's
`glide_launch.service` without changing it, and serves the topics a client asks for.

## Setup (laptop)

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12; see
`pyproject.toml` for why some dependencies are capped on Intel Macs.

```bash
uv sync
uv run pytest -q
```

## Capturing data from the robot

1. **Join the robot's network.** The robot connects to the Wi-Fi network "Glide" at
   boot. Room 209: robot at `10.42.0.45` (the default here). Room 506: `192.168.0.184`,
   so `export GLIDE_HOST=192.168.0.184` first.

2. **Optional: skip password prompts.** `ssh-copy-id root@$GLIDE_HOST`. Otherwise the
   helper asks once and reuses the connection for 10 minutes.

3. **Check the robot and stop the cloud uploader** (Glidance's service that uploads
   bags to their cloud):
   ```bash
   ./glide_robot.sh info            # boot config, service states, disk space
   ./glide_robot.sh uploader stop
   ```
   For training captures use the data-collection config (`glide_alpha2_dc.yaml`:
   640x360 color with depth registered to color, 6 fps). `info` shows which config
   booted. The normal config gives 424x240 color and 480x270 depth.

4. **Deploy and start the relay** (first time, and after any change to `robot/`):
   ```bash
   ./glide_robot.sh deploy
   ./glide_robot.sh start           # logs: ./glide_robot.sh log
   ```

5. **Look, then record:**
   ```bash
   uv run python view_cameras.py                                  # up/down RGB-D
   uv run python view_cameras.py --streams up_color segmentation
   uv run python record.py                                        # Ctrl-C to stop
   uv run python record.py --streams all --duration 300 --name hallway_01
   ```
   The viewer and recorder can run at the same time. The recorder reconnects on its
   own if the Wi-Fi drops and keeps writing the same session.

6. **When done:** `./glide_robot.sh stop`, and `./glide_robot.sh uploader start` if
   the lab wants the uploader back on.

### What gets recorded

By default `record.py` records four camera streams (`up_color`, `up_depth`,
`down_color`, `down_depth`), every frame, plus each camera's `camera_info` at 1 Hz
and the telemetry topics `/odom`, `/imu`, `/gps/fix`, `/gps/velocity`,
`/glide_mode`, `/inside_outside/is_inside`, `/brake_ratio`, `/wheels_headings`,
`/camera_*/depth_to_color` and `/tf_static`. Topics that aren't published are
reported once and recorded if they appear later.

```
data/sessions/<name>/
  meta.json                      host, topics, start/end, counts, drops, reconnects
  up_color/000000.jpg ...        frames exactly as sent (JPEG q90, or PNG with --lossless)
  up_color/index.csv             seq, ROS stamp, receive time, file, size, encodings, frame_id
  up_depth/000000.png ...        16-bit PNG, millimetres, 0 = no reading
  camera_up__color__camera_info.jsonl, odom.jsonl, ...   one JSON message per line
```

Load a depth frame in metres with
`np.asarray(Image.open(path), dtype=np.uint16) / 1000.0`. Match frames across
streams by the `stamp` column. `seq` gaps with no rate cap mean the relay had a
newer frame before the Wi-Fi link could send the old one; `meta.json` counts them.

Streams available in the `glide` preset: `up_color`, `up_depth`, `down_color`,
`down_depth`, `up_ir`, `down_ir`, `segmentation` (labels 0 other, 1 walkspace,
2 road; only published outdoors while operating) and `seg_viz`.

### On-robot bags

The robot's data-collection mode also records MCAP bags to `/data` at about 2 fps
(long-press the left button on the kickstand, or use the helper):

```bash
./glide_robot.sh bag start | bag stop
./glide_robot.sh bags                  # list with sizes
./glide_robot.sh pull-bags 3           # 3 newest finished bags -> data/bags/
```

## Using it from code

```python
from glide_bridge import GlideLink

link = GlideLink("10.42.0.45")                     # or "localhost" for a sim
cam = link.subscribe("/camera_up/color/image_raw", max_rate=6)
depth = link.subscribe("/camera_up/depth/image_raw")
odom = link.subscribe("/odom")

frame = cam.wait(timeout=5)       # blocks for the next frame; latest() doesn't block
rgb = frame.image                 # uint8 (H, W, 3) RGB
metres = depth.latest().image     # float32 (H, W), 0 = no reading
pose = odom.latest().image["pose"]["pose"]["position"]   # JSON topics decode to dicts
link.close()
```

`subscribe()` options: `max_rate` (Hz cap applied on the robot), `lossless` (PNG
color), `decode=False` (raw bytes via `add_listener`, as the recorder does).

## ROS 2 camera stream (`/camera_raw`)

The perception pipeline consumes `sensor_msgs/Image` on `/camera_raw` (intrinsics on
`/camera_raw/camera_info`). For now an MCAP replay feeds it; later the live Glide camera
gets remapped to the same topic. Needs ROS 2 Humble (the robot's distro) in a conda env:

```bash
conda env create -f env/ros_humble.yml        # once
source env/activate_ros.sh && pip install -e . --no-deps   # once
```

Get bags from the project's Google Drive "Data" folder with `fetch_bags.sh` (uses
[rclone](https://rclone.org), installed as a single binary in `~/.local/bin`; read-only
access with your own Google login):

```bash
./fetch_bags.sh setup        # once: sign in through the browser
./fetch_bags.sh list         # bags in the Drive folder
./fetch_bags.sh pull         # download any you don't have yet into data/bags/
./fetch_bags.sh get <drive URL or file id>   # a specific file from anywhere in Drive
```

Then:

```bash
./demos/stream_demo.sh data/bags/<bag>.mcap   # replay + live matplotlib window
./demos/stream_demo.sh                        # same, on a synthetic 6 fps bag

# or the two halves separately (two terminals, both: source env/activate_ros.sh)
python -m glide_bridge.ros.replay data/bags/<bag>.mcap --list    # topics in the bag
python -m glide_bridge.ros.replay data/bags/<bag>.mcap --loop    # recorded timing
python -m glide_bridge.ros.viewer --plot data/plots/timing.png   # window + timing report
```

The replay keeps the recorded frame timing (`--rate 2` for 2x, `--fps N` to force a
rate) and stamps frames with the publish time, so the viewer can show latency.
`env/activate_ros.sh` uses Cyclone DDS like the robot and keeps traffic on localhost.

## Simulators

The relay runs anywhere ROS 2 does. For the old Gazebo test sim in `../Glide`
(not this robot's sensors; used only to test the pipeline):

```bash
cd ../Glide && ./sim_headless.sh            # or ./sim_run.sh
docker cp ../local_glide_bridge/robot/glide_link_relay.py glide_sim:/tmp/
docker exec -d glide_sim bash -ic 'python3 /tmp/glide_link_relay.py > /tmp/relay.log 2>&1'
cd ../local_glide_bridge
uv run python view_cameras.py --host localhost --preset sim
uv run python record.py --host localhost --preset sim --duration 20
```

The Gazebo sim can also be read through rosbridge (`--backend rosbridge`,
`../Glide/sim_bridge.sh`). The Isaac Sim twin (plan phase 3) will run the same relay.

## Layout

| Path | Contents |
| --- | --- |
| `robot/glide_link_relay.py` | relay: ROS 2 topics -> TCP, encodes images, serves many clients |
| `robot/run_relay.sh` | starts the relay with the robot service's DDS environment |
| `glide_robot.sh` | laptop helper: deploy/start/stop relay, uploader, bags, info |
| `fetch_bags.sh` | download bags from the project Google Drive (rclone) |
| `glide_bridge/ros/` | ROS 2 nodes: MCAP replay to `/camera_raw`, matplotlib stream viewer |
| `demos/` | runnable visual checks (`stream_demo.sh`, synthetic bag generator) |
| `env/` | ROS 2 Humble conda env and activate script |
| `record.py` | session recorder |
| `view_cameras.py` | live matplotlib viewer |
| `glide_bridge/link.py` | `GlideLink` client, `TopicFeed` |
| `glide_bridge/topics.py` | stream presets (`glide`, `sim`) and telemetry topic list |
| `glide_bridge/decode.py`, `client.py` | rosbridge decoding and client (Gazebo sim) |
| `tests/` | framing and decoding tests |

## Troubleshooting

- **`Could not connect`**: the relay isn't running (`./glide_robot.sh status`) or the
  laptop isn't on the "Glide" network (`ping $GLIDE_HOST`).
- **A panel stays on "no publisher yet"**: the topic doesn't exist in the running
  config, e.g. segmentation indoors, or a CARLA overlay is running instead of the
  normal service. `./glide_robot.sh log` shows what the relay subscribed to.
- **Relay sees no topics at all**: it must use the same DDS settings as the stack.
  `run_relay.sh` assumes the normal service (Cyclone, localhost). For CARLA overlays,
  see the comment at the top of `run_relay.sh`.
- **Low frame rate or many drops**: lower `--rate`, record fewer streams, or move
  closer to the access point. The robot's cameras run at 6 fps in both configs.
