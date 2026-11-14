#!/usr/bin/env python3
"""Write a small rosbag2 MCAP that looks like a Glide camera recording.

Useful for exercising the replay -> robot camera topics -> viewer pipeline without real
data. Frames show a moving disc and the frame number, so dropped or out-of-order
frames are easy to spot in the viewer.

Run (glide_ros env)::

    python demos/make_synthetic_bag.py data/bags/synthetic --fps 6 --seconds 20
    python -m glide_bridge.ros.replay data/bags/synthetic --list
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from PIL import Image as PILImage
from PIL import ImageDraw
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo

from glide_bridge.ros.image_msgs import numpy_to_image

# (camera, kind) -> (recorded image topic, info topic), as the robot's data recorder names them
TOPICS: dict[tuple[str, str], tuple[str, str]] = {
    (cam, kind): (f"/camera_{cam}/{kind}/image_raw_rfps", f"/camera_{cam}/{kind}/camera_info")
    for cam in ("up", "down") for kind in ("color", "depth")
}


def render_depth(k: int, w: int, h: int, fps: float) -> np.ndarray:
    """Synthetic depth for frame ``k``: a floor receding upward plus the moving disc.

    Returns
    -------
    np.ndarray
        ``(h, w)`` uint16 depth in millimetres (``16UC1``), 0 = no reading.

    Notes
    -----
    Floor depth falls off linearly from 5 m at the top row to 0.5 m at the bottom,
    :math:`d(y) = 5000 - 4500\,y` mm with :math:`y \in [0, 1]`. The disc follows the
    same path as in :func:`render_frame` at a constant 0.8 m. A 20 px band on the
    left edge has no reading, as real depth cameras have at their borders.
    """
    y: np.ndarray = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    depth: np.ndarray = np.broadcast_to(5000 - 4500 * y, (h, w)).copy()   # (h, w) float32 mm
    t = k / fps
    cx, cy = w / 2 + 0.35 * w * np.cos(2 * np.pi * t / 4), h / 2 + 0.3 * h * np.sin(2 * np.pi * t / 4)
    yy, xx = np.mgrid[0:h, 0:w]
    depth[(xx - cx) ** 2 + (yy - cy) ** 2 <= (0.08 * h) ** 2] = 800
    depth[:, :20] = 0
    return depth.astype(np.uint16)


def render_frame(k: int, w: int, h: int, fps: float, label: str = "up") -> np.ndarray:
    """Draw frame ``k``: gradient background, a disc on a circular path, the frame number.

    Returns
    -------
    np.ndarray
        ``(h, w, 3)`` uint8 RGB.

    Notes
    -----
    The disc centre at time :math:`t = k / \\text{fps}` is
    :math:`(c_x + 0.35 w \\cos \\omega t,\\; c_y + 0.3 h \\sin \\omega t)` with
    :math:`\\omega = 2\\pi / 4` rad/s (one lap every 4 s).
    """
    x: np.ndarray = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y: np.ndarray = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    chans = (80 + 100 * x, 60 + 120 * y, 140 + 0 * x * y)
    if label == "down":                      # warmer palette so the cameras look different
        chans = (150 + 60 * y, 90 + 60 * x, 50 + 0 * x * y)
    bg: np.ndarray = np.stack(np.broadcast_arrays(*chans), axis=-1).astype(np.uint8)
    im = PILImage.fromarray(bg)
    d = ImageDraw.Draw(im)
    t = k / fps
    cx, cy = w / 2 + 0.35 * w * np.cos(2 * np.pi * t / 4), h / 2 + 0.3 * h * np.sin(2 * np.pi * t / 4)
    r = 0.08 * h
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(250, 200, 40))
    d.text((12, 10), f"{label}  frame {k:04d}  t={t:6.2f}s", fill=(255, 255, 255),
           font_size=max(14, h // 14))
    return np.asarray(im)


def camera_info(w: int, h: int, stamp: Time) -> CameraInfo:
    """Pinhole intrinsics with a ~69 deg horizontal FOV, like the Orbbec color camera.

    Notes
    -----
    :math:`f = (w / 2) / \\tan(\\text{hfov} / 2)`, principal point at the image centre.
    """
    f = (w / 2) / np.tan(np.deg2rad(69) / 2)
    msg = CameraInfo()
    msg.header.stamp, msg.header.frame_id = stamp, "camera_up_color_optical_frame"
    msg.width, msg.height = w, h
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [f, 0.0, w / 2, 0.0, 0.0, f, h / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main() -> int:
    """Write the bag; returns a process exit code."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path, help="bag folder to create (replaced if present)")
    p.add_argument("--fps", type=float, default=6.0, help="up (forward) camera rate")
    p.add_argument("--down-fps", type=float, default=2.0,
                   help="down camera rate; different on purpose, to show independent streams")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--size", type=int, nargs=2, default=(640, 360), metavar=("W", "H"))
    args = p.parse_args()
    w, h = args.size

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.out), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for image_topic, info_topic in TOPICS.values():
        for topic, ty in ((image_topic, "sensor_msgs/msg/Image"),
                          (info_topic, "sensor_msgs/msg/CameraInfo")):
            writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=ty,
                                                         serialization_format="cdr"))

    t0_ns = 1_700_000_000 * 10**9
    records: list[tuple[int, str, bytes]] = []        # (log time ns, topic, CDR bytes)
    for (cam, kind), (image_topic, info_topic) in TOPICS.items():
        fps = args.fps if cam == "up" else args.down_fps
        frame_id = f"camera_{cam}_{kind}_optical_frame"
        for k in range(int(fps * args.seconds)):
            # Small timing jitter (+-5 ms), as a real driver has.
            t_ns = t0_ns + int(k / fps * 1e9) + int(np.random.uniform(-5e6, 5e6))
            stamp = Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9)
            if k % max(int(fps), 1) == 0:                      # CameraInfo at ~1 Hz
                info = camera_info(w, h, stamp)
                info.header.frame_id = frame_id
                records.append((t_ns, info_topic, serialize_message(info)))
            img = (numpy_to_image(render_frame(k, w, h, fps, cam), "rgb8", stamp, frame_id)
                   if kind == "color" else
                   numpy_to_image(render_depth(k, w, h, fps), "16UC1", stamp, frame_id))
            records.append((t_ns + 2_000_000, image_topic, serialize_message(img)))
    for t_ns, topic, data in sorted(records, key=lambda r: r[0]):
        writer.write(topic, data, t_ns)
    del writer
    print(f"wrote {args.seconds:g} s to {args.out}: up camera {args.fps:g} fps, "
          f"down camera {args.down_fps:g} fps, color + depth ({w}x{h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
