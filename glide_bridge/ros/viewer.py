#!/usr/bin/env python3
"""Live matplotlib "video" of ``/camera_raw``, plus a timing report when it closes.

This is the consumer end of the pipeline and the visual check for it: if frames
appear in order at the camera's rate with low latency, the pub/sub path works.

Examples
--------
::

    source env/activate_ros.sh
    python -m glide_bridge.ros.viewer                        # window until closed
    python -m glide_bridge.ros.viewer --duration 30 --plot data/plots/timing.png
    python -m glide_bridge.ros.viewer --headless --duration 20 --plot timing.png

Close the window (or Ctrl-C) to stop. With ``--plot`` a figure is saved showing,
per received frame, the inter-arrival interval, the latency, an interval histogram,
and a few frames sampled across the run.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np


@dataclass
class StreamStats:
    """Per-frame arrival record.

    Attributes
    ----------
    recv : list of float
        Receive time of each frame, wall-clock seconds (``time.time()``).
    latency : list of float
        ``recv - header.stamp`` in seconds. Meaningful only when the publisher stamps
        with its publish time (the replay default) on the same clock.
    shape : tuple
        ``(H, W)`` or ``(H, W, C)`` of the last frame.
    encoding : str
        ROS encoding of the last frame.
    samples : list of tuple[int, np.ndarray]
        ``(frame index, image)`` thumbnails kept for the report.
    """

    recv: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    shape: tuple[int, ...] = ()
    encoding: str = ""
    samples: list[tuple[int, np.ndarray]] = field(default_factory=list)


def to_display(img: np.ndarray, encoding: str, depth_max_m: float = 5.0) -> np.ndarray:
    """Make any supported image displayable by ``imshow``.

    Parameters
    ----------
    img : np.ndarray
        Output of :func:`glide_bridge.ros.image_msgs.image_to_numpy`.
    encoding : str
        ROS encoding of ``img``.
    depth_max_m : float, optional
        Depth that maps to the top of the colour scale, metres.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 RGB. Depth is scaled as
        ``rgb = turbo(clip(d / depth_max_m, 0, 1))`` with ``d`` in metres
        (16-bit encodings are mm, so ``d = raw / 1000``); 0 (no reading) is black.
    """
    if img.ndim == 3:
        return img[..., :3]
    if encoding in ("16UC1", "mono16", "32FC1"):
        d: np.ndarray = img.astype(np.float32) / (1.0 if encoding == "32FC1" else 1000.0)
        rgb: np.ndarray = (matplotlib.colormaps["turbo"](np.clip(d / depth_max_m, 0, 1))[..., :3]
                           * 255).astype(np.uint8)
        rgb[d <= 0] = 0
        return rgb
    return np.repeat(img[..., None], 3, axis=2)


def save_report(stats: StreamStats, path: Path, topic: str) -> None:
    """Save the timing figure for a finished run.

    Parameters
    ----------
    stats : StreamStats
        Arrivals collected by the viewer.
    path : Path
        Output PNG.
    topic : str
        Topic name for the title.

    Notes
    -----
    Interval :math:`\\Delta_k = r_k - r_{k-1}` (receive times). Reported rate is
    :math:`(N - 1) / (r_{N-1} - r_0)`; jitter is the standard deviation of
    :math:`\\Delta`.
    """
    import matplotlib.pyplot as plt

    r: np.ndarray = np.asarray(stats.recv, dtype=np.float64)
    if len(r) < 2:
        print("fewer than 2 frames received; no report", file=sys.stderr)
        return
    t: np.ndarray = r - r[0]
    dt_ms: np.ndarray = np.diff(r) * 1e3
    lat_ms: np.ndarray = np.asarray(stats.latency) * 1e3
    fps = (len(r) - 1) / (r[-1] - r[0])

    n_thumbs = len(stats.samples)
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    grid = fig.add_gridspec(2 if n_thumbs else 1, 1, height_ratios=[1.6, 1] if n_thumbs else None)
    ax_dt, ax_lat, ax_hist = grid[0].subgridspec(1, 3).subplots()

    ax_dt.plot(t[1:], dt_ms, ".", ms=3)
    ax_dt.axhline(np.median(dt_ms), color="tab:red", lw=1,
                  label=f"median {np.median(dt_ms):.0f} ms")
    ax_dt.set(xlabel="time since first frame (s)", ylabel="interval (ms)",
              title="Inter-arrival interval")
    ax_dt.legend(loc="upper right", fontsize=8)

    ax_lat.plot(t, lat_ms, ".", ms=3, color="tab:green")
    ax_lat.set(xlabel="time since first frame (s)", ylabel="latency (ms)",
               title=f"Publish -> receive latency (median {np.median(lat_ms):.1f} ms)")

    ax_hist.hist(dt_ms, bins=40, color="tab:blue")
    ax_hist.set(xlabel="interval (ms)", ylabel="frames",
                title=f"{fps:.2f} fps, jitter {np.std(dt_ms):.1f} ms")

    thumb_grid = grid[1].subgridspec(1, n_thumbs) if n_thumbs else None
    for i, (k, img) in enumerate(stats.samples):
        ax = fig.add_subplot(thumb_grid[0, i])
        ax.imshow(to_display(img, stats.encoding))
        ax.set_title(f"frame {k}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{topic}: {len(r)} frames in {t[-1]:.1f} s, "
                 f"{stats.shape[1] if len(stats.shape) > 1 else '?'}x{stats.shape[0] if stats.shape else '?'} "
                 f"{stats.encoding}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}: {len(r)} frames, {fps:.2f} fps, median interval "
          f"{np.median(dt_ms):.0f} ms, jitter {np.std(dt_ms):.1f} ms, "
          f"median latency {np.median(lat_ms):.1f} ms")


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Parse our options; everything else (``--ros-args ...``) is passed to rclpy."""
    from . import CAMERA_TOPIC
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--topic", default=CAMERA_TOPIC, help=f"image topic (default: {CAMERA_TOPIC})")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--plot", type=Path, help="save a timing report PNG here on exit")
    p.add_argument("--headless", action="store_true",
                   help="no window; just collect stats (use with --plot)")
    p.add_argument("--thumbs", type=int, default=6, help="frames shown in the report")
    p.add_argument("--depth-max", type=float, default=5.0, help="depth colour range, m")
    return p.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m glide_bridge.ros.viewer [options]``."""
    args, ros_args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    from .image_msgs import image_to_numpy, stamp_to_sec

    stats = StreamStats()
    lock = threading.Lock()
    latest: list[tuple[int, np.ndarray] | None] = [None]   # (frame index, image)
    all_frames_for_thumbs: list[tuple[int, np.ndarray]] = []
    keep_every = 1

    def on_image(msg: Image) -> None:
        nonlocal keep_every
        now = time.time()
        img: np.ndarray = image_to_numpy(msg)
        with lock:
            k = len(stats.recv)
            stats.recv.append(now)
            stats.latency.append(now - stamp_to_sec(msg.header.stamp))
            stats.shape, stats.encoding = img.shape, msg.encoding
            latest[0] = (k, img)
            # Keep a bounded, evenly spaced set of frames for the report thumbnails.
            if k % keep_every == 0:
                all_frames_for_thumbs.append((k, img[::2, ::2].copy()))
                if len(all_frames_for_thumbs) > 4 * max(args.thumbs, 1):
                    del all_frames_for_thumbs[1::2]
                    keep_every *= 2

    rclpy.init(args=["viewer", *ros_args])
    node = Node("camera_stream_viewer")
    node.create_subscription(Image, args.topic, on_image, qos_profile_sensor_data)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    node.get_logger().info(f"subscribed to {args.topic}; waiting for frames")
    t_start = time.time()

    try:
        if args.headless:
            while args.duration is None or time.time() - t_start < args.duration:
                time.sleep(0.1)
        else:
            fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
            fig.canvas.manager.set_window_title(f"Glide camera stream: {args.topic}")
            ax.set_xticks([]); ax.set_yticks([])
            im = ax.imshow(np.zeros((2, 2, 3), np.uint8))
            ax.set_title(f"waiting for {args.topic} ...")
            shown: list[int] = [-1]

            def update(_: int) -> list:
                with lock:
                    item = latest[0]
                    r = stats.recv[-30:]
                    lat = stats.latency[-1] if stats.latency else 0.0
                if item is None or item[0] == shown[0]:
                    return []
                k, img = item
                shown[0] = k
                rgb = to_display(img, stats.encoding, args.depth_max)
                if im.get_array().shape[:2] != rgb.shape[:2]:
                    im.set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
                im.set_data(rgb)
                fps = (len(r) - 1) / (r[-1] - r[0]) if len(r) > 1 and r[-1] > r[0] else 0.0
                ax.set_title(f"frame {k}   {rgb.shape[1]}x{rgb.shape[0]} {stats.encoding}   "
                             f"{fps:.1f} fps   latency {lat * 1e3:.1f} ms", fontsize=10)
                return []

            # Poll at ~30 Hz and redraw only on a new frame, so display rate follows
            # the stream's rate instead of the timer's.
            anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
            if args.duration is not None:
                closer = fig.canvas.new_timer(interval=int(args.duration * 1000))
                closer.single_shot = True
                closer.add_callback(lambda: (anim.event_source.stop(), plt.close(fig)))
                closer.start()
            plt.show()
            del anim
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    with lock:
        n = len(stats.recv)
        if all_frames_for_thumbs and args.thumbs > 0:
            idx = np.linspace(0, len(all_frames_for_thumbs) - 1,
                              min(args.thumbs, len(all_frames_for_thumbs))).round().astype(int)
            stats.samples = [all_frames_for_thumbs[i] for i in idx]
    print(f"received {n} frames on {args.topic}")
    if args.plot:
        matplotlib.use("Agg", force=True)
        save_report(stats, args.plot, args.topic)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
