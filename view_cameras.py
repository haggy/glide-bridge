#!/usr/bin/env python3
"""Live view of Glide camera streams in matplotlib.

Reads from the glide_link relay on the robot (default) or, with --backend
rosbridge, from the Gazebo test sim's rosbridge server.

Examples:
    python view_cameras.py --host 10.42.0.45                     # robot, up/down RGB-D
    python view_cameras.py --host 10.42.0.45 --streams up_color segmentation
    python view_cameras.py --host localhost --preset sim         # sim via glide_link
    python view_cameras.py --backend rosbridge --preset sim      # sim via rosbridge
    python view_cameras.py --host 10.42.0.45 --snapshot frame.png
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from glide_bridge import DEFAULT_STREAMS, PRESETS, CameraFeed, GlideBridge, GlideLink, TopicFeed
from glide_bridge.link import DEFAULT_PORT


def parse_args() -> argparse.Namespace:
    """Command-line options (see the module docstring for examples)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="10.42.0.45",
                   help="relay host: robot IP, or localhost for the sim (default: 10.42.0.45)")
    p.add_argument("--backend", choices=["link", "rosbridge"], default="link",
                   help="link = glide_link relay (default); rosbridge = Gazebo test sim only")
    p.add_argument("--port", type=int, help=f"default: {DEFAULT_PORT} (link) or 9090 (rosbridge)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="glide",
                   help="topic names: glide (robot) or sim (default: glide)")
    p.add_argument("--streams", nargs="+",
                   help="streams to show (default: up/down color + depth for glide)")
    p.add_argument("--transport", choices=["compressed", "raw"], default="compressed",
                   help="rosbridge only: compressed (default) or raw sensor_msgs/Image")
    p.add_argument("--max-rate", type=float, default=10.0,
                   help="per-stream rate limit in Hz, applied by rosbridge (default: 10)")
    p.add_argument("--depth-max", type=float, default=5.0,
                   help="depth colour scale upper bound in metres (default: 5)")
    p.add_argument("--snapshot", metavar="PNG",
                   help="wait for a frame from every stream, save the figure, and exit")
    p.add_argument("--timeout", type=float, default=20.0,
                   help="seconds to wait for frames with --snapshot (default: 20)")
    return p.parse_args()


def main() -> int:
    """Connect, open one panel per stream and animate until the window closes.

    Returns
    -------
    int
        Exit code: 0 ok, 1 connection failed or a snapshot stream had no frames,
        2 unknown stream name.
    """
    args = parse_args()
    preset = PRESETS[args.preset]
    args.streams = args.streams or DEFAULT_STREAMS.get(args.preset, list(preset)[:2])
    args.port = args.port or (DEFAULT_PORT if args.backend == "link" else 9090)
    unknown = [s for s in args.streams if s not in preset]
    if unknown:
        print(f"Unknown stream(s) {unknown} for preset {args.preset!r}; "
              f"choose from {sorted(preset)}", file=sys.stderr)
        return 2

    print(f"Connecting to {args.backend} at {args.host}:{args.port} ...")
    try:
        if args.backend == "link":
            bridge: GlideLink | GlideBridge = GlideLink(args.host, args.port)
        else:
            bridge = GlideBridge(args.host, args.port)
    except Exception as exc:
        hint = ("Is the relay running? On the robot: ./glide_robot.sh start"
                if args.backend == "link" else "Is rosbridge running? ../Glide/sim_bridge.sh")
        print(f"Could not connect: {exc}\n{hint}", file=sys.stderr)
        return 1

    if args.backend == "link":
        feeds: list[TopicFeed | CameraFeed] = []
        for name in args.streams:
            feed: Any = bridge.subscribe(preset[name].base_topic, max_rate=args.max_rate)
            feed.stream = preset[name]   # TopicFeed has no .stream; the viewer adds it
            feeds.append(feed)
    else:
        feeds = [bridge.add_camera(preset[name], args.transport, args.max_rate)
                 for name in args.streams]
    for f in feeds:
        print(f"  {f.stream.name:8s} <- {f.topic_name} ({f.msg_type})")

    n = len(feeds)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.2 * rows), squeeze=False,
                             layout="constrained")
    fig.canvas.manager.set_window_title(f"Glide cameras ({args.preset} @ {args.host})")
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    artists: dict[str, tuple[Axes, AxesImage, list[Any]]] = {}   # name -> (ax, image, [last frame])
    for ax, feed in zip(axes.flat, feeds):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{feed.stream.name}: waiting for {feed.topic_name}", fontsize=9)
        if feed.stream.kind == "depth":
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("0.85")     # 0 = no depth reading; draw it light grey
            im = ax.imshow(np.ma.masked_equal(np.zeros((2, 2), np.float32), 0), cmap=cmap,
                           vmin=0.0, vmax=args.depth_max, interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.8, label="depth (m), grey = no reading")
        elif feed.stream.kind == "labels":   # 0 other, 1 walkspace, 2 road
            im = ax.imshow(np.zeros((2, 2), np.uint8), cmap="tab10", vmin=0, vmax=9,
                           interpolation="nearest")
        elif feed.stream.kind == "mono":
            im = ax.imshow(np.zeros((2, 2), np.uint8), cmap="gray", vmin=0, vmax=255)
        else:
            im = ax.imshow(np.zeros((2, 2, 3), np.uint8))
        artists[feed.stream.name] = (ax, im, [None])   # [last frame shown]

    def update(_: int | None) -> list:
        """Animation callback: draw each feed's newest frame if it changed."""
        for feed in feeds:
            ax, im, shown = artists[feed.stream.name]
            frame = feed.latest()
            errors, last_error = feed.error()
            if frame is None:
                if errors or getattr(feed, "state", "") == "waiting":
                    msg = last_error or getattr(feed, "status_error", "")
                    ax.set_title(f"{feed.stream.name}: {msg}"[:90], fontsize=9, color="tab:red")
                continue
            if frame is shown[0]:
                continue
            shown[0] = frame
            img: np.ndarray = frame.image   # (H, W, 3) uint8 RGB, (H, W) float32 m, or (H, W) uint8
            if im.get_array().shape[:2] != img.shape[:2]:
                im.set_extent((-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
            im.set_data(np.ma.masked_equal(img, 0) if feed.stream.kind == "depth" else img)
            age_ms = (time.time() - frame.received) * 1000
            ax.set_title(f"{feed.stream.name}  {img.shape[1]}x{img.shape[0]}  "
                         f"{feed.fps():.1f} fps  ({frame.encoding})", fontsize=9,
                         color="black")
        return []

    if args.snapshot:
        deadline = time.time() + args.timeout
        while time.time() < deadline and any(f.latest() is None for f in feeds):
            time.sleep(0.2)
        time.sleep(1.0)   # let the fps estimate settle
        update(None)
        missing = [f.stream.name for f in feeds if f.latest() is None]
        fig.savefig(args.snapshot, dpi=110)
        bridge.close()
        print(f"Saved {args.snapshot}" + (f" (no frames from {missing})" if missing else ""))
        return 1 if missing else 0

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    try:
        plt.show()
    finally:
        del anim
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
