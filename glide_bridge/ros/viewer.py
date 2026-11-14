#!/usr/bin/env python3
"""Live matplotlib "video" of Glide camera topics, plus a timing report when it closes.

One panel per stream, laid out with a column per camera (forward/up, down) and a
row per image kind (color on top, depth below). Each panel updates independently
whenever its own topic delivers a frame. This
is the consumer end of the pipeline and the visual check for it: if each panel shows
frames in order at its camera's rate with low latency, the pub/sub path works.

Examples
--------
::

    source env/activate_ros.sh
    python -m glide_bridge.ros.viewer                     # up/down x color/depth, 2x2
    python -m glide_bridge.ros.viewer --streams up_color down_color   # color only
    python -m glide_bridge.ros.viewer --duration 30 --plot data/plots/timing.png
    python -m glide_bridge.ros.viewer --headless --duration 20 --plot timing.png

Close the window (or Ctrl-C) to stop. With ``--plot`` a figure is saved showing, per
stream, the inter-arrival interval, the latency, an interval histogram, and a few
frames sampled across the run.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

from ..topics import CameraStream
from . import DEFAULT_STREAMS, GLIDE_STREAMS

CAMERA_LABELS: dict[str, str] = {"up": "forward (up)", "down": "downward"}
KIND_ROWS: tuple[str, ...] = ("color", "depth", "mono", "labels")   # top to bottom


def panel_label(stream: CameraStream) -> str:
    """Short panel title, e.g. ``forward (up) depth``."""
    cam, _, rest = stream.name.partition("_")
    return f"{CAMERA_LABELS[cam]} {rest}" if cam in CAMERA_LABELS and rest else stream.name


def grid_layout(streams: Sequence[CameraStream]) -> tuple[dict[str, tuple[int, int]], int, int]:
    """Place streams on a grid: one column per camera, one row per image kind.

    Parameters
    ----------
    streams : sequence of CameraStream
        Streams to show, e.g. ``up_color, down_color, up_depth, down_depth``.

    Returns
    -------
    cells : dict[str, tuple[int, int]]
        Stream name -> ``(row, col)``.
    n_rows, n_cols : int
        Grid size. Rows follow :data:`KIND_ROWS` (color above depth above IR);
        columns follow the order cameras first appear in ``streams``. A stream
        without a camera prefix (e.g. ``segmentation``) gets its own column.

    Examples
    --------
    ``up_color, down_color, up_depth, down_depth`` gives a 2x2 grid with up on the
    left, down on the right, color on top and depth below.
    """
    cams: list[str] = []
    for s in streams:
        cam = s.name.partition("_")[0] if s.name.partition("_")[2] else s.name
        if cam not in cams:
            cams.append(cam)
    kinds = [k for k in KIND_ROWS if any(s.kind == k for s in streams)]
    kinds += sorted({s.kind for s in streams} - set(kinds))
    cells: dict[str, tuple[int, int]] = {}
    for s in streams:
        cam = s.name.partition("_")[0] if s.name.partition("_")[2] else s.name
        cells[s.name] = (kinds.index(s.kind), cams.index(cam))
    return cells, len(kinds), len(cams)


class StreamMonitor:
    """Arrival record and newest frame for one image topic. Thread-safe.

    Parameters
    ----------
    stream : CameraStream
        Which stream this monitors; ``stream.base_topic`` is the subscribed topic.
    max_thumbs : int
        Upper bound on frames kept for the report (evenly spaced over the run).

    Attributes
    ----------
    recv : list of float
        Receive time of each frame, wall-clock seconds (``time.time()``).
    latency : list of float
        ``recv - header.stamp`` in seconds. Meaningful only when the publisher stamps
        with its publish time (the replay default) on the same clock.
    shape : tuple of int
        Shape of the last frame, ``(H, W)`` or ``(H, W, C)``.
    encoding : str
        ROS encoding of the last frame.
    """

    def __init__(self, stream: CameraStream, max_thumbs: int = 6) -> None:
        self.stream = stream
        self.max_thumbs = max_thumbs
        self.recv: list[float] = []
        self.latency: list[float] = []
        self.shape: tuple[int, ...] = ()
        self.encoding = ""
        self._latest: tuple[int, np.ndarray] | None = None   # (frame index, image)
        self._pool: list[tuple[int, np.ndarray]] = []         # thumbnail candidates
        self._keep_every = 1
        self._lock = threading.Lock()

    def add(self, img: np.ndarray, encoding: str, stamp_s: float) -> None:
        """Record one received frame.

        Parameters
        ----------
        img : np.ndarray
            Decoded image, ``(H, W)`` or ``(H, W, C)``.
        encoding : str
            ROS encoding.
        stamp_s : float
            ``header.stamp`` in seconds.

        Notes
        -----
        Thumbnails are kept by doubling decimation: every ``keep_every``-th frame
        is pooled, and when the pool exceeds ``4 * max_thumbs`` every other entry is
        dropped and ``keep_every`` doubles. Memory stays bounded and the pool stays
        roughly uniform over the run.
        """
        now = time.time()
        with self._lock:
            k = len(self.recv)
            self.recv.append(now)
            self.latency.append(now - stamp_s)
            self.shape, self.encoding = img.shape, encoding
            self._latest = (k, img)
            if self.max_thumbs > 0 and k % self._keep_every == 0:
                self._pool.append((k, img[::2, ::2].copy()))
                if len(self._pool) > 4 * self.max_thumbs:
                    del self._pool[1::2]
                    self._keep_every *= 2

    def latest(self) -> tuple[int, np.ndarray] | None:
        """Newest ``(frame index, image)``, or None before the first frame."""
        with self._lock:
            return self._latest

    def last_latency(self) -> float:
        """Latency of the newest frame in seconds (0 before the first frame)."""
        with self._lock:
            return self.latency[-1] if self.latency else 0.0

    def fps(self, window: int = 30) -> float:
        """Arrival rate over the last ``window`` frames: :math:`(n - 1) / (r_{-1} - r_{-n})`."""
        with self._lock:
            r = self.recv[-window:]
        return (len(r) - 1) / (r[-1] - r[0]) if len(r) > 1 and r[-1] > r[0] else 0.0

    def thumbnails(self) -> list[tuple[int, np.ndarray]]:
        """Up to ``max_thumbs`` pooled frames, evenly spaced over the run."""
        with self._lock:
            pool = list(self._pool)
        if not pool:
            return []
        idx: np.ndarray = np.linspace(0, len(pool) - 1, min(self.max_thumbs, len(pool)))
        return [pool[i] for i in np.unique(idx.round().astype(int))]


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


def save_report(monitors: list[StreamMonitor], path: Path, depth_max_m: float = 5.0) -> None:
    """Save the timing figure for a finished run.

    Parameters
    ----------
    monitors : list of StreamMonitor
        One per subscribed stream; streams with fewer than 2 frames get no curves.
    path : Path
        Output PNG.
    depth_max_m : float, optional
        Colour range for depth thumbnails, metres.

    Notes
    -----
    Per stream, interval :math:`\\Delta_k = r_k - r_{k-1}` (receive times), rate
    :math:`(N - 1) / (r_{N-1} - r_0)`, jitter :math:`\\operatorname{std}(\\Delta)`.
    Times are relative to the first frame of any stream, so streams line up.
    """
    import matplotlib.pyplot as plt

    active = [m for m in monitors if len(m.recv) >= 2]
    if not active:
        print("fewer than 2 frames on every stream; no report", file=sys.stderr)
        return
    t0 = min(m.recv[0] for m in active)
    # Streams can differ in rate by 10x or more; then intervals go on a log axis with
    # log-spaced histogram bins shared by all streams. Narrow ranges stay linear.
    all_dt: np.ndarray = np.concatenate([np.diff(m.recv) for m in active]) * 1e3
    lo, hi = max(float(all_dt.min()), 1.0), float(all_dt.max())
    log_dt = hi / lo > 4.0
    bins: np.ndarray = (np.geomspace(lo * 0.9, hi * 1.1, 40) if log_dt
                        else np.linspace(lo * 0.95, hi * 1.05, 40))
    dt_scale, dt_unit = ("log", "ms, log") if log_dt else ("linear", "ms")
    thumbs = {m.stream.name: m.thumbnails() for m in monitors}
    rows = [m for m in monitors if thumbs[m.stream.name]]

    fig = plt.figure(figsize=(13, 4.2 + 2.3 * len(rows)), layout="constrained")
    grid = fig.add_gridspec(1 + len(rows), 1, height_ratios=[1.8] + [1] * len(rows))
    ax_dt, ax_lat, ax_hist = grid[0].subgridspec(1, 3).subplots()
    summary = []
    for i, m in enumerate(active):
        color = f"C{i}"
        r: np.ndarray = np.asarray(m.recv, dtype=np.float64)
        dt_ms: np.ndarray = np.diff(r) * 1e3
        lat_ms: np.ndarray = np.asarray(m.latency) * 1e3
        fps = (len(r) - 1) / (r[-1] - r[0])
        label = f"{m.stream.name}: {fps:.2f} fps"
        ax_dt.plot(r[1:] - t0, dt_ms, ".", ms=3, color=color, label=label)
        ax_lat.plot(r - t0, lat_ms, ".", ms=3, color=color,
                    label=f"{m.stream.name}: median {np.median(lat_ms):.1f} ms")
        ax_hist.hist(dt_ms, bins=bins, color=color, alpha=0.6,
                     label=f"{m.stream.name}: jitter {np.std(dt_ms):.0f} ms")
        summary.append(f"{m.stream.name} {len(r)} frames {fps:.2f} fps, median interval "
                       f"{np.median(dt_ms):.0f} ms, median latency {np.median(lat_ms):.1f} ms")
    ax_dt.set(xlabel="time since first frame (s)", ylabel=f"interval ({dt_unit})",
              title="Inter-arrival interval", yscale=dt_scale)
    ax_lat.set(xlabel="time since first frame (s)", ylabel="latency (ms)",
               title="Publish -> receive latency")
    ax_hist.set(xlabel=f"interval ({dt_unit})", ylabel="frames", title="Interval histogram",
                xscale=dt_scale)
    for ax in (ax_dt, ax_lat, ax_hist):
        ax.legend(fontsize=8, loc="best")

    n_cols = max(len(t) for t in thumbs.values()) if rows else 0
    for row, m in enumerate(rows, start=1):
        axs = np.atleast_1d(grid[row].subgridspec(1, n_cols).subplots())
        for ax in axs:
            ax.axis("off")
        for ax, (k, img) in zip(axs, thumbs[m.stream.name]):
            ax.imshow(to_display(img, m.encoding, depth_max_m))
            ax.set_title(f"{m.stream.name} #{k}", fontsize=9)

    fig.suptitle(", ".join(f"{m.stream.base_topic}: {len(m.recv)} frames" for m in monitors))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved {path}")
    for line in summary:
        print(f"  {line}")


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Parse our options; everything else (``--ros-args ...``) is passed to rclpy."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS,
                   choices=sorted(GLIDE_STREAMS), metavar="STREAM",
                   help=f"streams to show, one panel each (default: "
                        f"{' '.join(DEFAULT_STREAMS)}); choices: "
                        f"{', '.join(sorted(GLIDE_STREAMS))}")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--plot", type=Path, help="save a timing report PNG here on exit")
    p.add_argument("--snapshot", type=Path,
                   help="save the window as it last looked to this PNG on exit")
    p.add_argument("--headless", action="store_true",
                   help="no window; just collect stats (use with --plot)")
    p.add_argument("--thumbs", type=int, default=6, help="frames per stream in the report")
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

    monitors = [StreamMonitor(GLIDE_STREAMS[name], args.thumbs)
                for name in dict.fromkeys(args.streams)]

    rclpy.init(args=["viewer", *ros_args])
    node = Node("camera_stream_viewer")
    for m in monitors:
        def on_image(msg: Image, m: StreamMonitor = m) -> None:
            m.add(image_to_numpy(msg), msg.encoding, stamp_to_sec(msg.header.stamp))
        node.create_subscription(Image, m.stream.base_topic, on_image,
                                 qos_profile_sensor_data)
        node.get_logger().info(f"{m.stream.name}: subscribed to {m.stream.base_topic}")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    t_start = time.time()

    try:
        if args.headless:
            while args.duration is None or time.time() - t_start < args.duration:
                time.sleep(0.1)
        else:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            cells, n_rows, n_cols = grid_layout([m.stream for m in monitors])
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 4.0 * n_rows),
                                     squeeze=False, layout="constrained")
            fig.canvas.manager.set_window_title("Glide camera streams")
            used = set(cells.values())
            for (r, c), ax in np.ndenumerate(axes):
                if (r, c) not in used:
                    ax.set_visible(False)
            panels = []
            for m in monitors:
                ax = axes[cells[m.stream.name]]
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(m.stream.base_topic, fontsize=9, family="monospace")
                ax.set_title(f"{panel_label(m.stream)}: waiting ...", fontsize=10)
                if m.stream.kind == "depth":
                    fig.colorbar(ScalarMappable(Normalize(0, args.depth_max), "turbo"), ax=ax,
                                 shrink=0.85, label="depth (m), black = no reading")
                panels.append((ax, ax.imshow(np.zeros((2, 2, 3), np.uint8)), [-1]))

            def update(_: int) -> list:
                for m, (ax, im, shown) in zip(monitors, panels):
                    item = m.latest()
                    if item is None or item[0] == shown[0]:
                        continue          # this stream has nothing new; others may
                    k, img = item
                    shown[0] = k
                    rgb = to_display(img, m.encoding, args.depth_max)
                    if im.get_array().shape[:2] != rgb.shape[:2]:
                        im.set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
                    im.set_data(rgb)
                    lat = m.last_latency()
                    ax.set_title(f"{panel_label(m.stream)}   frame {k}   "
                                 f"{m.fps():.1f} fps   latency {lat * 1e3:.1f} ms",
                                 fontsize=10)
                return []

            # Poll at ~30 Hz and redraw a panel only when its stream has a new frame,
            # so each panel runs at its own camera's rate.
            anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
            if args.duration is not None:
                closer = fig.canvas.new_timer(interval=int(args.duration * 1000))
                closer.single_shot = True
                closer.add_callback(lambda: (anim.event_source.stop(), plt.close(fig)))
                closer.start()
            plt.show()
            del anim
            if args.snapshot:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.snapshot, dpi=90)
                print(f"saved window snapshot {args.snapshot}")
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    for m in monitors:
        print(f"{m.stream.name}: received {len(m.recv)} frames on {m.stream.base_topic}")
    if args.plot:
        matplotlib.use("Agg", force=True)
        save_report(monitors, args.plot, args.depth_max)
    return 0 if any(m.recv for m in monitors) else 1


if __name__ == "__main__":
    sys.exit(main())
