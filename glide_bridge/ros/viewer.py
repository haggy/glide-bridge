#!/usr/bin/env python3
"""Live matplotlib "video" of Glide camera topics, plus a timing report when it closes.

One panel per stream, laid out with a column per camera (forward/up, down) and a
row per image kind (color on top, depth below). A third row shows each color frame
after :meth:`src.img_processing.obj_detection.ObjectDetection.apply_bounding_boxes`,
which runs on every received color frame of both cameras. With ``--affordance`` that
row is replaced by the affordance model (:mod:`src.model.infer`): on the left, the up
camera's color with the predicted sittability heatmap overlaid; on the right, the
bird's-eye view of that heatmap, projected with the up camera's depth (forward up,
1 m grid, camera = cyan triangle at the bottom). Each panel updates
independently whenever its own topic delivers a frame. This
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
    python -m glide_bridge.ros.viewer --no-detections     # skip the detection row
    python -m glide_bridge.ros.viewer --affordance        # heatmap + BEV row, newest best.pt
    python -m glide_bridge.ros.viewer --affordance data/runs/RUN/best.pt
    python -m glide_bridge.ros.viewer --record data/videos/window.mp4   # also save video
    python -m glide_bridge.ros.viewer --headless --record out.mp4       # offscreen video

With ``--record`` the window is written to a video at ``--record-fps`` (wall-clock
time, so playback speed matches what was on screen). With ``--headless`` the same
window is drawn offscreen instead of shown. ``demos/record_bag_video.sh`` records one
pass of a bag this way.

Close the window (or Ctrl-C) to stop. With ``--plot`` a figure is saved showing, per
stream, the inter-arrival interval, the latency, an interval histogram, and a few
frames sampled across the run.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import numpy as np

from ..topics import CameraStream
from . import DEFAULT_STREAMS, GLIDE_STREAMS

CAMERA_LABELS: dict[str, str] = {"up": "forward (up)", "down": "downward"}
# Top to bottom. "detections" and "affordance" are derived in the viewer, not robot topics.
KIND_ROWS: tuple[str, ...] = ("color", "depth", "detections", "affordance", "mono", "labels")
PSEUDO_LABELS: dict[str, str] = {"up_heatmap": "forward (up) sittability heatmap",
                                 "up_bev": "bird's-eye view of the heatmap (from up depth)"}


def detection_stream(source: CameraStream) -> CameraStream:
    """Pseudo-stream for the detector output on a color stream.

    Parameters
    ----------
    source : CameraStream
        A color stream, e.g. ``up_color``.

    Returns
    -------
    CameraStream
        Named ``<camera>_detections`` with kind ``"detections"``, so
        :func:`grid_layout` puts it in the source camera's column, below depth.
        ``base_topic`` is the source topic (the frames it was computed from); nothing
        is published there.
    """
    cam = source.name.partition("_")[0]
    return CameraStream(f"{cam}_detections", source.base_topic, "detections")


class VideoRecorder:
    """Write a matplotlib figure to a video file at a fixed wall-clock frame rate.

    Parameters
    ----------
    path : Path
        Output file. ``.mp4`` uses H.264 if OpenCV has it, else MPEG-4 Part 2
        (``mp4v``); ``.avi`` uses Motion JPEG.
    fps : float
        Video frame rate, frames/s.

    Notes
    -----
    Recording starts at the first :meth:`grab`. Each grab writes as many copies of the
    current canvas as are due, ``n_due = floor((t - t0) * fps) - n_written``, so video
    time tracks wall time even when the GUI timer runs late. Frames keep the size of the
    first grab (later ones are resized; H.264 needs even sizes, so it is rounded down).
    """

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer = None                       # cv2.VideoWriter, opened on first grab
        self._size: tuple[int, int] = (0, 0)      # (W, H) px
        self._t0 = 0.0
        self.n_written = 0

    def _open(self, size: tuple[int, int]) -> None:
        import cv2
        self.path.parent.mkdir(parents=True, exist_ok=True)
        codecs = ("avc1", "mp4v") if self.path.suffix.lower() == ".mp4" else ("MJPG",)
        level = cv2.getLogLevel()
        cv2.setLogLevel(0)  # silent: a failed try logs an error
        try:
            writers = [cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*c), self.fps, size)
                       for c in codecs[:1]]
        finally:
            cv2.setLogLevel(level)
        for codec in codecs:
            w = (writers.pop() if writers else
                 cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps, size))
            if w.isOpened():
                self._writer, self._size, self.codec = w, size, codec
                return
        raise RuntimeError(f"OpenCV can't write {self.path} (tried {', '.join(codecs)})")

    def grab(self, fig: "matplotlib.figure.Figure") -> None:
        """Append the figure's last drawn canvas, repeated as wall time requires."""
        import cv2
        rgba: np.ndarray = np.asarray(fig.canvas.buffer_rgba())   # (H, W, 4) uint8
        if self._writer is None:
            h, w = rgba.shape[:2]
            self._open((w - w % 2, h - h % 2))
            self._t0 = time.time()
        bgr: np.ndarray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if bgr.shape[1::-1] != self._size:
            bgr = cv2.resize(bgr, self._size, interpolation=cv2.INTER_AREA)
        n_due = int((time.time() - self._t0) * self.fps) + 1 - self.n_written
        for _ in range(max(n_due, 0)):
            self._writer.write(bgr)
        self.n_written += max(n_due, 0)

    def close(self) -> None:
        """Finish the file and print what was written."""
        if self._writer is None:
            print(f"no video written to {self.path} (nothing was drawn)")
            return
        self._writer.release()
        self._writer = None
        print(f"saved video {self.path}: {self.n_written} frames, "
              f"{self.n_written / self.fps:.1f} s at {self.fps:g} fps ({self.codec}, "
              f"{self._size[0]}x{self._size[1]})")


def panel_label(stream: CameraStream) -> str:
    """Short panel title, e.g. ``forward (up) depth``."""
    if stream.name in PSEUDO_LABELS:
        return PSEUDO_LABELS[stream.name]
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

    Notes
    -----
    A ``<camera>_bev`` stream goes one column right of its camera (the BEV sits next to
    the heatmap it projects), adding a column if needed.
    """
    cams: list[str] = []
    for s in streams:
        cam = s.name.partition("_")[0] if s.name.partition("_")[2] else s.name
        if cam not in cams:
            cams.append(cam)
    kinds = [k for k in KIND_ROWS if any(s.kind == k for s in streams)]
    kinds += sorted({s.kind for s in streams} - set(kinds))
    cells: dict[str, tuple[int, int]] = {}
    n_cols = len(cams)
    for s in streams:
        cam = s.name.partition("_")[0] if s.name.partition("_")[2] else s.name
        col = cams.index(cam) + (1 if s.name.endswith("_bev") else 0)
        n_cols = max(n_cols, col + 1)
        cells[s.name] = (kinds.index(s.kind), col)
    return cells, len(kinds), n_cols


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


class AffordanceWorker:
    """Run the affordance model on the newest up color frame, in its own thread.

    Parameters
    ----------
    ckpt : Path
        AffordGlide checkpoint.
    heat_out, bev_out : StreamMonitor
        Receive the overlay and BEV images (``rgb8``), stamped with the color frame.

    Notes
    -----
    Callbacks only store frames (:meth:`add_color`, :meth:`add_depth`, :meth:`set_k`),
    so the model never blocks the ROS executor. The thread takes the newest color
    frame (older ones are skipped if it falls behind), pairs it with the depth frame
    nearest in header stamp (the up camera's depth often lags color by 33-65 ms) and
    needs the color ``CameraInfo`` for the BEV; until then only the overlay is shown.
    One update costs ~20 ms (model, GPU) + ~150 ms (BEV floor fit and projection).
    """

    def __init__(self, ckpt: Path, heat_out: StreamMonitor, bev_out: StreamMonitor) -> None:
        from src.model.infer import HeatmapPredictor
        self.predict = HeatmapPredictor(ckpt)
        self.predict(np.zeros((360, 640, 3), np.uint8))     # warm up before frames arrive
        self.heat_out, self.bev_out = heat_out, bev_out
        self._color: tuple[float, np.ndarray] | None = None
        self._depth: deque[tuple[float, np.ndarray]] = deque(maxlen=12)
        self._k: np.ndarray | None = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2.0)

    def add_color(self, img: np.ndarray, stamp: float) -> None:
        """Newest up color frame, (H, W, 3) uint8 RGB; replaces any not yet processed."""
        with self._cv:
            self._color = (stamp, img)
            self._cv.notify()

    def add_depth(self, img: np.ndarray, stamp: float) -> None:
        """Up depth frame, (H, W) uint16 mm."""
        with self._cv:
            self._depth.append((stamp, img))

    def set_k(self, k: np.ndarray) -> None:
        """Color intrinsics (3, 3), px. Depth is registered to color."""
        with self._cv:
            self._k = k

    def _run(self) -> None:
        from src.model.bev import UpFrame
        from src.model.infer import bev_image, overlay
        while True:
            with self._cv:
                while self._color is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                stamp, rgb = self._color
                self._color = None
                depth = min(self._depth, key=lambda d: abs(d[0] - stamp), default=None)
                k = self._k
            heat: np.ndarray = self.predict(rgb)                   # (H, W) in [0, 1]
            self.heat_out.add(overlay(rgb, heat), "rgb8", stamp)
            if depth is not None and k is not None and depth[1].shape == rgb.shape[:2]:
                frame = UpFrame(0.0, rgb, depth[1], k, depth[0] - stamp)
                img, _ = bev_image(frame, heat)
                self.bev_out.add(img, "rgb8", stamp)


def build_window(monitors: Sequence[StreamMonitor], depth_max: float,
                 ) -> tuple["matplotlib.figure.Figure", Callable[[int], list]]:
    """Make the viewer figure and the function that refreshes its panels.

    Parameters
    ----------
    monitors : sequence of StreamMonitor
        One panel each, placed by :func:`grid_layout`.
    depth_max : float
        Depth colour range, m.

    Returns
    -------
    fig : matplotlib.figure.Figure
    update : callable
        ``update(frame_index) -> []``: redraws each panel whose stream has a new frame
        since the last call (the signature ``FuncAnimation`` expects).
    """
    import matplotlib.pyplot as plt
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
            fig.colorbar(ScalarMappable(Normalize(0, depth_max), "turbo"), ax=ax,
                         shrink=0.85, label="depth (m), black = no reading")
        panels.append((ax, ax.imshow(np.zeros((2, 2, 3), np.uint8)), [-1]))

    def update(_: int) -> list:
        for m, (ax, im, shown) in zip(monitors, panels):
            item = m.latest()
            if item is None or item[0] == shown[0]:
                continue          # this stream has nothing new; others may
            k, img = item
            shown[0] = k
            rgb = to_display(img, m.encoding, depth_max)
            if im.get_array().shape[:2] != rgb.shape[:2]:
                im.set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
            im.set_data(rgb)
            lat = m.last_latency()
            ax.set_title(f"{panel_label(m.stream)}   frame {k}   "
                         f"{m.fps():.1f} fps   latency {lat * 1e3:.1f} ms",
                         fontsize=10)
        return []

    return fig, update


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
                   help="no window; just collect stats (use with --plot), or draw the "
                        "window offscreen with --record")
    p.add_argument("--record", type=Path,
                   help="save the window as a video (.mp4, or .avi for Motion JPEG)")
    p.add_argument("--record-fps", type=float, default=15.0, help="video frame rate")
    p.add_argument("--thumbs", type=int, default=6, help="frames per stream in the report")
    p.add_argument("--depth-max", type=float, default=5.0, help="depth colour range, m")
    p.add_argument("--no-detections", action="store_true",
                   help="don't run ObjectDetection on color frames (no detection row)")
    p.add_argument("--affordance", nargs="?", const="latest", default=None, metavar="CKPT",
                   help="replace the detection row with the affordance heatmap (left) and "
                        "its BEV (right); CKPT defaults to the newest data/runs/*/best.pt")
    return p.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m glide_bridge.ros.viewer [options]``."""
    args, ros_args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    from .image_msgs import image_to_numpy, stamp_to_sec

    sources = [StreamMonitor(GLIDE_STREAMS[name], args.thumbs)
               for name in dict.fromkeys(args.streams)]
    # Color stream name -> monitor of its detector output (third row).
    detections: dict[str, StreamMonitor] = {}
    detector = None
    worker: AffordanceWorker | None = None
    affordance: list[StreamMonitor] = []
    if args.affordance is not None:
        from src.model.infer import latest_checkpoint
        ckpt = latest_checkpoint() if args.affordance == "latest" else Path(args.affordance)
        print(f"affordance model: {ckpt}")
        up_c, up_d = GLIDE_STREAMS["up_color"], GLIDE_STREAMS["up_depth"]
        affordance = [StreamMonitor(CameraStream("up_heatmap", up_c.base_topic, "affordance"),
                                    args.thumbs),
                      StreamMonitor(CameraStream("up_bev", up_d.base_topic, "affordance"),
                                    args.thumbs)]
        worker = AffordanceWorker(ckpt, *affordance)
    elif not args.no_detections:
        from src.img_processing.obj_detection import ObjectDetection
        detector = ObjectDetection()
        # Warm up (model load to GPU, first-call setup) before frames arrive.
        detector.apply_bounding_boxes(np.zeros((360, 640, 3), np.uint8))
        detections = {m.stream.name: StreamMonitor(detection_stream(m.stream), args.thumbs)
                      for m in sources if m.stream.kind == "color"}
    monitors = sources + list(detections.values()) + affordance

    rclpy.init(args=["viewer", *ros_args])
    node = Node("camera_stream_viewer")
    for m in sources:
        det = detections.get(m.stream.name)

        def on_image(msg: Image, m: StreamMonitor = m, det: StreamMonitor | None = det) -> None:
            img: np.ndarray = image_to_numpy(msg)        # (H, W, 3) uint8 RGB for color
            stamp = stamp_to_sec(msg.header.stamp)
            m.add(img, msg.encoding, stamp)
            if det is not None:
                # Copy so a detector that draws in place can't alter the color panel.
                # The detection panel's latency includes the detector's run time.
                boxed: np.ndarray = detector.apply_bounding_boxes(img.copy())
                det.add(boxed, "rgb8", stamp)
        node.create_subscription(Image, m.stream.base_topic, on_image,
                                 qos_profile_sensor_data)
        node.get_logger().info(f"{m.stream.name}: subscribed to {m.stream.base_topic}"
                               + (f" (+ {det.stream.name})" if det else ""))
    if worker is not None:
        from sensor_msgs.msg import CameraInfo
        up_c, up_d = GLIDE_STREAMS["up_color"], GLIDE_STREAMS["up_depth"]
        node.create_subscription(
            Image, up_c.base_topic,
            lambda msg: worker.add_color(image_to_numpy(msg), stamp_to_sec(msg.header.stamp)),
            qos_profile_sensor_data)
        node.create_subscription(
            Image, up_d.base_topic,
            lambda msg: worker.add_depth(image_to_numpy(msg), stamp_to_sec(msg.header.stamp)),
            qos_profile_sensor_data)
        if up_c.info_topic:
            node.create_subscription(
                CameraInfo, up_c.info_topic,
                lambda msg: worker.set_k(np.array(msg.k, float).reshape(3, 3)),
                qos_profile_sensor_data)
        worker.start()
        node.get_logger().info(f"affordance: {up_c.base_topic} + {up_d.base_topic} "
                               f"+ {up_c.info_topic}")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin() -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass                  # Ctrl-C / SIGINT shut the context down: normal stop

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()
    t_start = time.time()

    recorder = VideoRecorder(args.record, args.record_fps) if args.record else None
    try:
        if args.headless and recorder is not None:
            # Offscreen: refresh, draw and grab on our own clock.
            fig, update = build_window(monitors, args.depth_max)
            started = False
            while rclpy.ok() and (args.duration is None
                                  or time.time() - t_start < args.duration):
                t_next = time.time() + 1 / args.record_fps
                update(0)
                started = started or any(m.recv for m in monitors)
                if started:               # start the video at the first frame
                    fig.canvas.draw()
                    recorder.grab(fig)
                time.sleep(max(t_next - time.time(), 0.0))
        elif args.headless:
            while rclpy.ok() and (args.duration is None
                                  or time.time() - t_start < args.duration):
                time.sleep(0.1)
        else:
            fig, update = build_window(monitors, args.depth_max)
            # Poll at ~30 Hz and redraw a panel only when its stream has a new frame,
            # so each panel runs at its own camera's rate.
            anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
            if recorder is not None:
                # Grab what was last drawn; starts once any stream has delivered.
                rec_timer = fig.canvas.new_timer(interval=int(1000 / args.record_fps))
                rec_timer.add_callback(
                    lambda: recorder.grab(fig) if any(m.recv for m in monitors) else None)
                rec_timer.start()
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
        if worker is not None:
            worker.stop()
        if recorder is not None:
            recorder.close()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        # Exiting while the spin thread is still inside rclpy aborts the process
        # ("terminate called without an active exception").
        spin_thread.join(timeout=2.0)
        node.destroy_node()

    for m in monitors:
        print(f"{m.stream.name}: received {len(m.recv)} frames on {m.stream.base_topic}")
    if args.plot:
        matplotlib.use("Agg", force=True)
        save_report(monitors, args.plot, args.depth_max)
    return 0 if any(m.recv for m in monitors) else 1


if __name__ == "__main__":
    sys.exit(main())
