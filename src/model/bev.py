"""Bird's-eye view (BEV) of an up-camera frame and a heatmap, from the bag's depth.

Ported from ``model-tests/qwen-autolabel/bev_preview.py`` (same geometry), but the
floor comes from the up camera's own depth: in the 11-22 bag the down camera's depth
has only 88 frames, while up depth is complete.

Examples
--------
::

    uv run python -m src.model.bev 30.0     # BEV of the 11-22 up frame at t = 30.0 s

Notes
-----
Depth (16UC1, mm) is registered to the color camera (same size and K). Points in the
optical frame (x right, y down, z forward), m: ``P = z * K^-1 [u, v, 1]``; lens
distortion ignored. Flying pixels (depth gradient > 5% of depth) and depth outside
0.15 m-:data:`MAX_RANGE` are dropped. Stereo depth error grows with range squared, so
far seats (> ~6 m) land less precisely.

Floor: RANSAC plane on the frame's own points, accepted if it looks like the floor seen
from the robot (camera 0.55-0.85 m above it, pitch -20..+5 deg, >= 8% inliers).
Otherwise :data:`NOMINAL_FLOOR` is used. On the 11-22 bag the up fits give camera
height 0.67 m and pitch -5..-12 deg (camera tilted slightly up), which agrees with the
down camera's fits (0.70 m, ~32 deg down) minus the -38.9 deg mount offset measured on
the 11-21 bag.

BEV axes: x = right, y = forward (camera axis projected on the floor), camera at (0, 0),
cells :data:`CELL` m. RGB keeps the highest point per cell (below 2 m above the floor),
heat the hottest. Cells the camera didn't see stay empty. The up camera's depth often
lags color by 33-65 ms, so moving frames can be off by a few cm.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from glide_bridge.mcap_utils import image_to_numpy, mcap_files, stamp_ns

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
UP_COLOR: str = "/camera_up/color/image_raw_rfps"
UP_DEPTH: str = "/camera_up/depth/image_raw_rfps"
UP_INFO: str = "/camera_up/color/camera_info"
XLIM: tuple[float, float] = (-5.0, 5.0)    # m, right
YLIM: tuple[float, float] = (0.0, 10.0)    # m, forward
CELL: float = 0.05                         # m
MAX_RANGE: float = 10.0                    # m; atrium seats in 11-22 sit 5-10 m away
NOMINAL_FLOOR: tuple[float, float] = (-7.5, 0.67)   # (pitch down deg, camera height m)


@dataclass
class UpFrame:
    """One up-camera color frame with its nearest depth frame."""

    t: float               # s since the first up color frame (log time), as in frame names
    rgb: np.ndarray        # (H, W, 3) uint8 RGB, as recorded
    depth_mm: np.ndarray   # (H, W) uint16 mm, 0 = no reading
    k: np.ndarray          # (3, 3) intrinsics, px
    depth_lag_s: float     # depth stamp - color stamp, s


def load_up_frames(bag: Path, times: list[float]) -> dict[float, UpFrame]:
    """Up color frames nearest to ``times`` (s since the first up color frame), with depth.

    Parameters
    ----------
    bag : Path
        Bag folder or ``.mcap`` file.
    times : list of float
        Frame times as in the frame file names (``..._up_t30.0.png`` -> 30.0).

    Returns
    -------
    dict[float, UpFrame]
        Keyed by the requested time.

    Notes
    -----
    Frame names use ``log_time - first log_time`` of the color topic, so frames are
    matched on that; depth is then matched on the header stamps.
    """
    color: list[tuple[float, int, np.ndarray]] = []     # (t, stamp ns, rgb)
    depth: list[tuple[int, np.ndarray]] = []
    k: np.ndarray | None = None
    t0: float | None = None
    want = np.array(times, float)
    for f in mcap_files(bag):
        with open(f, "rb") as fh:
            r = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _, ch, m, msg in r.iter_decoded_messages(topics=[UP_COLOR, UP_DEPTH, UP_INFO],
                                                          log_time_order=True):
                if ch.topic == UP_INFO:
                    k = np.array(msg.k, float).reshape(3, 3)
                elif ch.topic == UP_DEPTH:
                    depth.append((stamp_ns(msg, m.log_time), image_to_numpy(msg)))
                else:
                    t0 = m.log_time / 1e9 if t0 is None else t0
                    t = m.log_time / 1e9 - t0
                    if len(want) and np.abs(want - t).min() < 0.15:   # keep candidates only
                        color.append((t, stamp_ns(msg, m.log_time), image_to_numpy(msg)))
    if k is None or not depth:
        raise ValueError(f"{bag}: no up-camera depth or camera_info")
    d_ts = np.array([d[0] for d in depth])
    out: dict[float, UpFrame] = {}
    for want_t in times:
        t, st, rgb = min(color, key=lambda c: abs(c[0] - want_t))
        i = int(np.abs(d_ts - st).argmin())
        out[want_t] = UpFrame(t, rgb, depth[i][1], k, (d_ts[i] - st) / 1e9)
    return out


def to_points(z: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth (H, W) m -> ``(N, 3)`` points (m, optical frame) and the (H, W) bool mask used."""
    gy, gx = np.gradient(z)
    good = (z > 0.15) & (z < MAX_RANGE) & (np.hypot(gx, gy) < 0.05 * np.maximum(z, 0.5))
    v, u = np.nonzero(good)
    zz = z[good]
    return np.stack([(u - k[0, 2]) / k[0, 0] * zz, (v - k[1, 2]) / k[1, 1] * zz, zz], 1), good


def fit_floor(pts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float] | None:
    """RANSAC floor plane: ``(up unit vector, camera height m, inlier fraction)`` or None.

    Notes
    -----
    Planes are kept if roughly horizontal (``n_y < -0.6``, ``|n_x| < 0.25``), 0.1-1.5 m
    below the camera, with < 5% of points more than 6 cm under them. The best (most
    points within 2 cm) is refined by least squares on its inliers.
    """
    if len(pts) < 100:
        return None
    p = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
    best: tuple[np.ndarray, float, float] | None = None
    for _ in range(1500):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        n = -n if n[1] > 0 else n                    # up is roughly -y
        cam_h = float(-n @ a)
        if n[1] > -0.6 or abs(n[0]) > 0.25 or not 0.1 < cam_h < 1.5:
            continue
        h = p @ n + cam_h
        inl = float((np.abs(h) < 0.02).mean())
        if (h < -0.06).mean() < 0.05 and (best is None or inl > best[2]):
            best = (n, cam_h, inl)
    if best is None:
        return None
    q = p[np.abs(p @ best[0] + best[1]) < 0.02]
    cen = q.mean(0)
    n = np.linalg.svd(q - cen, full_matrices=False)[2][-1]
    n = -n if n[1] > 0 else n
    return n, float(-n @ cen), best[2]


def up_floor(pts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, str]:
    """Floor for an up frame: ``(up unit vector, camera height m, "fit" | "nominal")``."""
    fl = fit_floor(pts[pts[:, 2] < 6], rng)    # near points: the floor fit is cleaner
    if fl is not None:
        n, h, inl = fl
        pitch = float(np.degrees(np.arcsin(-n[2])))
        if 0.55 <= h <= 0.85 and -20 <= pitch <= 5 and inl >= 0.08:
            return n, h, "fit"
    pitch, h = np.radians(NOMINAL_FLOOR[0]), NOMINAL_FLOOR[1]
    return np.array([0.0, -np.cos(pitch), -np.sin(pitch)]), h, "nominal"


def raster(x: np.ndarray, y: np.ndarray, order: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Splat ``val`` (N, C) into a ``(ny, nx, C)`` grid (NaN = empty); highest ``order`` wins."""
    nx, ny = round((XLIM[1] - XLIM[0]) / CELL), round((YLIM[1] - YLIM[0]) / CELL)
    grid = np.full((ny, nx, val.shape[1]), np.nan)
    s = np.argsort(order)
    i = np.floor((x[s] - XLIM[0]) / CELL).astype(int)
    j = np.floor((y[s] - YLIM[0]) / CELL).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    grid[j[ok], i[ok]] = val[s][ok]
    return grid


def bev(frame: UpFrame, heat: list[np.ndarray], seed: int = 0
        ) -> tuple[np.ndarray, list[np.ndarray], str]:
    """Project a frame and heatmaps of the same size into the BEV.

    Parameters
    ----------
    frame : UpFrame
    heat : list of np.ndarray
        (H, W) maps in [0, 1] on the color pixels (e.g. prediction, target).
    seed : int
        RANSAC seed.

    Returns
    -------
    bev_rgb : np.ndarray
        (ny, nx, 3) float RGB in [0, 1], NaN where empty.
    bev_heat : list of np.ndarray
        (ny, nx) per map: hottest value per cell, NaN where empty or 0.
    floor : str
        ``"fit"`` or ``"nominal"``.
    """
    pts, good = to_points(frame.depth_mm / 1000.0, frame.k)
    n, cam_h, how = up_floor(pts, np.random.default_rng(seed))
    fwd = np.array([0, 0, 1.0]) - n[2] * n
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, n)
    h, x, y = pts @ n + cam_h, pts @ right, pts @ fwd
    keep = (h > -0.1) & (h < 2.0)
    rgb = frame.rgb[good][keep] / 255.0
    bev_rgb = raster(x[keep], y[keep], h[keep], rgb ** 0.6)
    out: list[np.ndarray] = []
    for m in heat:
        v = m[good][keep]
        hot = v > 0
        out.append(raster(x[keep][hot], y[keep][hot], v[hot], v[hot][:, None])[..., 0])
    return bev_rgb, out, how


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bag = REPO_ROOT / "data/bags/rosbag2_alpha_033_2026-11-22_04-59-27"
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    fr = load_up_frames(bag, [t])[t]
    br, _, how = bev(fr, [])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].imshow(fr.rgb)
    ax[0].set_title(f"up t{fr.t:.1f}, depth lag {fr.depth_lag_s * 1e3:.0f} ms")
    ax[1].imshow(np.nan_to_num(br, nan=0.15), extent=(*XLIM, *YLIM), origin="lower")
    ax[1].set_title(f"BEV RGB (floor: {how})")
    out = REPO_ROOT / "data/plots/bev_up.png"
    fig.savefig(out, dpi=90)
    print(f"saved {out}")
