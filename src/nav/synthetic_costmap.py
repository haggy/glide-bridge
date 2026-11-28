"""Synthetic Glide local costmaps from a bag's depth and odometry.

A stand-in for ``/local_costmap/costmap`` until ``./glide_robot.sh nav`` bags exist
(see the README). It follows the robot's own Nav2 local-costmap settings: a 7 x 7 m
rolling window centred on the robot, 0.05 m cells, depth marking up to 3.4 m, ray
clearing 0.3-3.5 m, robot radius 0.2 m, inflation 0.15 m with cost scaling 2.0, and
unknown space tracked. The output uses Nav2's raw cost values, so planner code written
against it carries over to the real costmap.

Examples
--------
::

    PYTHONPATH=. uv run python -m src.nav.synthetic_costmap \\
        data/bags/rosbag2_alpha_033_2026-11-22_04-59-27
    # -> data/synthetic/<bag>/costmaps.npz + meta.json
    uv run python demos/synthetic_costmap_demo.py data/synthetic/<bag>   # preview

Notes
-----
Costs (``nav2_costmap_2d``): 0 free, 1-252 inflated, 253 inscribed (inside the robot
radius of an obstacle), 254 lethal, 255 unknown. :func:`to_occupancy` converts them to
the 0-100 / -1 values of the ``nav_msgs/OccupancyGrid`` on ``/local_costmap/costmap``.

Frames (REP-103): odom x forward at start, y left; ``base_footprint`` is on the floor
under the wheels. Each grid is stored with ``origin`` = odom (x, y) of the corner of
cell ``[0, 0]``; row = y index, column = x index, as in ``OccupancyGrid.data``.

Pipeline per depth frame (up camera, plus down camera where the bag has it):

1. depth -> points (``src.model.bev.to_points``) -> floor plane by RANSAC, accepted only
   if it looks like this camera's mount, else the last good fit, else the nominal mount.
   Height above floor ``h = n . p + cam_h``.
2. Points inside the 0.8 x 0.8 m robot-body box are dropped, as Glidance's
   traversability filter does.
3. Obstacle points: ``OBST_MIN_H < h < MAX_POINT_H`` (things to bump into, up to the
   overhead filter's 1.52 m) or ``h < DROP_H`` (drop-offs, e.g. stairs down).
4. The camera pose in odom comes from ``/odom`` interpolated at the depth stamp, plus
   the camera's offset from ``base_footprint`` (:data:`CAM_IN_BASE`).
5. As in Nav2's obstacle layer: cells outside the rolling window are forgotten; rays
   from the camera to every point (floor included) clear cells from RAY_MIN to
   min(range, RAY_MAX); obstacle points within OBST_MAX_RANGE are then marked lethal.
6. The window is cropped and inflated: inscribed (253) within ``ROBOT_RADIUS`` of a
   lethal cell, ``252 * exp(-COST_SCALING * (d - ROBOT_RADIUS))`` out to
   ``INFLATION_RADIUS``.

Differences from the robot (approximations):

- Glidance's traversability filter is closed source. Here an obstacle is a simple
  height threshold above the fitted floor. Ramps, curbs and steps will differ, and
  the real layer's ``min/max_ignore_obstacle_height`` of -0.5/0.5 hints that the real
  clouds encode more than plain height.
- No segmentation, crossing or free-space layers. Those only matter outdoors.
- The camera offset comes from the URDF chain (about 0.5 m behind the wheels; the
  floor fits give the height). Yaw is assumed to match the base.
- ``/odom``: use the pose, not the twist. In the bags, ``twist.linear.x`` reads about
  3.4x the speed implied by the pose (whose yaw rate matches the IMU).
- The up camera alone can't see the floor within about 1.3 m in front of the wheels,
  so that area fills in only from memory as the robot moves (the down camera covers it
  when its depth is in the bag).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from glide_bridge.mcap_utils import image_to_numpy, mcap_files, stamp_ns
from src.model.bev import fit_floor, to_points

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
OUT_ROOT: Path = REPO_ROOT / "data/synthetic"

# Glidance local costmap (glide_nav2_params_custom.yaml) and filters (glide_alpha2.yaml)
SIZE_M: float = 7.0
RES: float = 0.05
ROBOT_RADIUS: float = 0.2
INFLATION_RADIUS: float = 0.15
COST_SCALING: float = 2.0
OBST_MIN_RANGE, OBST_MAX_RANGE = 0.1, 3.4          # m, marking
RAY_MIN, RAY_MAX = 0.3, 3.5                        # m, clearing
MAX_POINT_H: float = 1.52                          # m, overhead_point_filter max_point_height
BODY_CROP: tuple[float, float] = (0.8, 0.8)        # m, traversability robot_body_crop (l, w)
# Approximations of the closed-source traversability filter
OBST_MIN_H: float = 0.10                           # m above floor
DROP_H: float = -0.15                              # m below floor

FREE, INSCRIBED, LETHAL, UNKNOWN = 0, 253, 254, 255


@dataclass(frozen=True)
class CameraModel:
    """One camera: bag topics, mount offset from ``base_footprint`` and floor-fit limits.

    ``pitch`` is degrees *down* from horizontal, as in ``src.model.bev.NOMINAL_FLOOR``.
    """

    name: str
    depth_topic: str
    info_topic: str
    xy_in_base: tuple[float, float]           # m, (forward, left) of base_footprint
    nominal: tuple[float, float]              # (pitch deg down, height m)
    pitch_ok: tuple[float, float]             # accepted fitted pitch, deg down
    height_ok: tuple[float, float] = (0.55, 0.85)


# URDF chain (glide_alpha_base.urdf.xacro): stem pitched back 0.85 rad from the base top,
# cameras ~0.30 m up the stem -> ~0.5 m behind the wheel axle, ~0.63-0.67 m high.
CAM_IN_BASE: tuple[float, float] = (-0.50, 0.0)
CAMERAS: tuple[CameraModel, ...] = (
    CameraModel("up", "/camera_up/depth/image_raw_rfps", "/camera_up/depth/camera_info",
                CAM_IN_BASE, nominal=(-7.5, 0.67), pitch_ok=(-20.0, 5.0)),
    CameraModel("down", "/camera_down/depth/image_raw_rfps", "/camera_down/depth/camera_info",
                CAM_IN_BASE, nominal=(32.0, 0.70), pitch_ok=(20.0, 45.0)),
)


def floor_for(cam: CameraModel, pts: np.ndarray, rng: np.random.Generator,
              last: tuple[np.ndarray, float] | None) -> tuple[np.ndarray, float, str]:
    """Floor ``(up unit vector in the optical frame, camera height m, source)``.

    ``source`` is ``"fit"``, ``"last"`` (previous good fit) or ``"nominal"``.
    """
    fl = fit_floor(pts[pts[:, 2] < 6], rng)
    if fl is not None:
        n, h, inl = fl
        pitch_down = float(np.degrees(np.arcsin(-n[2])))      # same convention as bev.up_floor
        if (cam.height_ok[0] <= h <= cam.height_ok[1]
                and cam.pitch_ok[0] <= pitch_down <= cam.pitch_ok[1] and inl >= 0.08):
            return n, h, "fit"
    if last is not None:
        return last[0], last[1], "last"
    p, h = np.radians(cam.nominal[0]), cam.nominal[1]
    # camera pitched down by p: floor normal (up) in the optical frame (x right, y down,
    # z forward) is (0, -cos p, -sin p)
    return np.array([0.0, -np.cos(p), -np.sin(p)]), h, "nominal"


def ground_coords(pts: np.ndarray, n: np.ndarray, cam_h: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points -> (forward m, left m, height above floor m) under the camera."""
    fwd = np.array([0, 0, 1.0]) - n[2] * n
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, n)
    return pts @ fwd, -(pts @ right), pts @ n + cam_h


def to_occupancy(cost: np.ndarray) -> np.ndarray:
    """Nav2 raw costs -> ``OccupancyGrid`` values (Costmap2DPublisher's table).

    0 -> 0, 253 -> 99, 254 -> 100, 255 -> -1, 1..252 -> 1 + 97 (c - 1) / 251.
    """
    c = cost.astype(np.int16)
    out = np.where(c == 0, 0, 1 + (97 * (c - 1)) // 251)
    out = np.where(c == INSCRIBED, 99, out)
    out = np.where(c == LETHAL, 100, out)
    return np.where(c == UNKNOWN, -1, out).astype(np.int8)


@dataclass
class RollingCostmap:
    """Nav2-style obstacle layer on a fixed odom grid, read out as a rolling window.

    Parameters
    ----------
    lo, hi : np.ndarray
        (2,) odom xy bounds the robot's window can reach, m.
    """

    lo: np.ndarray
    hi: np.ndarray
    state: np.ndarray = field(init=False)   # (ny, nx) int8: -1 unknown, 0 free, 1 obstacle

    def __post_init__(self) -> None:
        self.lo = np.floor(self.lo / RES) * RES
        shape = np.ceil((self.hi - self.lo) / RES).astype(int)[::-1]
        self.state = np.full(tuple(shape), -1, np.int8)
        self.n_win = int(round(SIZE_M / RES))

    def _cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.floor((xy - self.lo) / RES).astype(int)
        return ij[..., 1], ij[..., 0]          # row (y), col (x)

    def window_origin(self, robot_xy: np.ndarray) -> np.ndarray:
        """Odom xy of the window's cell [0, 0] corner, snapped to the grid."""
        return self.lo + np.round((robot_xy - SIZE_M / 2 - self.lo) / RES) * RES

    def update(self, robot_xy: np.ndarray, sensor_xy: np.ndarray, pts_xy: np.ndarray,
               obstacle: np.ndarray, dist: np.ndarray) -> None:
        """Forget outside the window, clear along rays, mark obstacles.

        Parameters
        ----------
        robot_xy, sensor_xy : np.ndarray
            (2,) odom m.
        pts_xy : np.ndarray
            (N, 2) odom m, every point from this frame.
        obstacle : np.ndarray
            (N,) bool.
        dist : np.ndarray
            (N,) horizontal distance from the sensor, m.
        """
        r0, c0 = self._cells(self.window_origin(robot_xy) + RES / 2)
        keep = np.zeros_like(self.state, bool)
        keep[max(r0, 0):r0 + self.n_win, max(c0, 0):c0 + self.n_win] = True
        self.state[~keep] = -1
        # clearing: unique endpoint cells, rays sampled every RES / 2 from RAY_MIN
        ok = dist > RAY_MIN
        d = (pts_xy[ok] - sensor_xy) / dist[ok, None]
        end = np.minimum(dist[ok], RAY_MAX)
        key = np.round(np.c_[d * end[:, None]] / RES).astype(np.int32)
        _, u = np.unique(key, axis=0, return_index=True)
        d, end = d[u], end[u]
        steps = np.arange(RAY_MIN, RAY_MAX + 1e-6, RES / 2)
        s = steps[None, :] * np.ones((len(d), 1))
        valid = s <= end[:, None]
        ray = sensor_xy + d[:, None, :] * s[..., None]                 # (M, S, 2)
        rr, cc = self._cells(ray[valid])
        inb = (rr >= 0) & (rr < self.state.shape[0]) & (cc >= 0) & (cc < self.state.shape[1])
        rr, cc = rr[inb], cc[inb]
        inwin = keep[rr, cc]
        self.state[rr[inwin], cc[inwin]] = 0
        # marking
        m = obstacle & (dist >= OBST_MIN_RANGE) & (dist <= OBST_MAX_RANGE)
        rr, cc = self._cells(pts_xy[m])
        inb = (rr >= 0) & (rr < self.state.shape[0]) & (cc >= 0) & (cc < self.state.shape[1])
        rr, cc = rr[inb], cc[inb]
        inwin = keep[rr, cc]
        self.state[rr[inwin], cc[inwin]] = 1

    def window(self, robot_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(cost (n, n) uint8 Nav2 values, origin (2,) odom m)`` around the robot."""
        org = self.window_origin(robot_xy)
        r0, c0 = self._cells(org + RES / 2)
        n = self.n_win
        st = np.full((n, n), -1, np.int8)
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + n, self.state.shape[0]), min(c0 + n, self.state.shape[1])
        if re > rs and ce > cs:
            st[rs - r0:re - r0, cs - c0:ce - c0] = self.state[rs:re, cs:ce]
        return inflate(st), org


def inflate(state: np.ndarray) -> np.ndarray:
    """Obstacle state (-1/0/1) -> Nav2 costs with inscribed and inflated cells."""
    cost = np.where(state < 0, UNKNOWN, FREE).astype(np.uint8)
    lethal = state == 1
    if lethal.any():
        d = cv2.distanceTransform((~lethal).astype(np.uint8), cv2.DIST_L2, 5) * RES   # m
        infl = np.where(d <= ROBOT_RADIUS, INSCRIBED,
                        np.floor(252 * np.exp(-COST_SCALING * (d - ROBOT_RADIUS))))
        near = (d <= max(INFLATION_RADIUS, 0.0)) & ~lethal
        cost = np.where(near, np.maximum(cost * (cost != UNKNOWN), infl).astype(np.uint8), cost)
        cost[lethal] = LETHAL
    return cost


def read_bag(bag: Path) -> tuple[np.ndarray, list[tuple[int, CameraModel, np.ndarray]],
                                 dict[str, np.ndarray]]:
    """``(odom (N, 4) [t ns, x, y, yaw], depth frames [(stamp ns, camera, mm)], K per camera)``."""
    by_topic = {c.depth_topic: c for c in CAMERAS}
    info = {c.info_topic: c.name for c in CAMERAS}
    odom: list[tuple[float, float, float, float]] = []
    frames: list[tuple[int, CameraModel, np.ndarray]] = []
    k: dict[str, np.ndarray] = {}
    for f in mcap_files(bag):
        with open(f, "rb") as fh:
            r = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _, ch, m, msg in r.iter_decoded_messages(
                    topics=["/odom", *by_topic, *info], log_time_order=True):
                if ch.topic == "/odom":
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
                    odom.append((stamp_ns(msg, m.log_time), p.x, p.y, yaw))
                elif ch.topic in info:
                    k[info[ch.topic]] = np.array(msg.k, float).reshape(3, 3)
                else:
                    frames.append((stamp_ns(msg, m.log_time), by_topic[ch.topic], image_to_numpy(msg)))
    if not odom:
        raise ValueError(f"{bag}: no /odom")
    od = np.array(odom, float)
    od = od[np.argsort(od[:, 0])]
    od[:, 3] = np.unwrap(od[:, 3])
    frames.sort(key=lambda fr: fr[0])
    return od, frames, k


def pose_at(od: np.ndarray, t_ns: int) -> np.ndarray:
    """Interpolated odom ``(x, y, yaw)`` at ``t_ns`` (clamped to the recording)."""
    return np.array([np.interp(t_ns, od[:, 0], od[:, i]) for i in (1, 2, 3)])


def build(bag: Path, out_dir: Path, every: int = 1, seed: int = 0) -> Path:
    """Run the pipeline over a bag and save ``costmaps.npz`` + ``meta.json``.

    Parameters
    ----------
    bag : Path
        Bag folder.
    out_dir : Path
        Output folder (created).
    every : int
        Use every n-th depth frame (all cameras together), for quick runs.
    seed : int
        RANSAC seed.

    Returns
    -------
    Path
        The ``.npz``. Arrays: ``cost`` (T, n, n) uint8 Nav2 costs; ``origin`` (T, 2)
        odom m; ``pose`` (T, 3) robot odom (x, y, yaw); ``stamp_ns`` (T,);
        ``camera`` (T,) which depth frame triggered the update; ``floor`` (T,) floor
        source; ``cam_h`` (T,) fitted camera height, m.
    """
    t0 = time.time()
    od, frames, k = read_bag(bag)
    frames = [fr for fr in frames if fr[1].name in k][::every]
    lo = od[:, 1:3].min(0) - SIZE_M
    hi = od[:, 1:3].max(0) + SIZE_M
    cm = RollingCostmap(lo, hi)
    rng = np.random.default_rng(seed)
    last: dict[str, tuple[np.ndarray, float]] = {}
    rec: dict[str, list] = {key: [] for key in ("cost", "origin", "pose", "stamp_ns",
                                                "camera", "floor", "cam_h")}
    for t, cam, mm in frames:
        pts, _ = to_points(mm / 1000.0, k[cam.name])
        n, cam_h, how = floor_for(cam, pts, rng, last.get(cam.name))
        if how == "fit":
            last[cam.name] = (n, cam_h)
        f, l, h = ground_coords(pts, n, cam_h)
        x, y, yaw = pose_at(od, t)
        bx, by = f + cam.xy_in_base[0], l + cam.xy_in_base[1]          # base_footprint frame
        body = (np.abs(bx) < BODY_CROP[0] / 2) & (np.abs(by) < BODY_CROP[1] / 2)
        keep = ~body & (h < MAX_POINT_H + 0.5)
        bx, by, h, f, l = bx[keep], by[keep], h[keep], f[keep], l[keep]
        c, s = np.cos(yaw), np.sin(yaw)
        pxy = np.c_[x + c * bx - s * by, y + s * bx + c * by]
        sxy = np.array([x + c * cam.xy_in_base[0] - s * cam.xy_in_base[1],
                        y + s * cam.xy_in_base[0] + c * cam.xy_in_base[1]])
        obstacle = ((h > OBST_MIN_H) & (h < MAX_POINT_H)) | (h < DROP_H)
        cm.update(np.array([x, y]), sxy, pxy, obstacle, np.hypot(f, l))
        cost, org = cm.window(np.array([x, y]))
        for key, v in (("cost", cost), ("origin", org), ("pose", (x, y, yaw)), ("stamp_ns", t),
                       ("camera", cam.name), ("floor", how), ("cam_h", cam_h)):
            rec[key].append(v)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "costmaps.npz"
    np.savez_compressed(npz, **{key: np.array(v) for key, v in rec.items()})
    floors = {f"{c}:{w}": int(sum(1 for cc, ww in zip(rec["camera"], rec["floor"]) if cc == c and ww == w))
              for c in {fr[1].name for fr in frames} for w in ("fit", "last", "nominal")}
    meta = {
        "bag": bag.name, "frames": len(rec["cost"]), "every": every,
        "cameras": {c.name: sum(1 for fr in frames if fr[1] is c) for c in CAMERAS},
        "floor_sources": floors,
        "params": {"size_m": SIZE_M, "resolution": RES, "robot_radius": ROBOT_RADIUS,
                   "inflation_radius": INFLATION_RADIUS, "cost_scaling_factor": COST_SCALING,
                   "obstacle_range": [OBST_MIN_RANGE, OBST_MAX_RANGE],
                   "raytrace_range": [RAY_MIN, RAY_MAX], "max_point_height": MAX_POINT_H,
                   "obstacle_min_height": OBST_MIN_H, "drop_height": DROP_H,
                   "camera_in_base": CAM_IN_BASE, "body_crop": BODY_CROP},
        "costs": {"free": FREE, "inscribed": INSCRIBED, "lethal": LETHAL, "unknown": UNKNOWN},
        "frame": "odom; origin = corner of cell [0,0]; row = y, col = x",
        "note": "synthetic stand-in for /local_costmap/costmap (src/nav/synthetic_costmap.py)",
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return npz


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: data/synthetic/<bag>")
    ap.add_argument("--every", type=int, default=1, help="use every n-th depth frame")
    a = ap.parse_args()
    path = build(a.bag, a.out or OUT_ROOT / a.bag.name, a.every)
    print(f"saved {path}")
    print((path.parent / "meta.json").read_text())
    sys.exit(0)
