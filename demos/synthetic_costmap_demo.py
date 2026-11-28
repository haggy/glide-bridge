"""Preview synthetic costmaps (``src.nav.synthetic_costmap``) next to the up camera.

Each panel pair shows the up-camera frame nearest the costmap's stamp, and the 7 x 7 m
local costmap in odom (as published: x right, y up). Colors: dark gray = unknown,
white = free, orange = inscribed (253), black = lethal (254). Cyan arrow = robot
(``base_footprint``) and its heading; cyan circle = 0.2 m robot radius.

Examples
--------
::

    PYTHONPATH=. uv run python demos/synthetic_costmap_demo.py \\
        data/synthetic/rosbag2_alpha_033_2026-11-22_04-59-27
    # -> <dir>/preview.png (8 moments); add --video for <dir>/preview.mp4 (every frame)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402
import numpy as np                 # noqa: E402
from matplotlib.axes import Axes   # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glide_bridge.mcap_utils import iter_images  # noqa: E402
from src.nav.synthetic_costmap import RES, SIZE_M  # noqa: E402

UP_COLOR: str = "/camera_up/color/image_raw_rfps"


def cost_rgb(cost: np.ndarray) -> np.ndarray:
    """(n, n) Nav2 costs -> (n, n, 3) uint8: unknown gray, free white, inflated ramp,
    inscribed orange, lethal black."""
    lut = np.zeros((256, 3), np.uint8)
    ramp = np.linspace(0, 1, 253)[:, None]
    lut[:253] = (255 * (1 - ramp) + ramp * np.array([230, 120, 30])).astype(np.uint8)
    lut[253] = (240, 140, 20)
    lut[254] = (0, 0, 0)
    lut[255] = (70, 70, 70)
    return lut[cost]


def color_frames(bag: Path, stamps: np.ndarray) -> list[np.ndarray | None]:
    """Up color frame nearest each stamp (ns), or None if the bag has none within 0.5 s."""
    found: list[tuple[int, np.ndarray]] = []
    want = np.sort(np.unique(stamps))
    for _, t, img in iter_images(bag, [UP_COLOR]):
        if np.abs(want - t).min() < 2e8:
            found.append((t, img))
    if not found:
        return [None] * len(stamps)
    ts = np.array([f[0] for f in found])
    out: list[np.ndarray | None] = []
    for s in stamps:
        i = int(np.abs(ts - s).argmin())
        out.append(found[i][1] if abs(ts[i] - s) < 5e8 else None)
    return out


def draw_costmap(ax: Axes, cost: np.ndarray, origin: np.ndarray, pose: np.ndarray) -> None:
    """Costmap in odom with the robot arrow."""
    n = cost.shape[0]
    ext = (origin[0], origin[0] + n * RES, origin[1], origin[1] + n * RES)
    ax.imshow(cost_rgb(cost), origin="lower", extent=ext, interpolation="nearest")
    x, y, yaw = pose
    ax.arrow(x, y, 0.6 * np.cos(yaw), 0.6 * np.sin(yaw), color="cyan", width=0.04,
             length_includes_head=True)
    ax.add_patch(plt.Circle((x, y), 0.2, fill=False, ec="cyan", lw=1))
    ax.set(xlim=ext[:2], ylim=ext[2:], aspect="equal")
    ax.tick_params(labelsize=6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dir", type=Path, help="data/synthetic/<bag>")
    p.add_argument("--n", type=int, default=8, help="moments in preview.png")
    p.add_argument("--video", action="store_true", help="also write preview.mp4 (every frame)")
    args = p.parse_args()
    meta = json.loads((args.dir / "meta.json").read_text())
    d = np.load(args.dir / "costmaps.npz")
    bag = REPO_ROOT / "data/bags" / meta["bag"]
    T = len(d["cost"])
    known = (d["cost"] != 255).mean((1, 2))
    lethal = (d["cost"] == 254).mean((1, 2))
    print(f"{meta['bag']}: {T} costmaps, known {known.mean():.0%} of the window on average, "
          f"lethal {lethal.mean():.1%}; floor sources {meta['floor_sources']}")

    pick = np.linspace(T * 0.05, T - 1, args.n).astype(int)
    frames = color_frames(bag, d["stamp_ns"][pick])
    fig, axes = plt.subplots(2, args.n, figsize=(2.6 * args.n, 6),
                             gridspec_kw={"height_ratios": [0.6, 1]})
    for j, i in enumerate(pick):
        a = axes[0, j]
        if frames[j] is not None:
            a.imshow(frames[j])
        a.set_title(f"t {(d['stamp_ns'][i] - d['stamp_ns'][0]) / 1e9:.1f} s ({d['camera'][i]})",
                    fontsize=8)
        a.axis("off")
        draw_costmap(axes[1, j], d["cost"][i], d["origin"][i], d["pose"][i])
        axes[1, j].set_title(f"known {known[i]:.0%}, floor {d['floor'][i]}", fontsize=7)
    fig.suptitle(f"Synthetic local costmap ({SIZE_M:g} m, {RES} m cells), {meta['bag']}: "
                 "gray unknown, white free, orange inscribed, black lethal, cyan robot", fontsize=9)
    fig.tight_layout()
    out = args.dir / "preview.png"
    fig.savefig(out, dpi=100)
    print(f"saved {out}")

    if args.video:
        frames = color_frames(bag, d["stamp_ns"])
        vw: cv2.VideoWriter | None = None
        path = args.dir / "preview.mp4"
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
        for i in range(T):
            for a in ax2:
                a.clear()
            if frames[i] is not None:
                ax2[0].imshow(frames[i])
            ax2[0].axis("off")
            ax2[0].set_title(f"up camera  t {(d['stamp_ns'][i] - d['stamp_ns'][0]) / 1e9:.1f} s")
            draw_costmap(ax2[1], d["cost"][i], d["origin"][i], d["pose"][i])
            ax2[1].set_title(f"costmap (update from {d['camera'][i]})", fontsize=9)
            fig2.canvas.draw()
            img = np.asarray(fig2.canvas.buffer_rgba())[..., :3]
            if vw is None:
                vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6,
                                     (img.shape[1], img.shape[0]))
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if vw is not None:
            vw.release()
            print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
