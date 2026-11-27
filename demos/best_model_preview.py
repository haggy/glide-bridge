"""Show a trained AffordGlide's heatmaps on validation frames, plus one frame in BEV.

Top rows (``--n`` frames, spread evenly over the validation recording, ``--empty`` of
them without seats): the frame with the predicted heatmap on top (transparent where
cold), the target's outline at 0.5 (white), kept label boxes (green), ignored boxes
(red, dashed) and the predicted peak (cyan cross).

Bottom row, one frame (``--bev-t``; default: the shown frame with the most kept seats):

- left: the raw up-camera RGB read from the bag (as recorded, not the extracted PNG),
  run through the model, with the predicted heatmap overlaid;
- right: the bird's-eye view of that heatmap, projected with the bag's up-camera depth
  (:mod:`src.model.bev`), over a gray BEV of the scene. White outline: the target
  heatmap at 0.5 in the BEV. Cyan triangle: the camera.

Examples
--------
::

    uv run python demos/best_model_preview.py data/runs/2026-11-27_combined_lr_3e-4
    uv run python demos/best_model_preview.py RUN --bev-t 61.5 --ckpt last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as pa    # noqa: E402
import matplotlib.pyplot as plt    # noqa: E402
import numpy as np                 # noqa: E402
import torch                       # noqa: E402
from matplotlib.axes import Axes   # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.model.affordance import AffordGlide            # noqa: E402
from src.model.bev import XLIM, YLIM, bev, load_up_frames  # noqa: E402
from src.model.dataset import (FRAME_SIZE, VAL_RECORDINGS, FrameLabel,  # noqa: E402
                               SeatHeatmapDataset, load_sources, split_by_recording)
from src.model.main import to_input                       # noqa: E402
from src.model.train import run_model                     # noqa: E402

CMAP = matplotlib.colormaps["inferno"]


def evenly(idx: list[int], n: int) -> list[int]:
    """``n`` items of ``idx`` spread evenly from first to last."""
    if n >= len(idx):
        return idx
    return [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, n)]


def crop(x: torch.Tensor) -> torch.Tensor:
    """Remove the stride padding from the last two dims (split evenly, as in ``to_input``)."""
    top = (x.shape[-2] - FRAME_SIZE[1]) // 2
    left = (x.shape[-1] - FRAME_SIZE[0]) // 2
    return x[..., top:top + FRAME_SIZE[1], left:left + FRAME_SIZE[0]]


def predict(model: AffordGlide, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """Heatmap for one padded (3, H', W') input: (H, W) in [0, 1], padding cropped."""
    with torch.no_grad():
        out: torch.Tensor = run_model(model, x[None].to(device), True)
    return crop(out.reshape(x.shape[1:]).cpu()).clamp(0, 1).numpy()


def draw_frame(ax: Axes, rgb: np.ndarray, h: np.ndarray, t: np.ndarray, lab: FrameLabel,
               title: str, fontsize: int = 8) -> None:
    """Frame, heat overlay, target outline, label boxes and the predicted peak."""
    ax.imshow(rgb)
    ax.imshow(np.ma.masked_less(h, 0.05), cmap=CMAP, vmin=0, vmax=1, alpha=0.6)
    if t.max() > 0:
        ax.contour(t, levels=[0.5], colors="white", linewidths=0.6)
    sx, sy = rgb.shape[1] / lab.width, rgb.shape[0] / lab.height
    for seats, col, ls in ((lab.seats, "lime", "-"), (lab.ignored, "red", "--")):
        for s in seats:
            x1, y1, x2, y2 = s.box
            ax.add_patch(pa.Rectangle((x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy,
                                      fill=False, ec=col, lw=0.8, ls=ls))
    py, px = np.unravel_index(np.argmax(h), h.shape)
    ax.plot(px, py, "c+", ms=9, mew=1.5)
    hit = "hit" if t[py, px] > 0 else ("empty" if t.max() == 0 else "miss")
    ax.set_title(f"{title}  max {h.max():.2f}  {hit}", fontsize=fontsize)
    ax.axis("off")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path, help="training run folder")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--n", type=int, default=15, help="frames in the top rows")
    p.add_argument("--empty", type=int, default=3, help="how many of them are empty frames")
    p.add_argument("--bev-t", type=float, default=None,
                   help="validation frame time (s) for the bottom row; default: most seats")
    p.add_argument("--out", type=Path, default=None, help="default: <run>/preview_best_bev.png")
    args = p.parse_args()

    kept, _ = load_sources(camera="up")
    _, val = split_by_recording(kept, VAL_RECORDINGS)
    val.sort(key=lambda lab: lab.t)
    pos = [i for i, lab in enumerate(val) if lab.seats]
    neg = [i for i, lab in enumerate(val) if not lab.seats]
    pick = sorted(evenly(pos, args.n - args.empty) + evenly(neg, args.empty), key=lambda i: val[i].t)
    labels = [val[i] for i in pick]
    if args.bev_t is None:
        bev_lab = max(labels, key=lambda lab: len(lab.seats))
    else:
        bev_lab = min(val, key=lambda lab: abs(lab.t - args.bev_t))
    ds = SeatHeatmapDataset(labels + [bev_lab])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffordGlide().to(device)
    model.load_state_dict(torch.load(args.run / args.ckpt, map_location=device))
    model.eval()

    cols = 5
    rows = -(-len(labels) // cols)
    fig = plt.figure(figsize=(cols * 3.6, rows * 2.3 + 5.2))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [2.3])
    for k, lab in enumerate(labels):
        img, tgt, _ = ds[k]
        draw_frame(fig.add_subplot(gs[k // cols, k % cols]), crop(img).permute(1, 2, 0).numpy(),
                   predict(model, img, device), crop(tgt).numpy(), lab, f"t{lab.t:g}")

    # bottom row: raw sensor frame from the bag, and its BEV
    frame = load_up_frames(REPO_ROOT / "data/bags" / bev_lab.recording, [bev_lab.t])[bev_lab.t]
    h = predict(model, to_input(frame.rgb)[0], device)          # (H, W) on the raw frame
    t = crop(ds[len(labels)][1]).numpy()                         # target, same frame
    draw_frame(fig.add_subplot(gs[rows, :3]), frame.rgb, h, t, bev_lab,
               f"raw up camera (bag) t{frame.t:.1f}, prediction overlaid", fontsize=10)
    bev_rgb, (bev_h, bev_t), how = bev(frame, [h, t])
    ax = fig.add_subplot(gs[rows, 3:])
    ext = (*XLIM, *YLIM)
    gray = np.where(np.isnan(bev_rgb).any(-1), 0.15, np.nan_to_num(bev_rgb.mean(-1)) * 0.45 + 0.2)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, extent=ext, origin="lower", interpolation="nearest")
    im = ax.imshow(np.ma.masked_less(np.nan_to_num(bev_h), 0.05), cmap=CMAP, vmin=0, vmax=1,
                   extent=ext, origin="lower", interpolation="nearest")
    if np.nanmax(np.nan_to_num(bev_t)) > 0:
        nx, ny = bev_t.shape[1], bev_t.shape[0]
        xs = np.linspace(XLIM[0], XLIM[1], nx)
        ys = np.linspace(YLIM[0], YLIM[1], ny)
        ax.contour(xs, ys, np.nan_to_num(bev_t), levels=[0.5], colors="white", linewidths=0.7)
    ax.plot(0, 0.08, "^", color="cyan", ms=10)
    ax.set(xlim=XLIM, ylim=YLIM, aspect="equal", xlabel="right (m)", ylabel="forward (m)")
    ax.set_title(f"BEV of the predicted heatmap (floor: {how}, "
                 f"depth lag {frame.depth_lag_s * 1e3:.0f} ms)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="predicted heat")

    fig.suptitle(f"{args.run.name}/{args.ckpt} on {VAL_RECORDINGS[0]} (validation): "
                 "heat = prediction, white = target 0.5, green = kept box, red = ignored, "
                 "+ = predicted peak", fontsize=10)
    fig.tight_layout()
    out = args.out or args.run / "preview_best_bev.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
