"""Show a trained AffordGlide's heatmaps on ~20 validation frames.

Each cell: the frame with the predicted heatmap on top (transparent where cold), the
target's outline at 0.5 (white), kept label boxes (green), ignored boxes (red, dashed)
and the predicted peak (cyan cross). Frames are spread evenly over the validation
recording; most have seats, a few are empty.

Examples
--------
::

    uv run python demos/best_model_preview.py data/runs/2026-11-27_combined_lr_3e-4
    uv run python demos/best_model_preview.py RUN --n 30 --empty 6 --ckpt last.pt
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

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.model.affordance import AffordGlide            # noqa: E402
from src.model.dataset import (FRAME_SIZE, VAL_RECORDINGS, SeatHeatmapDataset,  # noqa: E402
                               load_sources, split_by_recording)
from src.model.train import run_model                     # noqa: E402


def evenly(idx: list[int], n: int) -> list[int]:
    """``n`` items of ``idx`` spread evenly from first to last."""
    if n >= len(idx):
        return idx
    return [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, n)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path, help="training run folder")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--n", type=int, default=20, help="frames to show")
    p.add_argument("--empty", type=int, default=4, help="how many of them are empty frames")
    p.add_argument("--out", type=Path, default=None, help="default: <run>/preview_best_20.png")
    args = p.parse_args()

    kept, _ = load_sources(camera="up")
    _, val = split_by_recording(kept, VAL_RECORDINGS)
    val.sort(key=lambda lab: lab.t)
    pos = [i for i, lab in enumerate(val) if lab.seats]
    neg = [i for i, lab in enumerate(val) if not lab.seats]
    pick = sorted(evenly(pos, args.n - args.empty) + evenly(neg, args.empty), key=lambda i: val[i].t)
    labels = [val[i] for i in pick]
    ds = SeatHeatmapDataset(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffordGlide().to(device)
    model.load_state_dict(torch.load(args.run / args.ckpt, map_location=device))
    model.eval()

    cols = 5
    rows = -(-len(ds) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.3))
    for ax in axs.flat:
        ax.axis("off")
    cmap = matplotlib.colormaps["inferno"].copy()
    for k, (ax, lab) in enumerate(zip(axs.flat, labels)):
        img, tgt, valid = ds[k]
        with torch.no_grad():
            pred: torch.Tensor = run_model(model, img[None].to(device), True).reshape(img.shape[1:]).cpu()
        # crop the stride padding (split evenly, as in pad_to_stride)
        top = (img.shape[1] - FRAME_SIZE[1]) // 2
        left = (img.shape[2] - FRAME_SIZE[0]) // 2
        sl = (slice(top, top + FRAME_SIZE[1]), slice(left, left + FRAME_SIZE[0]))
        frame = img[:, sl[0], sl[1]].permute(1, 2, 0).numpy()
        h: np.ndarray = pred[sl].clamp(0, 1).numpy()            # (H, W) in [0, 1]
        t: np.ndarray = tgt[sl].numpy()
        ax.imshow(frame)
        ax.imshow(np.ma.masked_less(h, 0.05), cmap=cmap, vmin=0, vmax=1, alpha=0.6)
        if t.max() > 0:
            ax.contour(t, levels=[0.5], colors="white", linewidths=0.6)
        sx, sy = FRAME_SIZE[0] / lab.width, FRAME_SIZE[1] / lab.height
        for seats, col, ls in ((lab.seats, "lime", "-"), (lab.ignored, "red", "--")):
            for s in seats:
                x1, y1, x2, y2 = s.box
                ax.add_patch(pa.Rectangle((x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy,
                                          fill=False, ec=col, lw=0.8, ls=ls))
        py, px = np.unravel_index(np.argmax(h), h.shape)
        ax.plot(px, py, "c+", ms=9, mew=1.5)
        hit = "hit" if t[py, px] > 0 else ("empty" if t.max() == 0 else "miss")
        ax.set_title(f"t{lab.t:g}  max {h.max():.2f}  {hit}", fontsize=8)
    fig.suptitle(f"{args.run.name}/{args.ckpt} on {VAL_RECORDINGS[0]} (validation): "
                 "heat = prediction, white = target 0.5, green = kept box, red = ignored, "
                 "+ = predicted peak", fontsize=9)
    fig.tight_layout()
    out = args.out or args.run / f"preview_best_{len(ds)}.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
