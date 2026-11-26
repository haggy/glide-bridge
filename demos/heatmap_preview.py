#!/usr/bin/env python3
"""Visual check of :class:`AffordGlide`: camera frames next to the predicted heatmaps.

Saves a figure with one row per frame: the RGB frame on the left and the model's
heatmap on the right, cropped to the frame (the 12 px of padding above and below is
removed). Each heatmap has its own color scale, printed in the panel title.

Frames are PNGs saved by ``model-tests/qwen-autolabel/extract_frames.py``. The default
picks have chairs in view.

Examples
--------
::

    uv run python demos/heatmap_preview.py
    uv run python demos/heatmap_preview.py --frames a.png b.png --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import torch
from PIL import Image

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))   # for `src`

from src.model.main import to_input  # noqa: E402

FRAMES_DIR: Path = REPO_ROOT.parent / "model-tests/qwen-autolabel/data/frames_1121_all"
BAG_NAME: str = "rosbag2_alpha_033_2026-11-21_08-29-39"
DEFAULT_PICKS: tuple[str, ...] = ("up_t18.5", "up_t26.8", "up_t30.0", "down_t50.7")


def crop_padding(heat: np.ndarray, height: int, width: int) -> np.ndarray:
    """Remove the padding :func:`to_input` added around a frame.

    Parameters
    ----------
    heat : np.ndarray
        (H', W') heatmap at the padded size.
    height, width : int
        Original frame size in pixels.

    Returns
    -------
    np.ndarray
        (height, width) heatmap aligned with the frame.

    Notes
    -----
    ``to_input`` splits the padding evenly, with the smaller half first:
    ``top = (H' - height) // 2``, ``left = (W' - width) // 2``.
    """
    top: int = (heat.shape[0] - height) // 2
    left: int = (heat.shape[1] - width) // 2
    return heat[top: top + height, left: left + width]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model on each frame and save the side-by-side figure."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frames", type=Path, nargs="*", default=None,
                   help="PNG frames (default: chair frames from the 11-21 bag)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/plots/heatmap_preview.png")
    p.add_argument("--seed", type=int, default=0, help="torch seed for the untrained weights")
    p.add_argument("--show", action="store_true", help="also open a window")
    p.add_argument("--pad-head", action="store_true",
                   help="set padding=k//2 on heatmap_head convs in memory (model file unchanged)")
    args = p.parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.model.affordance import AffordGlide

    frames: list[Path] = args.frames or [FRAMES_DIR / f"{BAG_NAME}_{k}.png" for k in DEFAULT_PICKS]
    missing = [f for f in frames if not f.exists()]
    if missing:
        print(f"missing frames: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    model = AffordGlide()
    model.eval()
    if args.pad_head:
        for m in model.heatmap_head.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.padding = tuple(k // 2 for k in m.kernel_size)

    fig, axs = plt.subplots(len(frames), 2, figsize=(12, 3.6 * len(frames)),
                            layout="constrained", squeeze=False)
    for row, path in zip(axs, frames):
        rgb: np.ndarray = np.array(Image.open(path).convert("RGB"))     # (H, W, 3) uint8 RGB, writable
        with torch.no_grad():
            out: torch.Tensor = model(to_input(rgb))                     # (1, H', W') float32
        x_shape = tuple(to_input(rgb).shape[-2:])
        if tuple(out.shape[-2:]) != x_shape:
            print(f"{path.name}: model output {tuple(out.shape[-2:])} != input {x_shape}; "
                  "a conv without padding shrinks the map", file=sys.stderr)
            return 1
        heat: np.ndarray = crop_padding(out[0].cpu().numpy(), *rgb.shape[:2])   # (H, W)
        print(f"{path.name}: heatmap {heat.shape}, min {heat.min():.3g}, "
              f"max {heat.max():.3g}, mean {heat.mean():.3g}")

        name = path.stem.removeprefix(f"{BAG_NAME}_")
        row[0].imshow(rgb)
        row[0].set_title(f"{name}: raw")
        im = row[1].imshow(heat, cmap="inferno")
        row[1].set_title(f"{name}: heatmap  [{heat.min():.3g}, {heat.max():.3g}]")
        fig.colorbar(im, ax=row[1], fraction=0.03)
        for ax in row:
            ax.axis("off")

    fig.suptitle(f"AffordGlide, untrained head (seed {args.seed})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"saved {args.out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
