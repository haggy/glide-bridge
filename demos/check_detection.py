#!/usr/bin/env python3
"""Visual check of ``ObjectDetection.apply_bounding_boxes`` on a known YOLO test image.

Runs the method exactly as the camera viewer does (an ``(H, W, 3)`` uint8 **RGB**
array) and saves a figure with three panels:

1. the input image,
2. the method's output,
3. the method's output with the reference boxes drawn on top as dashed red outlines,
   labelled with class and confidence.

The reference is Ultralytics' own prediction with the same weights, run on the image
file (Ultralytics reads files with OpenCV, so channel order is handled for it). If the
method is right, every green box sits under a red dashed one.

Examples
--------
::

    source env/activate_ros.sh
    python demos/check_detection.py                       # ultralytics' bus.jpg
    python demos/check_detection.py --image path/to.jpg --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `src`


def default_image() -> Path:
    """Path of ``bus.jpg`` shipped with Ultralytics (810x1080, people + bus)."""
    import ultralytics
    return Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and save (or show) the comparison figure."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--image", type=Path, default=None, help="test image (default: bus.jpg)")
    p.add_argument("--out", type=Path, default=Path("data/plots/detection_check.png"))
    p.add_argument("--show", action="store_true", help="also open a window")
    args = p.parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from src.img_processing.obj_detection import ObjectDetection

    path = args.image or default_image()
    rgb: np.ndarray = np.asarray(Image.open(path).convert("RGB"))   # (H, W, 3) uint8 RGB
    det = ObjectDetection()
    out: np.ndarray = det.apply_bounding_boxes(rgb.copy())

    ref = det._model(str(path), verbose=False)[0]
    xyxy: np.ndarray = ref.boxes.xyxy.cpu().numpy()     # (N, 4) float, pixels
    conf: np.ndarray = ref.boxes.conf.cpu().numpy()     # (N,)
    cls: np.ndarray = ref.boxes.cls.cpu().numpy().astype(int)

    print(f"{path.name}: {rgb.shape[1]}x{rgb.shape[0]}, reference boxes (x1, y1, x2, y2):")
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        print(f"  {ref.names[k]:>10s} {c:.2f}  ({x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f})")
    changed = np.any(out != rgb, axis=-1)
    print(f"method changed {changed.mean() * 100:.1f}% of pixels; output {out.shape} {out.dtype}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 7), layout="constrained")
    for ax, img, title in zip(axs, (rgb, out, out),
                              ("input (RGB)", "apply_bounding_boxes",
                               "method (green) vs reference (red dashed)")):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        axs[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red",
                                   lw=1.5, ls="--"))
        axs[2].text(x1, y1 - 4, f"{ref.names[k]} {c:.2f}", color="white", fontsize=8,
                    bbox=dict(fc="red", ec="none", pad=1))
    fig.suptitle(f"{path.name}: {len(xyxy)} reference detections")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(f"saved {args.out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
