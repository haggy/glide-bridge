"""Run one forward pass of the affordance model on a color frame from a bag.

Examples
--------
::

    uv run python -m src.model.main      # from the repo root
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from glide_bridge.mcap_utils import iter_images
from src.model.affordance import AffordGlide

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BAG: Path = REPO_ROOT / "data/bags/rosbag2_alpha_033_2026-11-21_08-29-39"
TOPIC: str = "/camera_up/color/image_raw_rfps"
STRIDE: int = 32        # P5 stride; YOLO needs H and W divisible by it
PAD_VALUE: float = 114 / 255    # YOLO's letterbox gray


def load_frame(bag: Path, topic: str) -> np.ndarray:
    """Read the first image on ``topic``.

    Parameters
    ----------
    bag : Path
        Bag folder or ``.mcap`` file.
    topic : str
        Color image topic as recorded.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB.

    Raises
    ------
    ValueError
        If the bag has no images on ``topic``.
    """
    for _, _, img in iter_images(bag, [topic]):
        return img
    raise ValueError(f"no images on {topic} in {bag}")


def to_input(rgb: np.ndarray) -> torch.Tensor:
    """Convert an RGB frame to a padded model input.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 RGB.

    Returns
    -------
    torch.Tensor
        (1, 3, H', W') float32 RGB in [0, 1], where H' and W' are H and W rounded up to
        a multiple of ``STRIDE``.

    Notes
    -----
    Padding is split evenly between opposite sides, so 640x360 frames get 12 px above
    and below (640x384). No resizing. Depth must later be padded the same way.
    """
    x: torch.Tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
    pad_h: int = -x.shape[-2] % STRIDE
    pad_w: int = -x.shape[-1] % STRIDE
    top, left = pad_h // 2, pad_w // 2
    return F.pad(x, (left, pad_w - left, top, pad_h - top), value=PAD_VALUE)


if __name__ == "__main__":
    aff_model = AffordGlide()
    aff_model.eval()    # forward() reads YOLO's eval-mode output, a (y, preds) tuple

    img: torch.Tensor = to_input(load_frame(BAG, TOPIC))    # (1, 3, 384, 640)
    with torch.no_grad():
        print(f"Output shape: {aff_model(img).shape}")
