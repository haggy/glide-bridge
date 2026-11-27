"""Run a trained AffordGlide on single frames and render its heatmap and BEV as images.

Used by the live viewer (``glide_bridge.ros.viewer --affordance``), which needs plain
RGB arrays it can hand to ``imshow``.

Examples
--------
::

    uv run python -m src.model.infer data/runs/2026-11-27_14-15-25/best.pt 30.0
    # -> data/plots/infer_check.png: overlay | BEV for the 11-22 up frame at t = 30.0 s
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

from src.model.affordance import AffordGlide
from src.model.bev import CELL, XLIM, YLIM, UpFrame, bev
from src.model.dataset import FRAME_SIZE
from src.model.main import to_input

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_INFERNO = matplotlib.colormaps["inferno"]


def latest_checkpoint(runs: Path = REPO_ROOT / "data/runs") -> Path:
    """Newest ``best.pt`` under ``runs`` (by modification time).

    Raises
    ------
    FileNotFoundError
        If there is none.
    """
    ckpts = sorted(runs.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"no */best.pt under {runs}; train a model first")
    return ckpts[-1]


class HeatmapPredictor:
    """A trained AffordGlide, ready for one frame at a time.

    Parameters
    ----------
    ckpt : Path
        ``state_dict`` saved by ``src.model.train`` (``best.pt`` / ``last.pt``). It must
        match the current ``AffordGlide`` architecture.
    device : str, optional
        Default: CUDA if available.
    """

    def __init__(self, ckpt: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = AffordGlide().to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        self.ckpt = ckpt

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Heatmap for one frame.

        Parameters
        ----------
        rgb : np.ndarray
            (H, W, 3) uint8 RGB, any size (resized to the training size and back).

        Returns
        -------
        np.ndarray
            (H, W) float32 in [0, 1], on the same pixels as ``rgb``.
        """
        h, w = rgb.shape[:2]
        x: np.ndarray = rgb if (w, h) == FRAME_SIZE else cv2.resize(rgb, FRAME_SIZE,
                                                                  interpolation=cv2.INTER_LINEAR)
        inp: torch.Tensor = to_input(x).to(self.device)           # (1, 3, 384, 640)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):   # forward() prints
            out: torch.Tensor = self.model(inp)
        out = out.reshape(inp.shape[-2:])
        top = (out.shape[0] - FRAME_SIZE[1]) // 2
        left = (out.shape[1] - FRAME_SIZE[0]) // 2
        heat: np.ndarray = (out[top:top + FRAME_SIZE[1], left:left + FRAME_SIZE[0]]
                            .clamp(0, 1).float().cpu().numpy())
        if (w, h) != FRAME_SIZE:
            heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
        return heat


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.6, floor: float = 0.05
            ) -> np.ndarray:
    """Blend the inferno-colored heatmap onto the frame where ``heat >= floor``.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB. ``out = (1 - a) * rgb + a * inferno(heat)``, ``a = alpha``
        on warm pixels, 0 elsewhere. The hottest pixel gets a cyan cross.
    """
    col: np.ndarray = _INFERNO(heat)[..., :3] * 255                # (H, W, 3) float
    a: np.ndarray = np.where(heat >= floor, alpha, 0.0)[..., None]
    out: np.ndarray = ((1 - a) * rgb + a * col).astype(np.uint8)
    py, px = np.unravel_index(int(np.argmax(heat)), heat.shape)
    s = max(4, rgb.shape[1] // 80)
    cv2.line(out, (int(px) - s, int(py)), (int(px) + s, int(py)), (0, 255, 255), 2)
    cv2.line(out, (int(px), int(py) - s), (int(px), int(py) + s), (0, 255, 255), 2)
    return out


def bev_image(frame: UpFrame, heat: np.ndarray, floor: float = 0.05) -> tuple[np.ndarray, str]:
    """BEV of ``heat`` as an RGB image: forward up, right to the right, camera bottom center.

    Parameters
    ----------
    frame : UpFrame
        Color frame with its depth and intrinsics.
    heat : np.ndarray
        (H, W) in [0, 1] on the color pixels.
    floor : float
        Heat below this isn't drawn.

    Returns
    -------
    image : np.ndarray
        (ny, nx, 3) uint8: gray scene (dark = not seen), heat in inferno, a 1 m grid,
        the camera as a cyan triangle. Extent :data:`~src.model.bev.XLIM` x
        :data:`~src.model.bev.YLIM`, :data:`~src.model.bev.CELL` m per pixel.
    floor_source : str
        ``"fit"`` or ``"nominal"`` (see :func:`src.model.bev.up_floor`).
    """
    bev_rgb, (bev_h,), how = bev(frame, [heat])
    seen = ~np.isnan(bev_rgb).any(-1)
    gray: np.ndarray = np.where(seen, np.nan_to_num(bev_rgb.mean(-1)) * 0.45 + 0.2, 0.12)
    img: np.ndarray = np.repeat(gray[..., None], 3, -1)
    h = np.nan_to_num(bev_h)
    hot = h >= floor
    img[hot] = _INFERNO(h[hot])[:, :3]
    img = (img * 255).astype(np.uint8)
    ny, nx = img.shape[:2]
    for m in range(int(np.ceil(XLIM[0])), int(XLIM[1]) + 1):          # 1 m grid
        img[:, min(int((m - XLIM[0]) / CELL), nx - 1)] //= 2
    for m in range(int(YLIM[0]), int(YLIM[1]) + 1):
        img[min(int((m - YLIM[0]) / CELL), ny - 1), :] //= 2
    img = np.ascontiguousarray(img[::-1])                             # forward = up
    cx = int(-XLIM[0] / CELL)
    tri = np.array([[cx, ny - 1], [cx - 6, ny - 1], [cx, ny - 12], [cx + 6, ny - 1]], np.int32)
    cv2.fillPoly(img, [tri], (0, 255, 255))
    return img, how


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from src.model.bev import load_up_frames

    matplotlib.use("Agg")
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_checkpoint()
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    bag = REPO_ROOT / "data/bags/rosbag2_alpha_033_2026-11-22_04-59-27"
    fr = load_up_frames(bag, [t])[t]
    pred = HeatmapPredictor(ckpt)
    heat = pred(fr.rgb)
    b, how = bev_image(fr, heat)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.8, 1]})
    ax[0].imshow(overlay(fr.rgb, heat))
    ax[0].set_title(f"{ckpt.parent.name}: up t{fr.t:.1f}")
    ax[1].imshow(b, extent=(*XLIM, *YLIM))
    ax[1].set(title=f"BEV (floor: {how})", xlabel="right (m)", ylabel="forward (m)")
    out = REPO_ROOT / "data/plots/infer_check.png"
    fig.savefig(out, dpi=90)
    print(f"saved {out}")
