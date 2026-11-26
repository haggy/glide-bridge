"""Sittability heatmap dataset built from the Qwen + SAM auto-labels.

Each sample is a camera frame and its target heatmap, padded the same way as
:func:`src.model.main.to_input`. The target is the per-object distance transform of
the auto-labelled masks: 0 outside every object, rising to 1 at the point of each
object farthest from its edge.

Labels come from ``model-tests/qwen-autolabel/autolabel.py``:

- ``labels.jsonl``: one :class:`FrameLabel` per frame.
- ``masks/<frame stem>_<k>.png``: binary mask of seat ``k`` (255 = object).
- ``row_flags.csv``: frames whose boxes formed a suspiciously regular row.

Examples
--------
Summarize the split::

    uv run python -m src.model.dataset
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.model.main import PAD_VALUE, STRIDE

MODEL_TESTS: Path = Path(__file__).resolve().parents[3] / "model-tests/qwen-autolabel"
FRAMES_DIR: Path = MODEL_TESTS / "data/frames_1121_all"
LABELS_DIR: Path = MODEL_TESTS / "outputs/autolabel/bag_1121_all"

# Rules for leaving frames out of training; see the "Pre-processing" tab of the issue
# tracking doc (PP-1 and on).
GATE_BORDERLINE: tuple[float, float] = (0.2, 0.6)   # p_sit in [lo, hi): unsure frames
SEAT_CAP: int = 12                 # grounding stops at 12 objects, so the rest are missing
DUPLICATE_IOU: float = 0.9         # two boxes this similar are one object labelled twice
EXCLUDED_LABELS: frozenset[str] = frozenset({"bench"})   # in this bag, a wooden counter
# Flagged as a regular row but checked by hand and kept.
ROW_FLAG_KEEP: frozenset[str] = frozenset({"down_t50.7", "up_t18.5", "up_t18.7"})

_NAME = re.compile(r"_(?P<camera>up|down)_t(?P<t>\d+(?:\.\d+)?)\.png$")


@dataclass
class Seat:
    """One labelled sittable object, in pixels of the original frame."""

    label: str
    box: list[float]      # [x1, y1, x2, y2], px
    point: list[float]    # [x, y], px
    mask_area: int = 0    # px


@dataclass
class FrameLabel:
    """Auto-labels for one frame (a row of ``labels.jsonl``)."""

    image: str            # PNG file name
    width: int
    height: int
    p_sit: float          # gate probability that anything sittable is in view
    seats: list[Seat] = field(default_factory=list)
    error: str = ""

    @property
    def camera(self) -> str:
        """``"up"`` or ``"down"``, from the file name."""
        return _parse_name(self.image)[0]

    @property
    def t(self) -> float:
        """Seconds since the camera's first frame in the bag, from the file name."""
        return _parse_name(self.image)[1]

    @property
    def key(self) -> str:
        """Short name such as ``up_t18.5``."""
        cam, t = _parse_name(self.image)
        return f"{cam}_t{t:g}"


def _parse_name(name: str) -> tuple[str, float]:
    """Camera and time from ``..._<camera>_t<seconds>.png``."""
    m = _NAME.search(name)
    if m is None:
        raise ValueError(f"can't read camera and time from {name!r}")
    return m["camera"], float(m["t"])


def load_labels(labels_dir: Path = LABELS_DIR) -> list[FrameLabel]:
    """Read ``labels.jsonl``.

    Parameters
    ----------
    labels_dir : Path
        Autolabel run folder.

    Returns
    -------
    list[FrameLabel]
        In file order.
    """
    out: list[FrameLabel] = []
    with open(labels_dir / "labels.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            seats = [Seat(s["label"], s["box"], s["point"], s.get("mask_area", 0))
                     for s in r["seats"]]
            out.append(FrameLabel(r["image"], r["width"], r["height"], r["p_sit"], seats,
                                  r.get("error", "")))
    return out


def box_iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two ``[x1, y1, x2, y2]`` boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def exclusion_reasons(label: FrameLabel, row_flagged: set[str]) -> list[str]:
    """Why a frame should be left out of training (empty list: keep it).

    Parameters
    ----------
    label : FrameLabel
        The frame's auto-labels.
    row_flagged : set of str
        Image names listed in ``row_flags.csv``.

    Returns
    -------
    list[str]
        Any of ``error``, ``borderline``, ``label``, ``row``, ``cap``, ``duplicate``.
    """
    s = label.seats
    why: list[str] = []
    if label.error:
        why.append("error")
    if GATE_BORDERLINE[0] <= label.p_sit < GATE_BORDERLINE[1]:
        why.append("borderline")
    if any(x.label in EXCLUDED_LABELS for x in s):
        why.append("label")
    if label.image in row_flagged and label.key not in ROW_FLAG_KEEP:
        why.append("row")
    if len(s) >= SEAT_CAP:
        why.append("cap")
    if any(box_iou(s[i].box, s[j].box) > DUPLICATE_IOU for i in range(len(s)) for j in range(i)):
        why.append("duplicate")
    return why


def usable_labels(labels_dir: Path = LABELS_DIR) -> tuple[list[FrameLabel], Counter[str]]:
    """Frames that pass :func:`exclusion_reasons`.

    Returns
    -------
    kept : list[FrameLabel]
    dropped : Counter[str]
        Number of frames dropped for each reason (a frame can count under several).
    """
    flags_csv = labels_dir / "row_flags.csv"
    row_flagged: set[str] = set()
    if flags_csv.exists():
        with open(flags_csv) as fh:
            row_flagged = {r["image"] for r in csv.DictReader(fh)}
    kept: list[FrameLabel] = []
    dropped: Counter[str] = Counter()
    for lab in load_labels(labels_dir):
        why = exclusion_reasons(lab, row_flagged)
        dropped.update(why)
        if not why:
            kept.append(lab)
    return kept, dropped


def split_by_time(labels: list[FrameLabel], val_fraction: float = 0.2,
                  gap_s: float = 2.0) -> tuple[list[FrameLabel], list[FrameLabel]]:
    """Hold out the end of the recording for validation.

    Parameters
    ----------
    labels : list[FrameLabel]
    val_fraction : float
        Share of the recording's time span used for validation.
    gap_s : float
        Frames this close before the validation start are dropped from both sets.

    Returns
    -------
    train, val : list[FrameLabel]

    Notes
    -----
    ``t_val = t_max - val_fraction * (t_max - t_min)``. A random split would put nearly
    identical neighbouring frames (about 0.2 s apart) on both sides. Cutting by time
    keeps them apart, though the robot can still revisit the same chairs later in the
    recording. Up and down frames from the same moment land on the same side.
    """
    ts = [lab.t for lab in labels]
    t_val = max(ts) - val_fraction * (max(ts) - min(ts))
    train = [lab for lab in labels if lab.t < t_val - gap_s]
    val = [lab for lab in labels if lab.t >= t_val]
    return train, val


def seat_heatmap(masks: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Target heatmap from object masks.

    Parameters
    ----------
    masks : list of np.ndarray
        (H, W) bool masks, one per object.
    height, width : int
        Frame size, used when there are no masks.

    Returns
    -------
    np.ndarray
        (H, W) float32 in [0, 1].

    Notes
    -----
    For each mask, ``d = distanceTransform(mask)`` (px to the nearest pixel outside it)
    and ``h_k = d / max(d)``, so every object peaks at 1 whatever its size. Objects
    are combined with ``max_k h_k``, so overlaps don't add up past 1.
    """
    heat: np.ndarray = np.zeros((height, width), np.float32)
    for m in masks:
        d: np.ndarray = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        if d.max() > 0:
            np.maximum(heat, d / d.max(), out=heat)
    return heat


def pad_to_stride(x: torch.Tensor, value: float) -> torch.Tensor:
    """Pad the last two dims up to a multiple of ``STRIDE``, split evenly as in ``to_input``."""
    pad_h: int = -x.shape[-2] % STRIDE
    pad_w: int = -x.shape[-1] % STRIDE
    top, left = pad_h // 2, pad_w // 2
    return F.pad(x, (left, pad_w - left, top, pad_h - top), value=value)


class SeatHeatmapDataset(Dataset):
    """Frames, target heatmaps and valid-pixel masks, all in memory.

    Parameters
    ----------
    labels : list[FrameLabel]
        Frames to use, e.g. from :func:`split_by_time`.
    frames_dir : Path
        Folder with the frame PNGs.
    masks_dir : Path
        Folder with the per-seat mask PNGs.
    flip : bool
        Mirror each sample left-right with probability 0.5 (training augmentation).

    Notes
    -----
    ``__getitem__`` returns ``(image, target, valid)``:

    - image: (3, H', W') float32 RGB in [0, 1], padded with YOLO gray.
    - target: (H', W') float32 heatmap, 0 in the padding.
    - valid: (H', W') bool, True on real frame pixels, so the loss can skip padding.

    with H', W' the frame size rounded up to a multiple of 32 (384x640 here). Targets
    are built once at construction (about 1 MB per frame, image + target).
    """

    def __init__(self, labels: list[FrameLabel], frames_dir: Path = FRAMES_DIR,
                 masks_dir: Path = LABELS_DIR / "masks", flip: bool = False) -> None:
        self.labels = labels
        self.flip = flip
        self.images: list[np.ndarray] = []     # each (H, W, 3) uint8 RGB
        self.targets: list[np.ndarray] = []    # each (H, W) float16
        for lab in labels:
            rgb: np.ndarray = np.array(Image.open(frames_dir / lab.image).convert("RGB"))
            stem = Path(lab.image).stem
            masks = [np.array(Image.open(masks_dir / f"{stem}_{k}.png")) > 127
                     for k in range(len(lab.seats))]
            self.images.append(rgb)
            self.targets.append(seat_heatmap(masks, *rgb.shape[:2]).astype(np.float16))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img: torch.Tensor = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255
        tgt: torch.Tensor = torch.from_numpy(self.targets[i]).float()
        if self.flip and torch.rand(()) < 0.5:
            img, tgt = img.flip(-1), tgt.flip(-1)
        valid: torch.Tensor = torch.ones_like(tgt, dtype=torch.bool)
        return (pad_to_stride(img, PAD_VALUE), pad_to_stride(tgt, 0.0),
                pad_to_stride(valid, False))


if __name__ == "__main__":
    kept, dropped = usable_labels()
    train, val = split_by_time(kept)
    print(f"usable frames: {len(kept)} ({sum(bool(x.seats) for x in kept)} with seats); "
          f"dropped: {dict(dropped)}")
    for name, part in (("train", train), ("val", val)):
        cams = Counter(x.camera for x in part)
        print(f"{name}: {len(part)} frames, t {min(x.t for x in part):.1f}-"
              f"{max(x.t for x in part):.1f} s, {dict(cams)}, "
              f"{sum(len(x.seats) for x in part)} seats, "
              f"{sum(not x.seats for x in part)} empty")
