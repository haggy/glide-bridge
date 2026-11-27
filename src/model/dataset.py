"""Sittability heatmap dataset built from the Qwen + SAM auto-labels.

Each sample is a camera frame and its target heatmap, padded the same way as
:func:`src.model.main.to_input`. The target is the per-object distance transform of
the auto-labelled masks: 0 outside every object, rising to 1 at the point of each
object farthest from its edge.

Labels come from ``model-tests/qwen-autolabel/autolabel.py`` runs, one per
:class:`LabelSource` (a frames folder and the run folder labelling it):

- ``labels.jsonl``: one :class:`FrameLabel` per frame.
- ``masks/<frame stem>_<k>.png``: binary mask of seat ``k`` (255 = object).
- ``row_flags.csv``: frames whose boxes formed a suspiciously regular row.
- ``yolo_check.jsonl``: COCO YOLO26n's score for each seat box, from
  :mod:`src.model.yolo_check` (optional; missing means YOLO score 0).

Frames come from bags (``extract_frames.py``) or camera videos
(``extract_video_frames.py``), at 640x360 or 424x240; everything is resized to
:data:`FRAME_SIZE` so batches stack.

Examples
--------
Summarize the sources and the up-camera split::

    uv run python -m src.model.dataset
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
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
FRAME_SIZE: tuple[int, int] = (640, 360)    # (W, H) every frame is resized to


@dataclass(frozen=True)
class LabelSource:
    """A folder of frame PNGs and the autolabel run folder that labelled them."""

    frames_dir: Path
    labels_dir: Path


SOURCES: tuple[LabelSource, ...] = (
    # 11-21 bag, both cameras (up only for the first 42 s).
    LabelSource(FRAMES_DIR, LABELS_DIR),
    # 11-22 04-59-27 bag, up camera: a different building. Default validation set.
    LabelSource(MODEL_TESTS / "data/frames_1122_up", MODEL_TESTS / "outputs/autolabel/bag_1122_up"),
    # Up-camera videos from Drive: 07-10, 07-14, 11-21 08-27-30 and 11-23 01-19-36
    # (the last at every 3rd frame).
    LabelSource(MODEL_TESTS / "data/frames_videos", MODEL_TESTS / "outputs/autolabel/videos_up"),
)
VAL_RECORDINGS: tuple[str, ...] = ("rosbag2_alpha_033_2026-11-22_04-59-27",)

# Rules for leaving frames out of training; see the "Pre-processing" tab of the issue
# tracking doc (PP-1 and on).
GATE_BORDERLINE: tuple[float, float] = (0.2, 0.6)   # p_sit in [lo, hi): unsure empty frames
SEAT_CAP: int = 12                 # grounding stops at 12 objects, so the rest are missing
DUPLICATE_IOU: float = 0.9         # two boxes this similar are one object labelled twice
# Seat kinds we train on. A box with any other label ("wheelchair", "ledge", ...) is
# ignored (left out of the loss), not used as a target. "steps"/"stairs" are real
# staircases in the 11-22 bag: a last-resort seat, but still sittable.
SEAT_LABELS: frozenset[str] = frozenset(
    {"chair", "sofa", "couch", "armchair", "stool", "bench", "steps", "stairs"})
# Kinds COCO has no class for: YOLO can't confirm them, so their score is p_sit alone.
QWEN_ONLY_LABELS: frozenset[str] = frozenset({"stool", "steps", "stairs"})
# Keep a seat box if mean(p_sit, YOLO conf) >= this; see combined_score().
COMBINED_MIN: float = 0.5
# Flagged as a regular row but checked by hand and kept.
ROW_FLAG_KEEP: frozenset[str] = frozenset({"down_t50.7", "up_t18.5", "up_t18.7"})

_NAME = re.compile(r"^(?P<rec>.+)_(?P<camera>up|down)_t(?P<t>\d+(?:\.\d+)?)\.png$")


@dataclass
class Seat:
    """One labelled sittable object, in pixels of the original frame."""

    label: str
    box: list[float]      # [x1, y1, x2, y2], px
    point: list[float]    # [x, y], px
    mask_area: int = 0    # px
    index: int = 0        # k in masks/<frame stem>_<k>.png
    yolo_conf: float = 0.0   # best overlapping COCO seat detection (0: none)
    yolo_label: str = ""


@dataclass
class FrameLabel:
    """Auto-labels for one frame (a row of ``labels.jsonl``)."""

    image: str            # PNG file name
    width: int
    height: int
    p_sit: float          # gate probability that anything sittable is in view
    seats: list[Seat] = field(default_factory=list)
    error: str = ""
    source: LabelSource | None = None    # where the frame and its masks live
    ignored: list[Seat] = field(default_factory=list)   # boxes left out of the loss

    @property
    def recording(self) -> str:
        """Recording (bag) name, from the file name."""
        m = _NAME.search(self.image)
        assert m is not None
        return m["rec"]

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


def load_labels(source: LabelSource = SOURCES[0]) -> list[FrameLabel]:
    """Read a run's ``labels.jsonl``.

    Parameters
    ----------
    source : LabelSource
        Frames folder and autolabel run folder.

    Returns
    -------
    list[FrameLabel]
        In file order, each with ``source`` set.
    """
    yolo: dict[str, list[dict]] = {}
    cache = source.labels_dir / "yolo_check.jsonl"
    if cache.exists():
        with open(cache) as fh:
            yolo = {r["image"]: r["seats"] for r in map(json.loads, fh)}
    out: list[FrameLabel] = []
    with open(source.labels_dir / "labels.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            ys = yolo.get(r["image"], [{}] * len(r["seats"]))
            seats = [Seat(s["label"], s["box"], s["point"], s.get("mask_area", 0), k,
                          y.get("yolo_conf", 0.0), y.get("yolo_label", ""))
                     for k, (s, y) in enumerate(zip(r["seats"], ys))]
            out.append(FrameLabel(r["image"], r["width"], r["height"], r["p_sit"], seats,
                                  r.get("error", ""), source))
    return out


def box_iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two ``[x1, y1, x2, y2]`` boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def combined_score(label: FrameLabel, seat: Seat) -> float:
    """How sure the two labelers together are that ``seat`` is a seat.

    Parameters
    ----------
    label : FrameLabel
        The seat's frame (for Qwen's ``p_sit``).
    seat : Seat

    Returns
    -------
    float
        In [0, 1].

    Notes
    -----
    ``score = (p_sit + yolo_conf) / 2``, with ``p_sit`` Qwen's frame-level gate
    probability (Qwen gives no per-box score) and ``yolo_conf`` the confidence of
    the best-overlapping COCO chair/couch/bench detection (IoU >= 0.3, else 0). For
    kinds COCO has no class for (:data:`QWEN_ONLY_LABELS`), ``score = p_sit``.
    """
    if seat.label in QWEN_ONLY_LABELS:
        return label.p_sit
    return (label.p_sit + seat.yolo_conf) / 2


def split_seats(label: FrameLabel) -> FrameLabel:
    """Move boxes that fail :func:`combined_score` or :data:`SEAT_LABELS` to ``ignored``.

    Returns a copy; ``seats`` then holds only the training targets.
    """
    ok = [s for s in label.seats
          if s.label in SEAT_LABELS and combined_score(label, s) >= COMBINED_MIN]
    bad = [s for s in label.seats if s not in ok]
    return replace(label, seats=ok, ignored=label.ignored + bad)


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
        Any of ``error``, ``borderline``, ``row``, ``cap``, ``duplicate``.

    Notes
    -----
    Called before :func:`split_seats`, on all boxes. ``borderline`` (``p_sit`` in
    :data:`GATE_BORDERLINE`) applies to empty frames only: frames with boxes are judged
    box by box with :func:`combined_score`.
    """
    s = label.seats
    why: list[str] = []
    if label.error:
        why.append("error")
    if not s and GATE_BORDERLINE[0] <= label.p_sit < GATE_BORDERLINE[1]:
        why.append("borderline")
    if label.image in row_flagged and label.key not in ROW_FLAG_KEEP:
        why.append("row")
    if len(s) >= SEAT_CAP:
        why.append("cap")
    if any(box_iou(s[i].box, s[j].box) > DUPLICATE_IOU for i in range(len(s)) for j in range(i)):
        why.append("duplicate")
    return why


def usable_labels(source: LabelSource = SOURCES[0],
                  camera: str | None = None) -> tuple[list[FrameLabel], Counter[str]]:
    """Frames of one run that pass :func:`exclusion_reasons`.

    Parameters
    ----------
    source : LabelSource
    camera : str, optional
        ``"up"`` or ``"down"`` to keep one camera; None keeps both.

    Returns
    -------
    kept : list[FrameLabel]
        Passing frames, after :func:`split_seats`.
    dropped : Counter[str]
        Number of frames dropped for each reason (a frame can count under several).
    """
    flags_csv = source.labels_dir / "row_flags.csv"
    row_flagged: set[str] = set()
    if flags_csv.exists():
        with open(flags_csv) as fh:
            row_flagged = {r["image"] for r in csv.DictReader(fh)}
    kept: list[FrameLabel] = []
    dropped: Counter[str] = Counter()
    for lab in load_labels(source):
        if camera is not None and lab.camera != camera:
            continue
        why = exclusion_reasons(lab, row_flagged)
        dropped.update(why)
        if not why:
            kept.append(split_seats(lab))
    return kept, dropped


def load_sources(sources: tuple[LabelSource, ...] = SOURCES, camera: str | None = "up"
                 ) -> tuple[list[FrameLabel], Counter[str]]:
    """:func:`usable_labels` over several runs, skipping runs not labelled yet.

    Returns
    -------
    kept : list[FrameLabel]
    dropped : Counter[str]
    """
    kept: list[FrameLabel] = []
    dropped: Counter[str] = Counter()
    for src in sources:
        if not (src.labels_dir / "labels.jsonl").exists():
            print(f"skipping {src.labels_dir.name}: no labels.jsonl yet")
            continue
        k, d = usable_labels(src, camera)
        kept += k
        dropped += d
    return kept, dropped


def split_by_recording(labels: list[FrameLabel], val_recordings: tuple[str, ...] = VAL_RECORDINGS
                       ) -> tuple[list[FrameLabel], list[FrameLabel]]:
    """Validation = every frame of ``val_recordings``; training = the rest.

    Notes
    -----
    Holding out whole recordings keeps neighbouring, nearly identical frames out of
    both sets, and (for the 11-22 bag) tests a building the training data never saw.
    """
    val = [lab for lab in labels if lab.recording in val_recordings]
    train = [lab for lab in labels if lab.recording not in val_recordings]
    return train, val


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
        Frames to use, e.g. from :func:`split_by_recording`. Each label's ``source``
        says where its frame and masks are.
    flip : bool
        Mirror each sample left-right with probability 0.5 (training augmentation).

    Notes
    -----
    ``__getitem__`` returns ``(image, target, valid)``:

    - image: (3, H', W') float32 RGB in [0, 1], padded with YOLO gray.
    - target: (H', W') float32 heatmap, 0 in the padding.
    - valid: (H', W') bool, True on real frame pixels outside every ignored box, so
      the loss skips padding and boxes that failed the label checks.

    with H', W' the :data:`FRAME_SIZE` rounded up to a multiple of 32 (384x640).
    Frames of another size are resized first (bilinear; masks nearest, before the
    distance transform, so every object still peaks at 1). Targets are built once at
    construction (about 1 MB per frame, image + target).
    """

    def __init__(self, labels: list[FrameLabel], flip: bool = False) -> None:
        self.labels = labels
        self.flip = flip
        self.images: list[np.ndarray] = []     # each (H, W, 3) uint8 RGB
        self.targets: list[np.ndarray] = []    # each (H, W) float16
        self.valid: list[np.ndarray] = []      # each (H, W) bool
        for lab in labels:
            if lab.source is None:
                raise ValueError(f"{lab.image}: no source; load labels with load_labels()")
            rgb: np.ndarray = np.array(Image.open(lab.source.frames_dir / lab.image).convert("RGB"))
            stem = Path(lab.image).stem
            masks = [np.array(Image.open(lab.source.labels_dir / "masks" / f"{stem}_{s.index}.png"))
                     > 127 for s in lab.seats]
            sx, sy = FRAME_SIZE[0] / rgb.shape[1], FRAME_SIZE[1] / rgb.shape[0]
            if rgb.shape[1::-1] != FRAME_SIZE:
                rgb = cv2.resize(rgb, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
                masks = [cv2.resize(m.astype(np.uint8), FRAME_SIZE,
                                    interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
            self.images.append(rgb)
            self.targets.append(seat_heatmap(masks, *rgb.shape[:2]).astype(np.float16))
            valid: np.ndarray = np.ones(rgb.shape[:2], bool)
            for s in lab.ignored:    # box in original px -> FRAME_SIZE px
                x1, y1, x2, y2 = s.box
                valid[int(y1 * sy):int(np.ceil(y2 * sy)), int(x1 * sx):int(np.ceil(x2 * sx))] = False
            self.valid.append(valid)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img: torch.Tensor = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255
        tgt: torch.Tensor = torch.from_numpy(self.targets[i]).float()
        valid: torch.Tensor = torch.from_numpy(self.valid[i])
        if self.flip and torch.rand(()) < 0.5:
            img, tgt, valid = img.flip(-1), tgt.flip(-1), valid.flip(-1)
        return (pad_to_stride(img, PAD_VALUE), pad_to_stride(tgt, 0.0),
                pad_to_stride(valid, False))


if __name__ == "__main__":
    kept, dropped = load_sources(camera="up")
    train, val = split_by_recording(kept)
    print(f"usable up-camera frames: {len(kept)} ({sum(bool(x.seats) for x in kept)} with "
          f"seats); dropped: {dict(dropped)}")
    print(f"seat boxes: {sum(len(x.seats) for x in kept)} kept "
          f"{dict(Counter(s.label for x in kept for s in x.seats))}; "
          f"{sum(len(x.ignored) for x in kept)} ignored "
          f"{dict(Counter(s.label for x in kept for s in x.ignored))}")
    for rec, n in sorted(Counter(x.recording for x in kept).items()):
        part = [x for x in kept if x.recording == rec]
        print(f"  {rec:48s} {n:5d} frames  {sum(len(x.seats) for x in part):5d} seats  "
              f"{sum(len(x.ignored) for x in part):4d} ignored  "
              f"{sum(not x.seats for x in part):4d} empty")
    for name, part in (("train", train), ("val", val)):
        print(f"{name}: {len(part)} frames, {sum(len(x.seats) for x in part)} seats, "
              f"{sum(not x.seats for x in part)} empty")
