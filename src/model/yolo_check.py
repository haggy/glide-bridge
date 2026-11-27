"""Cross-check the Qwen seat boxes with COCO YOLO26n and cache the scores.

For every labelled seat box, find the YOLO detection of a seat class (chair, couch,
bench) that overlaps it most. Its confidence is the box's YOLO score (0 if none
overlaps enough). :mod:`src.model.dataset` averages this with Qwen's ``p_sit`` and
thresholds the result.

Writes ``<labels_dir>/yolo_check.jsonl`` next to each run's ``labels.jsonl``: one line
per frame with seats, ``{"image", "seats": [{"yolo_conf", "yolo_label", "iou"}, ...]}``
in the same seat order.

Examples
--------
::

    uv run python -m src.model.yolo_check            # all runs in dataset.SOURCES
    uv run python -m src.model.yolo_check --plot     # + score histogram per label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.model.dataset import SOURCES, LabelSource, box_iou, load_labels

YOLO_WEIGHTS: str = "yolo26n.pt"
YOLO_MIN_CONF: float = 0.05     # keep weak detections: they still add to the score
MATCH_IOU: float = 0.3          # a YOLO box must overlap the Qwen box this much
SEAT_CLASSES: dict[int, str] = {13: "bench", 56: "chair", 57: "couch"}    # COCO ids
CACHE_NAME: str = "yolo_check.jsonl"


def check_source(model: YOLO, source: LabelSource) -> int:
    """Score every seat box of one label run; returns the number of boxes scored.

    Parameters
    ----------
    model : YOLO
        COCO YOLO26n.
    source : LabelSource

    Notes
    -----
    Matching ignores which seat class each side used: a Qwen "bench" that YOLO calls a
    couch counts. The label check is "is there a seat here", not "is it this kind".
    """
    n = 0
    with open(source.labels_dir / CACHE_NAME, "w") as fh:
        for lab in load_labels(source):
            if not lab.seats:
                continue
            r = model.predict(str(source.frames_dir / lab.image), conf=YOLO_MIN_CONF,
                              classes=list(SEAT_CLASSES), verbose=False)[0]
            boxes: np.ndarray = r.boxes.xyxy.cpu().numpy()    # (N, 4) px, frame size
            cls: np.ndarray = r.boxes.cls.cpu().numpy().astype(int)
            conf: np.ndarray = r.boxes.conf.cpu().numpy()
            out = []
            for s in lab.seats:
                best = {"yolo_conf": 0.0, "yolo_label": "", "iou": 0.0}
                for b, c, f in zip(boxes, cls, conf):
                    iou = box_iou(s.box, b.tolist())
                    if iou >= MATCH_IOU and f > best["yolo_conf"]:
                        best = {"yolo_conf": round(float(f), 4), "yolo_label": SEAT_CLASSES[c],
                                "iou": round(iou, 3)}
                out.append(best)
                n += 1
            fh.write(json.dumps({"image": lab.image, "seats": out}) + "\n")
    return n


def plot_scores(path: Path) -> None:
    """Histogram of the combined score per Qwen label, all runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.model.dataset import COMBINED_MIN, combined_score

    by_label: dict[str, list[float]] = {}
    for src in SOURCES:
        for lab in load_labels(src):
            for s in lab.seats:
                by_label.setdefault(s.label, []).append(combined_score(lab, s))
    labels = sorted(by_label, key=lambda k: -len(by_label[k]))[:6]
    fig, axs = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 2.8), sharex=True)
    for ax, k in zip(np.atleast_1d(axs), labels):
        v = np.array(by_label[k])
        ax.hist(v, bins=np.linspace(0, 1, 21), color="tab:blue")
        ax.axvline(COMBINED_MIN, color="tab:red", lw=1)
        ax.set_title(f"{k}: {len(v)} boxes, {(v >= COMBINED_MIN).mean():.0%} kept", fontsize=9)
        ax.set_xlabel("combined score")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plot", action="store_true", help="save data/plots/combined_scores.png")
    p.add_argument("--plot-only", action="store_true", help="skip YOLO; plot the cache")
    args = p.parse_args()
    if not args.plot_only:
        yolo = YOLO(YOLO_WEIGHTS)
        for src in SOURCES:
            if (src.labels_dir / "labels.jsonl").exists():
                print(f"{src.labels_dir.name}: {check_source(yolo, src)} boxes scored")
    if args.plot or args.plot_only:
        plot_scores(Path(__file__).resolve().parents[2] / "data/plots/combined_scores.png")
