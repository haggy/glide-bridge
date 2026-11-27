"""Train the AffordGlide heatmap head on the auto-labelled frames.

YOLO stays frozen; only the head's layers are optimized. The loss is MSE between the
predicted and target heatmaps over real frame pixels (padding is ignored).

Each run writes to ``data/runs/<timestamp>/``:

- ``log.csv``: per epoch, training loss and the evaluation set's loss and metrics.
- ``best.pt`` / ``last.pt``: model ``state_dict`` at the lowest evaluation loss and
  at the end.
- ``preview_eNNN.png``: evaluation frames, target and prediction side by side.
- ``curves.png``: loss and metric curves.

Frames come from every labelled source in :data:`src.model.dataset.SOURCES`, one
camera (``--camera``, default up). The evaluation set is whole held-out recordings
(``--val-recordings``, default the 11-22 bag from another building), or with
``--train-only`` the training frames themselves (a pipeline check, not a measure of
generalization).
Training stops after ``--epochs``, or earlier once the evaluation loss stops
improving by more than ``--min-delta`` (relative) for ``--patience`` epochs.

Metrics (see :func:`evaluate`):

- peak hit rate: share of frames with seats whose hottest predicted pixel lies on a
  labelled object. This is what navigation relies on.
- warm IoU: overlap of ``pred > 0.5`` and ``target > 0.5``.
- empty peak: mean of the hottest predicted value on frames with no labelled object
  (false alarms; lower is better).

Most frames from the videos are empty, so each training epoch uses every frame with
seats plus a fresh random draw of ``--empty-ratio`` times as many empty frames.

Examples
--------
::

    uv run python -m src.model.train
    uv run python -m src.model.train --train-only --epochs 300   # stops at a plateau
    uv run python -m src.model.train --epochs 50 --lr 3e-4 --batch-size 8
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.model.affordance import AffordGlide
from src.model.dataset import (VAL_RECORDINGS, SeatHeatmapDataset, load_sources,
                               split_by_recording)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid pixels.

    Parameters
    ----------
    pred, target : torch.Tensor
        (B, H, W) float32 heatmaps.
    valid : torch.Tensor
        (B, H, W) bool, True where the pixel is part of the frame.

    Returns
    -------
    torch.Tensor
        Scalar ``sum(valid * (pred - target)^2) / sum(valid)``.
    """
    err: torch.Tensor = (pred - target).square() * valid
    return err.sum() / valid.sum().clamp(min=1)


def run_model(model: AffordGlide, x: torch.Tensor, quiet: bool) -> torch.Tensor:
    """Forward pass, optionally hiding what ``forward`` prints."""
    if not quiet:
        return model(x)
    with contextlib.redirect_stdout(io.StringIO()):
        return model(x)


def evaluate(model: AffordGlide, loader: DataLoader, device: torch.device,
             quiet: bool, warm: float = 0.5) -> dict[str, float]:
    """Loss and metrics over a loader.

    Parameters
    ----------
    model : AffordGlide
    loader : DataLoader
        Yields ``(image, target, valid)`` batches.
    device : torch.device
    quiet : bool
        Hide what ``forward`` prints.
    warm : float
        Heat level that counts as "warm" for the IoU.

    Returns
    -------
    dict[str, float]
        ``loss``: masked MSE, weighted by valid pixels.
        ``peak_hit``: over frames whose target has any object, the share where
        ``target[argmax(pred)] > 0`` (argmax over valid pixels only).
        ``warm_iou``: ``|P & T| / |P | T|`` with ``P = pred > warm``,
        ``T = target > warm``, summed over all frames.
        ``empty_peak``: mean over frames without objects of ``max(pred)`` (valid
        pixels), or NaN if there are none.
    """
    model.eval()
    total, count = 0.0, 0.0
    hits, frames = 0, 0
    inter, union = 0.0, 0.0
    empty_peaks: list[float] = []
    with torch.no_grad():
        for img, tgt, valid in loader:
            img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
            pred = run_model(model, img, quiet)
            n = float(valid.sum())
            total += float(masked_mse(pred, tgt, valid)) * n
            count += n
            flat_pred = pred.masked_fill(~valid, float("-inf")).flatten(1)   # (B, H*W)
            at_peak = tgt.flatten(1).gather(1, flat_pred.argmax(1, keepdim=True))[:, 0]
            has_obj = tgt.flatten(1).amax(1) > 0                              # (B,)
            hits += int((at_peak[has_obj] > 0).sum())
            empty_peaks += flat_pred.amax(1)[~has_obj].tolist()
            frames += int(has_obj.sum())
            p, t = (pred > warm) & valid, (tgt > warm) & valid
            inter += float((p & t).sum())
            union += float((p | t).sum())
    return {"loss": total / max(count, 1.0), "peak_hit": hits / max(frames, 1),
            "warm_iou": inter / max(union, 1.0),
            "empty_peak": float(np.mean(empty_peaks)) if empty_peaks else float("nan")}


def save_preview(model: AffordGlide, ds: SeatHeatmapDataset, idx: Sequence[int],
                 device: torch.device, path: Path, title: str, quiet: bool) -> None:
    """Save rows of frame | target | prediction for a few validation samples.

    Target and prediction share one color scale, 0 to 1, so they compare directly;
    values outside it are clipped in the plot only.
    """
    import matplotlib.pyplot as plt

    model.eval()
    fig, axs = plt.subplots(len(idx), 3, figsize=(15, 3.2 * len(idx)), layout="constrained",
                            squeeze=False)
    for row, i in zip(axs, idx):
        img, tgt, valid = ds[i]
        with torch.no_grad():
            pred = run_model(model, img[None].to(device), quiet)[0].cpu()
        rows = valid.any(1).nonzero().squeeze(1)      # crop the padding
        cols = valid.any(0).nonzero().squeeze(1)
        crop = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))
        p: np.ndarray = pred.numpy()[crop]
        row[0].imshow(img.permute(1, 2, 0).numpy()[crop])
        row[0].set_title(f"{ds.labels[i].recording.removeprefix('rosbag2_alpha_033_')} "
                         f"{ds.labels[i].key}")
        row[1].imshow(tgt.numpy()[crop], cmap="inferno", vmin=0, vmax=1)
        row[1].set_title("target")
        im = row[2].imshow(p, cmap="inferno", vmin=0, vmax=1)
        row[2].set_title(f"prediction  [{p.min():.2f}, {p.max():.2f}]")
        fig.colorbar(im, ax=row[2], fraction=0.03)
        for ax in row:
            ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def plot_curves(log: list[dict[str, float]], eval_name: str, path: Path) -> None:
    """Plot loss (left) and metrics (right) per epoch."""
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    ep = [r["epoch"] for r in log]
    ax.plot(ep, [r["train_loss"] for r in log], label="training batches (augmented)")
    ax.plot(ep, [r["eval_loss"] for r in log], label=f"{eval_name}, eval mode")
    ax.axhline(log[0]["eval_zero_loss"], color="gray", ls="--", lw=1,
               label=f"{eval_name}, predict all zeros")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.set_yscale("log")
    ax.set_title("loss")
    ax.legend()
    bx.plot(ep, [r["peak_hit"] for r in log], label="peak hit rate")
    bx.plot(ep, [r["warm_iou"] for r in log], label="warm IoU (> 0.5)")
    bx.plot(ep, [r["empty_peak"] for r in log], label="peak heat on empty frames (lower is better)")
    bx.set_ylim(0, 1)
    bx.set_xlabel("epoch")
    bx.set_title(f"accuracy on {eval_name}")
    bx.legend()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Train, validate each epoch, and save checkpoints, log and previews."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=30, help="maximum number of epochs")
    p.add_argument("--patience", type=int, default=8,
                   help="stop after this many epochs without enough improvement (0: never)")
    p.add_argument("--min-delta", type=float, default=0.01,
                   help="relative drop in evaluation loss that counts as improvement")
    p.add_argument("--train-only", action="store_true",
                   help="train on every usable frame and evaluate on the same frames")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--empty-ratio", type=float, default=1.0,
                   help="empty training frames per frame with seats, redrawn each epoch "
                        "(negative: use every frame)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--camera", choices=["up", "down", "both"], default="up")
    p.add_argument("--val-recordings", nargs="+", default=list(VAL_RECORDINGS),
                   help="recordings (bag names) held out for validation")
    p.add_argument("--out", type=Path, default=None,
                   help="run folder (default: data/runs/<timestamp>)")
    p.add_argument("--preview-every", type=int, default=5, help="epochs between previews")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: cuda if available); cpu works while the GPU is busy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="show what forward() prints")
    args = p.parse_args(argv)
    matplotlib.use("Agg")
    quiet = not args.verbose

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out: Path = args.out or REPO_ROOT / "data/runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)

    kept, dropped = load_sources(camera=None if args.camera == "both" else args.camera)
    if args.train_only:
        train_labels, val_labels = kept, kept
        eval_name = "training frames"
    else:
        train_labels, val_labels = split_by_recording(kept, tuple(args.val_recordings))
        eval_name = "validation"
        if not val_labels:
            print(f"no frames from {args.val_recordings}; label them or pass --train-only")
            return 1
    print(f"{len(kept)} usable frames (dropped: {dict(dropped)}); "
          f"train {len(train_labels)}, evaluate on {eval_name}: {len(val_labels)}")
    t0 = time.time()
    train_ds = SeatHeatmapDataset(train_labels, flip=True)
    val_ds = SeatHeatmapDataset(val_labels)   # no flips
    print(f"loaded images and built targets in {time.time() - t0:.1f} s")
    val_dl = DataLoader(val_ds, args.batch_size)
    pos_idx = [i for i, lab in enumerate(train_labels) if lab.seats]
    neg_idx = [i for i, lab in enumerate(train_labels) if not lab.seats]
    n_neg = len(neg_idx) if args.empty_ratio < 0 else min(len(neg_idx),
                                                          round(args.empty_ratio * len(pos_idx)))
    print(f"training frames: {len(pos_idx)} with seats, {len(neg_idx)} empty "
          f"({n_neg} drawn per epoch)")

    model = AffordGlide().to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    print(f"device {device}; trainable parameters: {sum(q.numel() for q in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Reference: the loss of a model that always predicts 0 (most pixels are 0).
    val_zero = sum(float(masked_mse(torch.zeros_like(t), t, v)) * float(v.sum())
                   for _, t, v in val_dl) / sum(float(v.sum()) for _, _, v in val_dl)
    with_seats = [i for i, lab in enumerate(val_labels) if lab.seats]
    preview_idx = with_seats[:: max(1, len(with_seats) // 4)][:4]

    log: list[dict[str, float]] = []
    best = float("inf")
    plateau_ref, stale = float("inf"), 0     # loss to beat by min_delta, epochs without
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        total, n_batches = 0.0, 0
        pick = torch.randperm(len(neg_idx))[:n_neg].tolist()
        epoch_idx = pos_idx + [neg_idx[j] for j in pick]
        train_dl = DataLoader(Subset(train_ds, epoch_idx), args.batch_size, shuffle=True,
                              drop_last=True)
        for img, tgt, valid in train_dl:
            img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
            loss = masked_mse(run_model(model, img, quiet), tgt, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        train_loss = total / max(n_batches, 1)
        ev = evaluate(model, val_dl, device, quiet)
        log.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": ev["loss"],
                    "eval_zero_loss": val_zero, "peak_hit": ev["peak_hit"],
                    "warm_iou": ev["warm_iou"], "empty_peak": ev["empty_peak"],
                    "seconds": time.time() - t_ep})
        mark = ""
        if ev["loss"] < best:
            best = ev["loss"]
            torch.save(model.state_dict(), out / "best.pt")
            mark = "  (best)"
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  eval {ev['loss']:.4f} "
              f"(all zeros {val_zero:.4f})  peak hit {ev['peak_hit']:.2f}  "
              f"warm IoU {ev['warm_iou']:.2f}  empty peak {ev['empty_peak']:.2f}  "
              f"{time.time() - t_ep:.0f} s{mark}")

        if ev["loss"] < plateau_ref * (1 - args.min_delta):
            plateau_ref, stale = ev["loss"], 0
        else:
            stale += 1
        stop = args.patience > 0 and stale >= args.patience
        if epoch == 1 or epoch % args.preview_every == 0 or epoch == args.epochs or stop:
            save_preview(model, val_ds, preview_idx, device, out / f"preview_e{epoch:03d}.png",
                         f"epoch {epoch}: {eval_name} loss {ev['loss']:.4f}", quiet)
        if stop:
            print(f"stopping: loss improved by less than {args.min_delta:.0%} "
                  f"in the last {args.patience} epochs")
            break

    torch.save(model.state_dict(), out / "last.pt")
    with open(out / "log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    plot_curves(log, eval_name, out / "curves.png")
    print(f"best {eval_name} loss {best:.4f} after {len(log)} epochs; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
