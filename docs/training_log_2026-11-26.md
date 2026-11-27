# Training log, 2026-11-26 (unattended session)

Request: label the new 11-22 bag and the up-camera videos, then train on up-camera data
with a realistic train/validation split, tuning the learning rate. Nothing here is
committed; no files were deleted except the failed, empty 11-22 label run (see 2.).

Entries are in order; times are local.

## 1. New data pulled

| Source | What | Where |
|---|---|---|
| Drive folder `1vrBIbnQ…` | 2 bags: 11-22 04-48-34 (122 MB, 7 s desk test, no seats), 11-22 04-59-27 (1.06 GB, 101 s, new building) | `data/bags/` |
| Drive folder `1zNq4Q4z…` | 7 `Glide-rosbag*-up.mp4` (220 MB) | `data/videos/drive_up/` |

- `fetch_bags.sh pull` didn't search Drive subfolders; fixed (`lsf -R`). You have
  since committed that (6b71d4c).
- 11-22 04-59-27: up camera 567 frames, no gaps; down color only 88 frames in three
  bursts (depth/IR complete). Contact sheets: `data/plots/bags/`.
- Videos hold exactly one frame per bag frame (the 08-29-39 and 04-59-27 videos have
  229 and 567 frames, matching their bags), at the robot's ~5.6 fps.

| Video | Size | Frames | Used? |
|---|---|---|---|
| 2026-07-10 10-37-29 | 424x240 | 243 | yes, all frames |
| 07-14 1 fps 10 s test | 640x360 | 10 | yes |
| 11-21 08-27-30 | 640x360 | 293 | yes, all frames |
| 11-21 08-29-39 | 640x360 | 229 | no: same frames as the MCAP bag already labelled |
| 11-22 04-48-34 | 640x360 | 45 | no: desk test |
| 11-22 04-59-27 | 640x360 | 567 | no: labelled from the MCAP bag instead |
| 11-23 01-19-36 | 424x240 | 6305 (18.7 min) | yes, every 3rd frame (~2 fps): 2102 |

Frame extraction: new `model-tests/qwen-autolabel/extract_video_frames.py` (names like
the bag frames, `t = frame_index / fps`, approximate). 2648 video frames in
`model-tests/qwen-autolabel/data/frames_videos/`, 567 bag frames in `data/frames_1122_up/`.

## 2. Qwen server memory

`serve.sh` no longer started: the desktop now holds ~1 GB of GPU memory (was ~0.77 GB),
and at `--gpu-memory-utilization 0.92` no KV cache was left. Tried: fewer sequences,
`--enforce-eager`, smaller image cap; none fit. What works: a fixed KV cache.
SAM inside `autolabel.py` also needs GPU memory, so the first label run died on frame 1
(out of memory; its empty output folder was removed). Final settings, now the defaults
in `serve.sh`: `--gpu-memory-utilization 0.80 --kv-cache-memory=600000000` (one
4096-token request needs 0.3 GB). Server + SAM use ~11.5 of 12 GB, so the GPU can't
train while labeling runs.

## 3. Code changes (glide-bridge, uncommitted)

- `src/model/dataset.py`: several label runs (`SOURCES`: 11-21 bag, 11-22 bag,
  videos); `--camera`-style filter (training defaults to up); `split_by_recording`
  holds out whole recordings (default validation: 11-22 04-59-27, another building);
  frames of any size are resized to 640x360 (masks nearest, before the distance
  transform).
- `src/model/train.py`: `--camera`, `--val-recordings`, `--device` (cpu works while the
  GPU is busy).

## 4. Labeling

### 4a. 11-22 04-59-27 bag, up camera

- 567 frames, 451 passed the gate, 1530 masks: chair 1400, bench 93, steps 27, sofa 10.
  16 frames hit the 12-object cap; 120 have a borderline gate score (0.2-0.6).
- `check_rows.py bag_1122_up data/frames_1122_up`: 14 frames flagged as regular rows
  (not reviewed by eye; excluded like the unreviewed 11-21 flags).
- **"steps" = staircases** (all 8 examples checked:
  `model-tests/qwen-autolabel/outputs/autolabel/bag_1122_up/check_steps.png`).
  "bench" = low dark objects along the window wall, too small to judge
  (`check_bench.png`). Both now excluded: `dataset.py` keeps a frame only if every
  label is in `SEAT_LABELS` = chair, sofa, couch, armchair, stool (was: drop "bench").
- Usable after all filters: 381 frames (300 with seats, 1045 seats, 81 empty).
  This is the validation set: a building the training data never saw.

### 4b. Up-camera videos

- 2648 frames, 0 errors. Labels: bench 437, chair 136, steps 11, wheelchair 7, sofa 6,
  stool 4, ledge 1. `check_rows.py videos_up data/frames_videos`: 2 frames flagged.
- Checked by eye (`model-tests/qwen-autolabel/outputs/autolabel/videos_up/check_bench.png`,
  `check_chair.png`):
  - "bench" is mostly wrong: counters, a reception desk, a wall panel, a trash can. A
    few could be a built-in wooden window bench (11-21 08-27-30). Excluded (allowlist).
  - "chair" in 07-10 (an office/lab) is mostly real office chairs.
  - "chair" in 11-23 is often wrong: trash can, glass door, notice board, distant
    specks. That video is 424x240 and mostly corridors. 31 seats survive the filters;
    some are probably false. **Worth a manual pass or dropping 11-23 positives.**

| Recording | Usable frames | Seats | Empty |
|---|---|---|---|
| 2026-07-10 10-37-29 (video) | 203 | 58 | 168 |
| 07-14 test (video) | 10 | 0 | 10 |
| 11-21 08-27-30 (video) | 106 | 0 | 106 |
| 11-21 08-29-39 (bag) | 199 | 919 | 7 |
| 11-22 04-59-27 (bag) | 381 | 1045 | 81 |
| 11-23 01-19-36 (video) | 1624 | 31 | 1605 |

Every empty frame had a gate score below 0.2, so they are reasonable negatives.

Qwen server stopped after labeling (GPU free for training).

## 5. Training

Split: validation = 11-22 04-59-27 (381 frames, other building); training = the rest
(2142 frames: 246 with seats, 1896 empty). 89% of training frames are empty, so
`train.py` now draws, each epoch, every frame with seats plus an equal number of
empty frames (`--empty-ratio 1.0`, redrawn each epoch). New metric "empty peak": mean
of the hottest predicted pixel on empty validation frames (false alarms).

Model: your `AffordGlide` as of this morning (frozen YOLO26n, three 1x1 projections,
one 3x3 conv, nearest x8 upsample; 122k trainable parameters). Not changed.

Learning-rate sweep, AdamW, batch 8, weight decay 1e-4, stop when validation loss
improves < 1% in 10 epochs (max 150). Runs in `data/runs/2026-11-26_lr_*`.

### 5a. Results: validation on 11-22 (held-out building)

| lr | Epochs (stopped) | Best val loss | Peak hit | Warm IoU | Empty peak |
|---|---|---|---|---|---|
| 3e-5 | 53 | 0.0047 (ep 47) | 0.84 | 0.01 | 0.14 |
| 1e-4 | 39 | 0.0046 (ep 29) | 0.85 | 0.01 | 0.13 |
| 3e-4 | 31 | 0.0047 (ep 21) | 0.85 | 0.01 | 0.13 |
| 1e-3 | 29 | 0.0047 (ep 28) | 0.87 | 0.01 | 0.15 |

Predicting all zeros: val loss 0.0074. Each epoch takes ~5 s on the GPU.

- **Learning rate isn't the limit.** All four end at the same loss; higher rates only
  get there sooner. 3e-4 is a good default (fast, stable after epoch 1; 1e-3 has a
  0.22 first-batch loss spike).
- **Peak hit 0.85 on a building it never saw.** Baselines on the same 300 frames: the
  peak of the average training heatmap hits 0.18; the best possible single fixed pixel
  (chosen with the answers) hits 0.32. So the model is locating seats, not a prior.
- **Reverse split** (`data/runs/2026-11-26_fold2_val1121_lr_3e-4`: validate on 11-21
  08-29-39, train on everything else incl. 11-22): best val loss 0.0091 (zeros 0.0152),
  peak hit 0.86-0.88, empty peak 0.27. Same picture with the buildings swapped.
- **Heat stays weak.** Predictions reach 0.13-0.43 where the target is 1
  (`data/runs/2026-11-26_lr_3e-4/preview_e031.png`), so warm IoU ~0.01-0.02. The peak
  is in the right place but the map is flat and blurry. Training loss ≈ validation
  loss, so this is underfitting: the head is one linear 3x3 conv on frozen features.

### 5b. Follow-up (2026-11-27): steps kept

Per your review: steps/stairs can be sittable in extreme cases, so `SEAT_LABELS` now
includes "steps" and "stairs" ("bench" stays out because those labels were wrong, not
because benches aren't seats). Effect: validation grows 381 -> 401 frames (1045 -> 1090
seats). Training is unchanged: all 11 video "steps" frames (11-23, t292-353) are already
dropped as borderline (gate 0.32-0.5). The 5a numbers used the 381-frame validation set.

### 5c. Follow-up (2026-11-27): Qwen + YOLO combined score

"bench" turned out to be real in 11-23 t1062-1090 (wooden hallway benches;
`data/plots/bench_yolo_agree.png`). Instead of hand-checking, each seat box now gets
`score = (p_sit + yolo_conf) / 2`:

- `yolo_conf`: best-overlapping COCO YOLO26n chair/couch/bench detection (IoU >= 0.3,
  conf >= 0.05; else 0). Cached per label run in `yolo_check.jsonl` by
  `uv run python -m src.model.yolo_check --plot`.
- stool/steps/stairs have no COCO class: score = `p_sit`.
- Boxes with score < `COMBINED_MIN` = 0.5, or a label outside `SEAT_LABELS` (now with
  bench), are **ignored** (cut out of the loss via `valid`), not used as targets, and
  the frame stays in. The borderline-gate rule now applies only to empty frames.
- Score histograms: `data/plots/combined_scores.png`. Bench stretch, kept vs ignored:
  `data/plots/bench_1123_combined.png` (ignored ones are mostly blurred or distant).

New split: train 2540 frames (910 seats), val 508 frames (674 seats). Kept boxes:
chair 1488, bench 54, steps 22, sofa 10, stool 10; ignored 1225 (chair 737, bench 452).
Note 1036 of 3830 chairs get no YOLO match at all (median p_sit 0.9), so at 0.5 most of
them are ignored. 5a numbers are not comparable to runs on this split.

Retrained on this split (`data/runs/2026-11-27_combined_lr_3e-4`, lr 3e-4, patience
10): stopped at epoch 27, best val loss 0.0037 at epoch 17 (all zeros 0.0052), peak hit
0.86-0.88, warm IoU 0.03-0.06 (was ~0.01), empty peak 0.15-0.21. Preview of best.pt on
20 validation frames: `demos/best_model_preview.py RUN` ->
`data/runs/2026-11-27_combined_lr_3e-4/preview_best_20.png`.

### 6. Suggested next steps (not done)

1. Give the head capacity: 3x3 conv -> SiLU -> 3x3 conv (-> SiLU -> 1x1), bilinear
   upsample. Rerun `uv run python -m src.model.train --lr 3e-4 --epochs 150 --patience 10`
   and compare to the table above (and the fold-2 run with
   `--val-recordings rosbag2_alpha_033_2026-11-21_08-29-39`).
2. Clean the 11-23 positives (31 seats, some false: trash can, door, notice board) or
   drop them; label more seat-rich up-camera data (only 246 training frames have seats).
3. Weighted loss (more weight on object pixels) if peaks stay low after 1.

### Files from this session

- glide-bridge (uncommitted): `src/model/dataset.py`, `src/model/train.py`, this log.
- model-tests (uncommitted): `qwen-autolabel/extract_video_frames.py` (new),
  `qwen-autolabel/serve.sh` (memory flags).
- Data (git-ignored): `data/bags/rosbag2_alpha_033_2026-11-22_*`,
  `data/videos/drive_up/`, `data/runs/2026-11-26_*`, `data/plots/bags/`;
  model-tests `data/frames_1122_up`, `data/frames_videos`,
  `outputs/autolabel/{bag_1122_up,videos_up}`.
- Disk: 289 GB free at the end (this session added ~1.5 GB).
- Qwen server: stopped.
