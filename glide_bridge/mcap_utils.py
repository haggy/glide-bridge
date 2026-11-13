#!/usr/bin/env python3
"""Read Glide rosbag2 MCAP files into numpy / PIL images. No ROS install needed.

Requires:  pip install mcap mcap-ros2-support numpy pillow

Use as a library (for a training pipeline):

    from glide_mcap import iter_images, pair_by_stamp

    for topic, stamp_ns, img in iter_images("rosbag2_alpha_033_.../"):
        ...  # img is a numpy array: HxWx3 uint8 (color), HxW uint16 mm (depth), HxW uint8 (IR)

    pairs = pair_by_stamp(bag, "/camera_up/color/image_raw_rfps",
                          "/camera_up/depth/image_raw_rfps", max_dt_ms=20)

Or from the command line, to dump every image to PNG plus an index.csv per topic:

    python glide_mcap.py rosbag2_alpha_033_.../ --out extracted/
    python glide_mcap.py bag/ --out extracted/ --topics /camera_up/color/image_raw_rfps --every 2
    python glide_mcap.py bag/ --list        # topics, types and counts
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}

# encoding -> (numpy dtype, channels)
_ENCODINGS = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "8UC3": (np.uint8, 3),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def mcap_files(path: str | Path) -> list[Path]:
    """A bag folder (all its .mcap files, in order) or a single .mcap file."""
    p = Path(path)
    files = sorted(p.glob("*.mcap")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no .mcap files in {p}")
    return files


def image_to_numpy(msg, to_rgb: bool = True) -> np.ndarray:
    """Decode a sensor_msgs/Image or CompressedImage into a numpy array.

    Color comes back as HxWx3 uint8 RGB (BGR is converted unless to_rgb=False),
    depth as HxW uint16 in millimetres (0 = no reading), IR/labels as HxW uint8.
    """
    if hasattr(msg, "format"):  # CompressedImage
        img = Image.open(io.BytesIO(bytes(msg.data)))
        return np.asarray(img)

    enc = msg.encoding
    if enc not in _ENCODINGS:
        raise ValueError(f"unsupported image encoding {enc!r}")
    dtype, ch = _ENCODINGS[enc]
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    itemsize = dt.itemsize
    buf = np.frombuffer(bytes(msg.data), dtype=dt)
    # Rows may be padded: honour msg.step (bytes per row).
    row = buf.reshape(msg.height, msg.step // itemsize)[:, : msg.width * ch]
    arr = row.reshape(msg.height, msg.width, ch) if ch > 1 else row.reshape(msg.height, msg.width)
    arr = arr.astype(dt.newbyteorder("="), copy=True)  # native byte order, writable

    if to_rgb and enc in ("bgr8", "bgra8"):
        arr = arr[..., [2, 1, 0]] if ch == 3 else arr[..., [2, 1, 0, 3]]
    return arr


def stamp_ns(msg, log_time_ns: int) -> int:
    """Sensor timestamp from the message header; falls back to the bag's log time."""
    st = getattr(getattr(msg, "header", None), "stamp", None)
    if st is not None and (st.sec or st.nanosec):
        return st.sec * 1_000_000_000 + st.nanosec
    return log_time_ns


def list_topics(path: str | Path) -> dict[str, tuple[str, int]]:
    """topic -> (message type, count), from the MCAP summary (fast, no decoding)."""
    out: dict[str, tuple[str, int]] = {}
    for f in mcap_files(path):
        with open(f, "rb") as fh:
            summary = make_reader(fh).get_summary()
        if summary is None:
            continue
        counts = summary.statistics.channel_message_counts if summary.statistics else {}
        for cid, chan in summary.channels.items():
            schema = summary.schemas.get(chan.schema_id)
            name = schema.name if schema else "?"
            prev = out.get(chan.topic, (name, 0))
            out[chan.topic] = (name, prev[1] + counts.get(cid, 0))
    return out


def iter_images(path: str | Path, topics: Sequence[str] | None = None,
                to_rgb: bool = True) -> Iterator[tuple[str, int, np.ndarray]]:
    """Yield (topic, stamp_ns, numpy image) for every image message, in log order.

    topics=None reads every Image/CompressedImage topic in the bag.
    """
    if topics is None:
        topics = [t for t, (ty, _) in list_topics(path).items() if ty in IMAGE_TYPES]
    for f in mcap_files(path):
        with open(f, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for schema, channel, message, msg in reader.iter_decoded_messages(topics=list(topics)):
                if schema.name not in IMAGE_TYPES:
                    continue
                yield channel.topic, stamp_ns(msg, message.log_time), image_to_numpy(msg, to_rgb)


def pair_by_stamp(path: str | Path, topic_a: str, topic_b: str,
                  max_dt_ms: float = 20.0) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Match each topic_a frame to the nearest topic_b frame in time (e.g. color with depth).

    Returns (stamp_ns of a, image a, image b). Loads both streams into memory,
    so use it per bag, not across a whole dataset at once.
    """
    a, b = [], []
    for topic, ts, img in iter_images(path, [topic_a, topic_b]):
        (a if topic == topic_a else b).append((ts, img))
    b.sort(key=lambda x: x[0])
    b_ts = [t for t, _ in b]
    out, max_dt = [], int(max_dt_ms * 1e6)
    for ts, img in a:
        i = bisect.bisect_left(b_ts, ts)
        best = min((j for j in (i - 1, i) if 0 <= j < len(b)), key=lambda j: abs(b_ts[j] - ts), default=None)
        if best is not None and abs(b_ts[best] - ts) <= max_dt:
            out.append((ts, img, b[best][1]))
    return out


def save_image(arr: np.ndarray, path: Path) -> None:
    """PNG, lossless. uint16 depth stays 16-bit; float depth is saved as .npy."""
    if arr.dtype == np.float32:
        np.save(path.with_suffix(".npy"), arr)
    else:
        Image.fromarray(arr).save(path)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag", help="bag folder or single .mcap file")
    ap.add_argument("--out", default="extracted", help="output folder (default: extracted)")
    ap.add_argument("--topics", nargs="*", help="image topics to extract (default: all image topics)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame per topic")
    ap.add_argument("--list", action="store_true", help="only list topics, types and counts")
    args = ap.parse_args(argv)

    topics = list_topics(args.bag)
    if args.list:
        for t, (ty, n) in sorted(topics.items()):
            print(f"{n:8d}  {ty:35s}  {t}")
        return 0

    out = Path(args.out)
    seen: dict[str, int] = {}
    writers: dict[str, tuple[object, csv.writer]] = {}
    try:
        for topic, ts, img in iter_images(args.bag, args.topics):
            k = seen.get(topic, 0)
            seen[topic] = k + 1
            if k % args.every:
                continue
            d = out / topic.strip("/").replace("/", "__")
            if topic not in writers:
                d.mkdir(parents=True, exist_ok=True)
                fh = open(d / "index.csv", "w", newline="")
                w = csv.writer(fh)
                w.writerow(["stamp_ns", "file", "height", "width", "dtype", "channels"])
                writers[topic] = (fh, w)
            name = f"{ts}.png"
            save_image(img, d / name)
            ch = img.shape[2] if img.ndim == 3 else 1
            writers[topic][1].writerow([ts, name, img.shape[0], img.shape[1], img.dtype, ch])
    finally:
        for fh, _ in writers.values():
            fh.close()

    for t, n in sorted(seen.items()):
        kept = (n + args.every - 1) // args.every
        print(f"{t}: {kept} of {n} frames -> {out / t.strip('/').replace('/', '__')}")
    if not seen:
        print("no image messages found; run with --list to see the bag's topics", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())