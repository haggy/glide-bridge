#!/usr/bin/env python3
"""Record Glide camera streams and telemetry from a glide_link relay to disk.

Examples:
    python record.py --host 10.42.0.45                       # 4 RGB-D streams + telemetry
    python record.py --host 10.42.0.45 --streams up_color up_depth --duration 120
    python record.py --host 10.42.0.45 --lossless            # PNG color instead of JPEG
    python record.py --host localhost --preset sim           # Gazebo test sim

Session layout (data/sessions/<name>/):
    meta.json                     host, topics, start/end time, per-topic counts
    <stream>/NNNNNN.jpg|png       frames exactly as the relay sent them
    <stream>/index.csv            seq, ROS stamp, receive time, file, size, encodings
    <topic_slug>.jsonl            non-image messages (camera_info, odom, imu, gps, ...)

Depth frames are 16-bit PNGs in millimetres (0 = no reading). Color frames are
JPEG (quality 90) unless --lossless. Ctrl-C stops and finalizes meta.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from glide_bridge import DEFAULT_STREAMS, GLIDE_TELEMETRY, PRESETS, GlideLink
from glide_bridge.link import DEFAULT_PORT

EXT = {"jpeg": "jpg", "png": "png", "png16": "png"}


def slug(topic: str) -> str:
    """Topic name -> file-safe name, e.g. ``/camera_up/color`` -> ``camera_up__color``."""
    return topic.strip("/").replace("/", "__") or "root"


class SessionWriter:
    """Write relay messages to a session folder on a background thread.

    Parameters
    ----------
    root : Path
        Session folder.
    names : dict[str, str]
        Image topic -> stream folder name. Other image topics use :func:`slug`.

    Notes
    -----
    Messages are queued (max 2000) so slow disks never block the network reader;
    overflow is dropped and counted in ``dropped_local``.
    """

    def __init__(self, root: Path, names: dict[str, str]) -> None:
        self.root = root
        self.names = names                   # topic -> folder name for image streams
        self.q: queue.Queue[tuple[dict, bytes] | None] = queue.Queue(maxsize=2000)
        self.counts: dict[str, int] = {}
        self.bytes = 0
        self.dropped_local = 0
        self._csv: dict[str, tuple[IO[str], Any, Path]] = {}   # topic -> (file, csv.writer, dir)
        self._jsonl: dict[str, IO[str]] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def on_message(self, header: dict, payload: bytes) -> None:
        """GlideLink listener: queue the raw message without blocking."""
        try:
            self.q.put_nowait((header, payload))
        except queue.Full:                   # disk can't keep up; count and drop
            self.dropped_local += 1

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            header, payload = item
            try:
                self._write(header, payload)
            except Exception as exc:
                print(f"\nwrite error on {header.get('topic')}: {exc}", file=sys.stderr)

    def _write(self, header: dict, payload: bytes) -> None:
        """Images -> ``<stream>/NNNNNN.<ext>`` plus an index.csv row; else one JSONL line."""
        topic = header["topic"]
        n = self.counts.get(topic, 0)
        enc = header.get("enc")
        if enc in EXT or enc == "passthru":
            name = self.names.get(topic, slug(topic))
            if topic not in self._csv:
                d = self.root / name
                d.mkdir(parents=True, exist_ok=True)
                f = open(d / "index.csv", "w", newline="")
                w = csv.writer(f)
                w.writerow(["seq", "stamp", "recv", "file", "width", "height",
                            "src_encoding", "enc", "frame_id"])
                self._csv[topic] = (f, w, d)
            f, w, d = self._csv[topic]
            ext = EXT.get(enc) or ("png" if "png" in header.get("format", "") else "jpg")
            fname = f"{n:06d}.{ext}"
            (d / fname).write_bytes(payload)
            w.writerow([header.get("seq"), header.get("stamp"), header.get("recv"), fname,
                        header.get("width"), header.get("height"), header.get("src_encoding"),
                        enc, header.get("frame_id")])
            if n % 30 == 0:
                f.flush()
        else:
            f = self._jsonl.get(topic)
            if f is None:
                f = self._jsonl[topic] = open(self.root / f"{slug(topic)}.jsonl", "w")
            rec = {"seq": header.get("seq"), "stamp": header.get("stamp"),
                   "recv": header.get("recv"), "type": header.get("type"),
                   "msg": json.loads(payload)}
            f.write(json.dumps(rec) + "\n")
        self.counts[topic] = n + 1
        self.bytes += len(payload)

    def close(self) -> None:
        """Drain the queue (up to 30 s) and close all files."""
        self.q.put(None)
        self._thread.join(timeout=30)
        for f, _, _ in self._csv.values():
            f.close()
        for f in self._jsonl.values():
            f.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="10.42.0.45", help="relay host (robot IP or localhost)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--preset", choices=sorted(PRESETS), default="glide")
    p.add_argument("--streams", nargs="+", help="camera streams (default: the preset's RGB-D set; 'all' for every one)")
    p.add_argument("--no-info", action="store_true", help="don't record camera_info topics")
    p.add_argument("--no-telemetry", action="store_true", help="don't record odom/imu/gps/mode")
    p.add_argument("--extra", nargs="*", default=[], help="additional topics to record")
    p.add_argument("--rate", type=float, default=0.0, help="per-camera rate cap in Hz (default: every frame)")
    p.add_argument("--lossless", action="store_true", help="PNG instead of JPEG for color")
    p.add_argument("--out", default="data/sessions", help="parent directory for sessions")
    p.add_argument("--name", help="session name (default: timestamp)")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    names = (list(preset) if args.streams == ["all"] else
             args.streams or DEFAULT_STREAMS.get(args.preset, list(preset)))
    unknown = [n for n in names if n not in preset]
    if unknown:
        print(f"Unknown stream(s) {unknown}; choose from {sorted(preset)}", file=sys.stderr)
        return 2
    streams = [preset[n] for n in names]
    image_topics = {s.base_topic: s.name for s in streams}
    other = []
    if not args.no_info:
        other += [s.info_topic for s in streams if s.info_topic]
    if not args.no_telemetry and args.preset == "glide":
        other += GLIDE_TELEMETRY
    elif not args.no_telemetry:
        other += ["/odometry/filtered", "/wheel/odometry", "/imu/data", "/tf_static"]
    other += args.extra
    other = list(dict.fromkeys(t for t in other if t not in image_topics))

    root = Path(args.out) / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True, exist_ok=True)
    writer = SessionWriter(root, image_topics)
    meta = {"host": args.host, "port": args.port, "preset": args.preset,
            "streams": {s.name: {"topic": s.base_topic, "kind": s.kind, "info_topic": s.info_topic}
                        for s in streams},
            "other_topics": other, "lossless": args.lossless, "rate_cap": args.rate,
            "start": time.time(), "start_local": datetime.now().isoformat(timespec="seconds")}

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def connect() -> GlideLink | None:
        try:
            link = GlideLink(args.host, args.port)
        except OSError as exc:
            print(f"\rcannot reach relay at {args.host}:{args.port} ({exc}); retrying...   ",
                  end="", flush=True)
            return None
        link.add_listener(writer.on_message)
        for t in image_topics:
            link.subscribe(t, max_rate=args.rate, lossless=args.lossless, decode=False)
        for t in other:   # intrinsics don't change; 1 Hz is plenty
            link.subscribe(t, max_rate=1.0 if t.endswith("camera_info") else 0.0, decode=False)
        meta.setdefault("relay", {k: v for k, v in link.hello.items() if k != "topics"})
        meta.setdefault("topics_at_start", link.available_topics)
        return link

    print(f"Recording to {root}")
    print("  images: " + ", ".join(f"{n} <- {t}" for t, n in image_topics.items()))
    print(f"  other:  {', '.join(other) or '-'}")
    link = None
    reconnects = 0
    t0 = time.time()
    last: tuple[float, dict[str, int]] = (t0, {})
    try:
        while not stop.is_set():
            if args.duration and time.time() - t0 >= args.duration:
                break
            if link is None or not link.connected:
                if link is not None:
                    reconnects += 1
                    print(f"\nconnection lost ({link.disconnect_reason}); reconnecting...")
                link = connect()
                if link is None:
                    stop.wait(2.0)
                    continue
                missing = [t for t in list(image_topics) + other if t not in link.available_topics]
                if missing:
                    print(f"\n  not published right now: {', '.join(missing)}")
            stop.wait(2.0)
            now = time.time()
            counts = dict(writer.counts)
            dt = now - last[0]
            parts = []
            for t, n in image_topics.items():
                c = counts.get(t, 0)
                feed = link.feeds.get(t) if link else None
                drop = f"/{feed.dropped}d" if feed and feed.dropped else ""
                parts.append(f"{n} {c} ({(c - last[1].get(t, 0)) / dt:.1f}/s{drop})")
            n_other = sum(counts.get(t, 0) for t in other)
            print(f"\r[{now - t0:6.0f}s] " + "  ".join(parts) +
                  f"  other {n_other}  {writer.bytes / 1e6:.0f} MB   ", end="", flush=True)
            last = (now, counts)
    finally:
        if link is not None:
            link.close()
        writer.close()
        meta.update(end=time.time(), duration=time.time() - t0, reconnects=reconnects,
                    counts=writer.counts, bytes=writer.bytes,
                    dropped_on_disk_queue=writer.dropped_local,
                    dropped_on_link={t: f.dropped for t, f in (link.feeds.items() if link else [])})
        (root / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\nSaved {sum(writer.counts.values())} messages, {writer.bytes / 1e6:.1f} MB -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
