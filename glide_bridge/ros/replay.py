#!/usr/bin/env python3
"""Replay Glide camera streams from an MCAP recording as live ROS 2 topics.

Stands in for the robot's camera drivers: each stream comes out on the robot's own
topic (``/camera_up/color/image_raw``, ``/camera_down/color/image_raw``, ...) with the
same message type and the recorded timing, so downstream nodes can't tell a replay
from the live Glide. Streams are independent: a camera with fewer frames simply
publishes less often.

Examples
--------
::

    source env/activate_ros.sh
    python -m glide_bridge.ros.replay data/bags/bag.mcap --list     # what the bag holds
    python -m glide_bridge.ros.replay data/bags/bag.mcap            # up + down color
    python -m glide_bridge.ros.replay data/bags/bag.mcap --loop --rate 6
    python -m glide_bridge.ros.replay data/bags/bag.mcap --streams up_color up_depth

Published (per stream, names from ``glide_bridge.topics.PRESETS["glide"]``)
-------------------------------------------------------------------------
``<stream base_topic>`` : sensor_msgs/msg/Image
    e.g. ``/camera_up/color/image_raw``. Recorded ``..._rfps`` topics lose the suffix.
    Raw images pass through untouched (encoding as recorded); CompressedImage
    recordings are decoded to ``rgb8``.
``<stream info_topic>`` : sensor_msgs/msg/CameraInfo
    e.g. ``/camera_up/color/camera_info``: the latest recorded intrinsics, re-sent with
    every frame of that stream and the same stamp, if the bag has them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from ..mcap_utils import list_topics
from . import DEFAULT_STREAMS, GLIDE_STREAMS
from .image_msgs import compressed_to_image
from .mcap_source import RecordedStream, find_recorded, iter_bag


class McapCameraReplay(Node):
    """Publish recorded camera streams on the robot's topics with recorded timing.

    Parameters
    ----------
    bag : Path
        Bag folder or ``.mcap`` file.
    streams : list of RecordedStream
        Streams to replay, from :func:`~glide_bridge.ros.mcap_source.find_recorded`.
    rate : float
        Playback speed multiplier; 1.0 is real time.
    restamp : bool
        Overwrite ``header.stamp`` with the publish time, as a live driver would.
        Needed for consumers to measure latency; turn off to keep sensor stamps.

    Notes
    -----
    All streams share one clock. A message logged at bag time :math:`t_k` (ns, any
    stream) is published at wall time

    .. math:: w_k = w_0 + (t_k - t_0) / \\text{rate}

    where :math:`t_0` is the first logged image. Messages are read in log order, so
    the cameras interleave exactly as recorded. If decoding falls behind, messages go
    out immediately and the lag :math:`\\max(0, w_{now} - w_k)` is logged; nothing is
    skipped.
    """

    def __init__(self, bag: Path, streams: list[RecordedStream], rate: float = 1.0,
                 restamp: bool = True) -> None:
        super().__init__("mcap_camera_replay")
        self.bag, self.streams, self.rate, self.restamp = bag, streams, rate, restamp
        # recorded topic -> (stream, image publisher, info publisher or None)
        self.routes: dict[str, tuple[RecordedStream, Publisher, Publisher | None]] = {}
        self.info_of: dict[str, RecordedStream] = {}      # recorded info topic -> stream
        for rs in streams:
            image_pub = self.create_publisher(Image, rs.stream.base_topic,
                                              qos_profile_sensor_data)
            info_pub = None
            if rs.info_topic:
                info_pub = self.create_publisher(CameraInfo, rs.stream.info_topic,
                                                 qos_profile_sensor_data)
                self.info_of[rs.info_topic] = rs
            self.routes[rs.image_topic] = (rs, image_pub, info_pub)
            self.get_logger().info(
                f"{rs.stream.name}: {rs.image_topic} ({rs.count} frames) -> "
                f"{rs.stream.base_topic}"
                + (f" + {rs.stream.info_topic}" if info_pub else " (no CameraInfo)"))
        self.published: dict[str, int] = {rs.stream.name: 0 for rs in streams}
        self.max_lag_s = 0.0
        self.get_logger().info(f"{bag.name} at {rate}x real time")

    def play_once(self) -> int:
        """Publish every recorded frame of every stream once.

        Returns
        -------
        int
            Image messages published in this pass (stops early on ROS shutdown).
        """
        topics = list(self.routes) + list(self.info_of)
        latest_info: dict[str, CameraInfo] = {}           # stream name -> intrinsics
        t0_ns: int | None = None
        w0 = time.monotonic()
        last_log = w0
        n = 0
        for bm in iter_bag(self.bag, topics):
            if not rclpy.ok():
                break
            if bm.topic in self.info_of:
                latest_info[self.info_of[bm.topic].stream.name] = bm.msg
                continue
            rs, image_pub, info_pub = self.routes[bm.topic]
            if t0_ns is None:
                t0_ns = bm.log_time_ns
            delay = w0 + (bm.log_time_ns - t0_ns) * 1e-9 / self.rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self.max_lag_s = max(self.max_lag_s, -delay)

            img: Image = (compressed_to_image(bm.msg) if isinstance(bm.msg, CompressedImage)
                          else bm.msg)
            if self.restamp:
                img.header.stamp = self.get_clock().now().to_msg()
            info = latest_info.get(rs.stream.name)
            if info_pub is not None and info is not None:
                info.header.stamp = img.header.stamp
                info_pub.publish(info)
            image_pub.publish(img)
            self.published[rs.stream.name] += 1
            n += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                counts = ", ".join(f"{k} {v}" for k, v in self.published.items())
                self.get_logger().info(f"published {counts}; max lag "
                                       f"{self.max_lag_s * 1e3:.0f} ms")
                last_log = now
        return n


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Parse our options; everything else (``--ros-args ...``) is passed to rclpy."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                epilog="Run with --list first to see the bag's topics.")
    p.add_argument("bag", type=Path, help="bag folder or .mcap file")
    p.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS,
                   choices=sorted(GLIDE_STREAMS), metavar="STREAM",
                   help=f"streams to replay (default: {' '.join(DEFAULT_STREAMS)}); "
                        f"choices: {', '.join(sorted(GLIDE_STREAMS))}")
    p.add_argument("--rate", type=float, default=1.0, help="playback speed, 1 = real time")
    p.add_argument("--loop", action="store_true", help="replay forever")
    p.add_argument("--keep-stamps", action="store_true",
                   help="publish recorded sensor stamps instead of the publish time")
    p.add_argument("--list", action="store_true", help="list the bag's topics and exit")
    return p.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m glide_bridge.ros.replay BAG [options]``."""
    args, ros_args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.list:
        for t, (ty, n) in sorted(list_topics(args.bag).items()):
            print(f"{n:8d}  {ty:35s}  {t}")
        return 0
    streams: list[RecordedStream] = []
    for name in args.streams:
        rs = find_recorded(args.bag, GLIDE_STREAMS[name])
        if rs is None:
            print(f"warning: no recording of {name} ({GLIDE_STREAMS[name].base_topic}) "
                  f"in {args.bag.name}; skipping", file=sys.stderr)
        else:
            streams.append(rs)
    if not streams:
        print("nothing to replay; run with --list to see the bag's topics", file=sys.stderr)
        return 1

    rclpy.init(args=["replay", *ros_args])
    node = McapCameraReplay(args.bag, streams, args.rate, restamp=not args.keep_stamps)
    # Give subscribers a moment to discover the publishers so frame 0 isn't lost.
    time.sleep(0.5)
    try:
        passes = 0
        while rclpy.ok():
            n = node.play_once()
            passes += 1
            node.get_logger().info(f"pass {passes} done: {n} frames")
            if not args.loop or n == 0:
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        # A signal (Ctrl-C, kill) shuts the rclpy context down mid-publish; that's a
        # normal stop. Anything else is a real error.
        if rclpy.ok():
            raise
    finally:
        # print, not the ROS logger: the context may already be shut down here.
        counts = ", ".join(f"{k} {v}" for k, v in node.published.items())
        print(f"published {counts}; max lag {node.max_lag_s * 1e3:.0f} ms")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
