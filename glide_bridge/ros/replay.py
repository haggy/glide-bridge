#!/usr/bin/env python3
"""Replay a Glide MCAP recording as a live ROS 2 camera stream on ``/camera_raw``.

Stands in for the robot's camera driver: frames come out on the same topic, with the
same message type and the original timing, so downstream nodes can't tell a replay
from a live stream. To switch to the real Glide later, remap its camera topic to
``/camera_raw`` instead of running this node.

Examples
--------
::

    source env/activate_ros.sh
    python -m glide_bridge.ros.replay data/bags/rosbag2_alpha_033_....mcap --list
    python -m glide_bridge.ros.replay data/bags/bag.mcap                 # real time, once
    python -m glide_bridge.ros.replay data/bags/bag.mcap --loop --rate 2 # 2x, forever
    python -m glide_bridge.ros.replay data/bags/bag.mcap --fps 15        # fixed rate
    python -m glide_bridge.ros.replay bag.mcap --topic /camera_down/color/image_raw_rfps
    # standard ROS remapping also works:
    python -m glide_bridge.ros.replay bag.mcap --ros-args -r /camera_raw:=/camera_up_raw

Published
---------
``/camera_raw`` : sensor_msgs/msg/Image
    Raw images pass through untouched (encoding as recorded). CompressedImage topics
    are decoded to ``rgb8``.
``/camera_raw/camera_info`` : sensor_msgs/msg/CameraInfo
    The latest recorded intrinsics, re-sent with every frame and the same stamp, if
    the bag has a CameraInfo topic for that camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from ..mcap_utils import list_topics
from . import CAMERA_INFO_TOPIC, CAMERA_TOPIC
from .image_msgs import compressed_to_image, stamp_to_sec
from .mcap_source import iter_bag, info_topic_for, pick_camera_topic


class McapCameraReplay(Node):
    """Publish one camera topic from a bag on ``/camera_raw`` with recorded timing.

    Parameters
    ----------
    bag : Path
        Bag folder or ``.mcap`` file.
    image_topic : str
        Recorded Image or CompressedImage topic to replay.
    info_topic : str or None
        Recorded CameraInfo topic for the same camera, or None to skip intrinsics.
    rate : float
        Playback speed multiplier; 1.0 is real time. Ignored when ``fps`` is set.
    fps : float or None
        Publish at a fixed rate instead of the recorded timing.
    restamp : bool
        Overwrite ``header.stamp`` with the publish time, as a live driver would.
        Needed for consumers to measure latency; turn off to keep sensor stamps.
    out_topic : str
        Output image topic. Intrinsics go to ``out_topic + "/camera_info"``.

    Notes
    -----
    Frame ``k`` (recorded at sensor time :math:`t_k`) is published at wall time

    .. math:: w_k = w_0 + (t_k - t_0) / \\text{rate}

    or :math:`w_k = w_0 + k / \\text{fps}` with ``--fps``. If decoding falls behind,
    frames go out immediately and the lag :math:`\\max(0, w_{now} - w_k)` is logged;
    no frames are skipped.
    """

    def __init__(self, bag: Path, image_topic: str, info_topic: str | None,
                 rate: float = 1.0, fps: float | None = None, restamp: bool = True,
                 out_topic: str = CAMERA_TOPIC) -> None:
        super().__init__("mcap_camera_replay")
        self.bag, self.image_topic, self.info_topic = bag, image_topic, info_topic
        self.rate, self.fps, self.restamp = rate, fps, restamp
        self.image_pub = self.create_publisher(Image, out_topic, qos_profile_sensor_data)
        info_out = (CAMERA_INFO_TOPIC if out_topic == CAMERA_TOPIC
                    else out_topic.rstrip("/") + "/camera_info")
        self.info_pub = self.create_publisher(CameraInfo, info_out, qos_profile_sensor_data)
        self.published = 0
        self.max_lag_s = 0.0
        self.get_logger().info(
            f"{bag.name}: {image_topic} -> {self.image_pub.topic_name}"
            + (f", {info_topic} -> {self.info_pub.topic_name}" if info_topic else
               " (no CameraInfo in bag)")
            + (f" at {fps} fps" if fps else f" at {rate}x real time"))

    def play_once(self) -> int:
        """Publish every frame in the bag once.

        Returns
        -------
        int
            Frames published in this pass (stops early if ROS shuts down).
        """
        topics = [self.image_topic] + ([self.info_topic] if self.info_topic else [])
        info: CameraInfo | None = None
        t0_bag: float | None = None
        w0 = time.monotonic()
        k = 0
        last_log = w0
        for bm in iter_bag(self.bag, topics):
            if not rclpy.ok():
                break
            if bm.topic == self.info_topic:
                info = bm.msg
                continue
            img: Image = (compressed_to_image(bm.msg) if isinstance(bm.msg, CompressedImage)
                          else bm.msg)
            t_bag = stamp_to_sec(img.header.stamp) or bm.log_time_ns * 1e-9
            if t0_bag is None:
                t0_bag = t_bag
            target = w0 + (k / self.fps if self.fps else (t_bag - t0_bag) / self.rate)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self.max_lag_s = max(self.max_lag_s, -delay)

            if self.restamp:
                img.header.stamp = self.get_clock().now().to_msg()
            if info is not None:
                info.header.stamp = img.header.stamp
                self.info_pub.publish(info)
            self.image_pub.publish(img)
            k += 1
            self.published += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                self.get_logger().info(
                    f"{k} frames, {k / (now - w0):.1f} fps, "
                    f"{img.width}x{img.height} {img.encoding}, max lag {self.max_lag_s * 1e3:.0f} ms")
                last_log = now
        return k


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Parse our options; everything else (``--ros-args ...``) is passed to rclpy."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                epilog="Run with --list first to see the bag's topics.")
    p.add_argument("bag", type=Path, help="bag folder or .mcap file")
    p.add_argument("--topic", help="recorded image topic (default: the up color camera)")
    p.add_argument("--out", default=CAMERA_TOPIC, help=f"output topic (default: {CAMERA_TOPIC})")
    p.add_argument("--rate", type=float, default=1.0, help="playback speed, 1 = real time")
    p.add_argument("--fps", type=float, help="fixed publish rate instead of recorded timing")
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
    image_topic = args.topic or pick_camera_topic(args.bag)
    info_topic = info_topic_for(args.bag, image_topic)

    rclpy.init(args=["replay", *ros_args])
    node = McapCameraReplay(args.bag, image_topic, info_topic, args.rate, args.fps,
                            restamp=not args.keep_stamps, out_topic=args.out)
    # Give subscribers a moment to discover the publisher so frame 0 isn't lost.
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
        print(f"published {node.published} frames, max lag {node.max_lag_s * 1e3:.0f} ms")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
