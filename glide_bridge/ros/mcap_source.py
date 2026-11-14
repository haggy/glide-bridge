"""Read typed ROS 2 messages out of rosbag2 MCAP files.

Messages are deserialized with ``rclpy.serialization`` into the real ``sensor_msgs``
classes, so they can be republished unchanged, byte for byte, apart from the header
fields the replay node chooses to overwrite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ..mcap_utils import list_topics, mcap_files

IMAGE_TYPES: frozenset[str] = frozenset({"sensor_msgs/msg/Image",
                                         "sensor_msgs/msg/CompressedImage"})
CAMERA_INFO_TYPE: str = "sensor_msgs/msg/CameraInfo"


@dataclass(frozen=True)
class BagMessage:
    """One message from a bag.

    Attributes
    ----------
    topic : str
        Channel topic as recorded.
    type_name : str
        ROS type, e.g. ``sensor_msgs/msg/Image``.
    log_time_ns : int
        Time rosbag2 wrote the message, ns since the Unix epoch.
    msg : Any
        Deserialized ROS message instance.
    """

    topic: str
    type_name: str
    log_time_ns: int
    msg: Any


def image_topics(path: str | Path) -> dict[str, str]:
    """Image topics in a bag, ``topic -> type``."""
    return {t: ty for t, (ty, _) in list_topics(path).items() if ty in IMAGE_TYPES}


def pick_camera_topic(path: str | Path, preferred: Sequence[str] = (
        "/camera_up/color/image_raw_rfps", "/camera_up/color/image_raw",
        "/camera_up/color/image_raw/compressed")) -> str:
    """Choose the color camera topic to replay when none is given.

    Parameters
    ----------
    path : str or Path
        Bag folder or ``.mcap`` file.
    preferred : sequence of str, optional
        Topics tried in order; the up camera faces forward at handle height.

    Returns
    -------
    str
        The first preferred topic present with at least half as many frames as the
        busiest color topic; otherwise the busiest color topic (a warning is
        printed); otherwise the busiest image topic.

    Raises
    ------
    ValueError
        If the bag holds no image topics.

    Notes
    -----
    The frame-count check exists because a camera can drop out during a recording:
    in ``rosbag2_alpha_033_2026-11-12_20-39-49`` the up color camera has 13 frames
    against 172 for the down camera.
    """
    counts = {t: n for t, (ty, n) in list_topics(path).items() if ty in IMAGE_TYPES}
    if not counts:
        raise ValueError(f"no image topics in {path}")
    color = {t: n for t, n in counts.items() if "color" in t} or counts
    busiest = max(sorted(color), key=lambda t: color[t])
    for t in preferred:
        if t in counts:
            if counts[t] * 2 >= color[busiest]:
                return t
            print(f"warning: {t} has only {counts[t]} frames; replaying {busiest} "
                  f"({color[busiest]} frames) instead. Pass --topic to override.",
                  file=sys.stderr)
            break
    return busiest


def info_topic_for(path: str | Path, image_topic: str) -> str | None:
    """Find the ``CameraInfo`` topic that belongs to ``image_topic``, if recorded.

    Notes
    -----
    Matches by camera prefix: ``/camera_up/color/image_raw_rfps`` pairs with a
    ``CameraInfo`` topic under ``/camera_up/color/``.
    """
    prefix = image_topic.rsplit("/", 1)[0] + "/"
    if prefix.endswith("/image_raw/"):          # .../image_raw/compressed
        prefix = prefix[: -len("image_raw/")]
    infos = sorted(t for t, (ty, _) in list_topics(path).items()
                   if ty == CAMERA_INFO_TYPE and t.startswith(prefix))
    return infos[0] if infos else None


def iter_bag(path: str | Path, topics: Sequence[str]) -> Iterator[BagMessage]:
    """Yield deserialized messages on ``topics`` in log-time order.

    Parameters
    ----------
    path : str or Path
        Bag folder (all ``.mcap`` files, in name order) or one ``.mcap`` file.
    topics : sequence of str
        Topics to read; others are skipped without being deserialized.

    Yields
    ------
    BagMessage
    """
    classes: dict[str, type] = {}
    for f in mcap_files(path):
        with open(f, "rb") as fh:
            reader = make_reader(fh)
            for schema, channel, message in reader.iter_messages(topics=list(topics),
                                                                 log_time_order=True):
                type_name = schema.name if schema else ""
                cls = classes.get(type_name)
                if cls is None:
                    cls = classes[type_name] = get_message(type_name)
                yield BagMessage(channel.topic, type_name, message.log_time,
                                 deserialize_message(message.data, cls))
