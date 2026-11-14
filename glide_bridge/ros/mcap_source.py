"""Read typed ROS 2 messages out of rosbag2 MCAP files.

Messages are deserialized with ``rclpy.serialization`` into the real ``sensor_msgs``
classes, so they can be republished unchanged, byte for byte, apart from the header
fields the replay node chooses to overwrite. File discovery and topic listing come
from :mod:`glide_bridge.mcap_utils`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ..mcap_utils import list_topics, mcap_files
from ..topics import CameraStream

IMAGE_TYPES: frozenset[str] = frozenset({"sensor_msgs/msg/Image",
                                         "sensor_msgs/msg/CompressedImage"})
CAMERA_INFO_TYPE: str = "sensor_msgs/msg/CameraInfo"

# How a live topic can appear in a bag: as published, rate-reduced by the robot's
# data-collection recorder (``_rfps``), or through the compressed image_transport.
RECORDED_SUFFIXES: tuple[str, ...] = ("", "_rfps", "/compressed")


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


@dataclass(frozen=True)
class RecordedStream:
    """Where one live camera stream is stored in a bag.

    Attributes
    ----------
    stream : CameraStream
        The live stream (its ``base_topic``/``info_topic`` are what gets published).
    image_topic : str
        Recorded image topic, e.g. ``/camera_up/color/image_raw_rfps``.
    info_topic : str or None
        Recorded CameraInfo topic, or None if the bag has none for this camera.
    count : int
        Number of recorded image messages.
    """

    stream: CameraStream
    image_topic: str
    info_topic: str | None
    count: int


def find_recorded(path: str | Path, stream: CameraStream) -> RecordedStream | None:
    """Find a live camera stream's recorded topics in a bag.

    Parameters
    ----------
    path : str or Path
        Bag folder or ``.mcap`` file.
    stream : CameraStream
        Live stream from :data:`glide_bridge.topics.PRESETS`.

    Returns
    -------
    RecordedStream or None
        None if no image topic in :data:`RECORDED_SUFFIXES` form is in the bag.
        If several forms are present, the one with the most frames wins.
    """
    topics = list_topics(path)
    found = [(topics[t][1], t) for t in (stream.base_topic + s for s in RECORDED_SUFFIXES)
             if t in topics and topics[t][0] in IMAGE_TYPES]
    if not found:
        return None
    count, image_topic = max(found)
    info = stream.info_topic
    info_topic = info if info and topics.get(info, ("", 0))[0] == CAMERA_INFO_TYPE else None
    return RecordedStream(stream, image_topic, info_topic, count)


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
