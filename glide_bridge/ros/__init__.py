"""ROS 2 nodes for streaming Glide camera images over pub/sub.

Everything in this subpackage needs ``rclpy`` and runs in the ``glide_ros`` conda env
(ROS 2 Humble, see ``env/ros_humble.yml``). Nothing outside this subpackage imports it,
so the uv env keeps working without ROS.

Pipeline contract
-----------------
Camera images use the **robot's own topic names**, from
``glide_bridge.topics.PRESETS["glide"]``, so the topic identifies the camera and a
consumer works unchanged against the live Glide:

=============  ===============================  ==================================
stream         image (``sensor_msgs/Image``)    intrinsics (``CameraInfo``)
=============  ===============================  ==================================
``up_color``   ``/camera_up/color/image_raw``   ``/camera_up/color/camera_info``
``down_color`` ``/camera_down/color/image_raw`` ``/camera_down/color/camera_info``
=============  ===============================  ==================================

The up camera faces forward at handle height; the down camera looks at the ground in
front of the wheels. Depth and IR streams follow the same pattern. Today the source
is :mod:`glide_bridge.ros.replay` (an MCAP recording, whose ``*_rfps`` topics are
republished without the suffix); later it is the robot itself. Consumers such as
:mod:`glide_bridge.ros.viewer` subscribe by stream name.
"""

import sys as _sys

try:
    import rclpy as _rclpy  # noqa: F401  (only checking it's importable)
except ImportError as _exc:
    raise ImportError(
        f"glide_bridge.ros needs ROS 2 (rclpy), which this Python doesn't have "
        f"({_sys.executable}). Activate the ROS env first:\n\n"
        f"    source env/activate_ros.sh\n\n"
        f"(one-time setup: see 'ROS 2 camera streams' in README.md)") from _exc

from ..topics import PRESETS, CameraStream

GLIDE_STREAMS: dict[str, CameraStream] = PRESETS["glide"]
# Forward and down cameras, color and depth: the viewer shows them as a 2x2 grid.
DEFAULT_STREAMS: list[str] = ["up_color", "down_color", "up_depth", "down_depth"]
