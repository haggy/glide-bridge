"""ROS 2 nodes for streaming Glide camera images over pub/sub.

Everything in this subpackage needs ``rclpy`` and runs in the ``glide_ros`` conda env
(ROS 2 Humble, see ``env/ros_humble.yml``). Nothing outside this subpackage imports it,
so the uv env keeps working without ROS.

Pipeline contract
-----------------
Sources publish ``sensor_msgs/msg/Image`` on ``/camera_raw`` and the matching
``sensor_msgs/msg/CameraInfo`` on ``/camera_raw/camera_info``. Today the source is
:mod:`glide_bridge.ros.replay` (an MCAP recording); later it will be the live Glide
camera remapped to the same topic. Consumers such as :mod:`glide_bridge.ros.viewer`
only ever subscribe to ``/camera_raw``.
"""

import sys as _sys

try:
    import rclpy as _rclpy  # noqa: F401  (only checking it's importable)
except ImportError as _exc:
    raise ImportError(
        f"glide_bridge.ros needs ROS 2 (rclpy), which this Python doesn't have "
        f"({_sys.executable}). Activate the ROS env first:\n\n"
        f"    source env/activate_ros.sh\n\n"
        f"(one-time setup: see 'ROS 2 camera stream' in README.md)") from _exc

CAMERA_TOPIC: str = "/camera_raw"
CAMERA_INFO_TOPIC: str = "/camera_raw/camera_info"
