"""Camera topic names for the Glide robot (Glidance alpha 2) and the Gazebo test sim."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraStream:
    """A camera topic and how to show it.

    Attributes
    ----------
    name : str
        Short stream name used on the command line (e.g. ``up_color``).
    base_topic : str
        Raw ``sensor_msgs/Image`` topic.
    kind : str
        ``"color"``, ``"depth"``, ``"mono"`` or ``"labels"``.
    info_topic : str or None
        ``sensor_msgs/CameraInfo`` topic with the intrinsics.
    """

    name: str
    base_topic: str          # raw sensor_msgs/Image topic
    kind: str                # "color", "depth", "mono" or "labels"
    info_topic: str | None = None   # sensor_msgs/CameraInfo for intrinsics

    def topic(self, transport: str) -> tuple[str, str]:
        """Topic and message type for an image_transport transport.

        Parameters
        ----------
        transport : str
            ``"raw"`` or ``"compressed"`` (depth uses ``compressedDepth``).

        Returns
        -------
        tuple[str, str]
            (topic name, rosbridge message type).
        """
        if transport == "raw":
            return self.base_topic, "sensor_msgs/Image"
        suffix = "compressedDepth" if self.kind == "depth" else "compressed"
        return f"{self.base_topic}/{suffix}", "sensor_msgs/CompressedImage"


PRESETS: dict[str, dict[str, CameraStream]] = {
    # sim_navigate.launch.py with the realsense_gazebo_plugin: 1280x720
    "sim": {
        "color": CameraStream("color", "/color/image_raw", "color", "/color/camera_info"),
        "depth": CameraStream("depth", "/depth/image_rect_raw", "depth", "/depth/camera_info"),
        "infra1": CameraStream("infra1", "/infra1/image_raw", "mono"),
        "infra2": CameraStream("infra2", "/infra2/image_raw", "mono"),
    },
    # Glidance alpha 2 (alpha033), two Orbbec cameras. Normal config: color 424x240,
    # depth 480x270, 6 fps. Data-collection config: 640x360, depth registered to color.
    "glide": {
        "up_color": CameraStream("up_color", "/camera_up/color/image_raw", "color",
                                 "/camera_up/color/camera_info"),
        "up_depth": CameraStream("up_depth", "/camera_up/depth/image_raw", "depth",
                                 "/camera_up/depth/camera_info"),
        "down_color": CameraStream("down_color", "/camera_down/color/image_raw", "color",
                                   "/camera_down/color/camera_info"),
        "down_depth": CameraStream("down_depth", "/camera_down/depth/image_raw", "depth",
                                   "/camera_down/depth/camera_info"),
        "up_ir": CameraStream("up_ir", "/camera_up/left_ir/image_raw", "mono",
                              "/camera_up/left_ir/camera_info"),
        "down_ir": CameraStream("down_ir", "/camera_down/left_ir/image_raw", "mono",
                                "/camera_down/left_ir/camera_info"),
        # 0 other, 1 walkspace, 2 road; only published outdoors while operating
        "segmentation": CameraStream("segmentation", "/segmentation/segmentation_map", "labels"),
        "seg_viz": CameraStream("seg_viz", "/segmentation/visualization", "color"),
    },
}

DEFAULT_STREAMS = {
    "glide": ["up_color", "up_depth", "down_color", "down_depth"],
    "sim": ["color", "depth"],
}

# Non-image topics worth recording alongside the cameras on the robot.
GLIDE_TELEMETRY = [
    "/odom", "/imu", "/gps/fix", "/gps/velocity", "/glide_mode",
    "/inside_outside/is_inside", "/brake_ratio", "/wheels_headings",
    "/camera_up/depth_to_color", "/camera_down/depth_to_color", "/tf_static",
]
