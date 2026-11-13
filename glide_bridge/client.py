"""Subscribe to camera topics over rosbridge (Gazebo test sim) and keep the latest frame of each.

No ROS install is needed on this machine: roslibpy talks to a rosbridge
WebSocket server running next to ROS (the sim container or the robot).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import roslibpy

from .decode import decode
from .topics import CameraStream


@dataclass
class Frame:
    image: np.ndarray
    encoding: str
    stamp: float            # ROS header time, seconds
    received: float         # local time.time() when decoded
    header: dict | None = None   # glide_link message header (topic, seq, enc, ...)


@dataclass
class _FeedState:
    latest: Frame | None = None
    arrivals: deque = field(default_factory=lambda: deque(maxlen=30))
    errors: int = 0
    last_error: str = ""


class CameraFeed:
    """One camera topic. ``latest()`` returns the newest decoded frame or None."""

    def __init__(self, ros: roslibpy.Ros, stream: CameraStream,
                 transport: str = "compressed", max_rate_hz: float = 15.0):
        self.stream = stream
        self.transport = transport
        self.topic_name, self.msg_type = stream.topic(transport)
        self._state = _FeedState()
        self._lock = threading.Lock()
        self._topic = roslibpy.Topic(
            ros, self.topic_name, self.msg_type,
            throttle_rate=int(1000 / max_rate_hz) if max_rate_hz > 0 else 0,
            queue_length=1,          # drop old frames rather than queueing them
        )
        self._topic.subscribe(self._on_message)

    def _on_message(self, msg: dict) -> None:
        try:
            image, encoding = decode(msg, self.msg_type)
        except Exception as exc:  # keep the stream alive on one bad frame
            with self._lock:
                self._state.errors += 1
                self._state.last_error = f"{type(exc).__name__}: {exc}"
            return
        stamp = msg.get("header", {}).get("stamp", {})
        now = time.time()
        frame = Frame(image, encoding,
                      stamp.get("sec", 0) + stamp.get("nanosec", 0) * 1e-9, now)
        with self._lock:
            self._state.latest = frame
            self._state.arrivals.append(now)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._state.latest

    def fps(self) -> float:
        with self._lock:
            t = list(self._state.arrivals)
        if len(t) < 2 or time.time() - t[-1] > 2.0:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])

    def error(self) -> tuple[int, str]:
        with self._lock:
            return self._state.errors, self._state.last_error

    def close(self) -> None:
        self._topic.unsubscribe()


class GlideBridge:
    """Connection to a rosbridge server plus any number of camera feeds."""

    def __init__(self, host: str = "localhost", port: int = 9090,
                 timeout: float = 10.0):
        self.ros = roslibpy.Ros(host=host, port=port)
        self.ros.run(timeout=timeout)   # starts the client in a background thread
        self.feeds: dict[str, CameraFeed] = {}

    def add_camera(self, stream: CameraStream, transport: str = "compressed",
                   max_rate_hz: float = 15.0) -> CameraFeed:
        feed = CameraFeed(self.ros, stream, transport, max_rate_hz)
        self.feeds[stream.name] = feed
        return feed

    def topics(self) -> list[str]:
        """Topics the ROS graph currently advertises (via rosbridge's rosapi, if running)."""
        try:
            return sorted(self.ros.get_topics())
        except Exception:
            return []

    def close(self) -> None:
        for feed in self.feeds.values():
            feed.close()
        self.ros.terminate()
