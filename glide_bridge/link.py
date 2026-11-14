"""Client for the glide_link relay (robot/glide_link_relay.py).

One TCP connection carries every subscribed topic. Images arrive as JPEG or
PNG (16-bit PNG in millimetres for depth), other messages as JSON. See the
relay's docstring for the wire format.

    link = GlideLink("10.42.0.45")
    cam = link.subscribe("/camera_up/color/image_raw", max_rate=6)
    odom = link.subscribe("/odom")
    frame = cam.latest()       # Frame or None; frame.image is RGB uint8
"""

from __future__ import annotations

import io
import json
import socket
import struct
import threading
import time
from collections import deque
from typing import Callable

import numpy as np
from PIL import Image as PILImage

from .client import Frame
from .decode import decode_compressed

HDR = struct.Struct("!II")
DEFAULT_PORT = 8770


def pack(header: dict, payload: bytes = b"") -> bytes:
    """Frame a message: ``!II`` (header_len, payload_len), JSON header, payload."""
    h = json.dumps(header).encode()
    return HDR.pack(len(h), len(payload)) + h + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes; raises ConnectionError if the relay closes first."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("relay closed the connection")
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    """Read one framed message. Returns (JSON header, payload bytes)."""
    hl, pl = HDR.unpack(_recv_exact(sock, HDR.size))
    header = json.loads(_recv_exact(sock, hl))
    return header, (_recv_exact(sock, pl) if pl else b"")


def decode_payload(header: dict, payload: bytes) -> tuple[np.ndarray | dict, str]:
    """Decode one relay message payload.

    Parameters
    ----------
    header : dict
        Message header; ``enc`` selects the decoder (json, passthru, jpeg, png, png16).
    payload : bytes
        Encoded message.

    Returns
    -------
    data : np.ndarray or dict
        (H, W, 3) uint8 RGB, (H, W) uint8 mono/labels, (H, W) float32 metres for
        png16 with ``unit == "mm"`` (``m = mm / 1000``), (H, W) uint16 for other
        png16, or a dict for JSON.
    encoding : str
        Source encoding when known, else the wire encoding.
    """
    enc = header.get("enc")
    if enc == "json":
        return json.loads(payload), "json"
    if enc == "passthru":
        return decode_compressed({"format": header.get("format", ""), "data": payload})
    img = PILImage.open(io.BytesIO(payload))
    if enc == "png16":
        arr: np.ndarray = np.array(img, dtype=np.uint16)   # (H, W)
        if header.get("unit") == "mm":
            return arr.astype(np.float32) / 1000.0, header.get("src_encoding", "16UC1")
        return arr, header.get("src_encoding", "mono16")
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return np.asarray(img), header.get("src_encoding", enc)


class TopicFeed:
    """Newest message on one topic, plus rate and error tracking."""

    def __init__(self, link: "GlideLink", topic: str, decode: bool = True) -> None:
        self.link, self.topic, self.decode = link, topic, decode
        self.type: str | None = None
        self.state = "requested"           # waiting | subscribed | error
        self.status_error = ""
        self._latest: Frame | None = None
        self._arrivals: deque[float] = deque(maxlen=30)
        self._errors, self._last_error = 0, ""
        self.count = 0
        self.max_rate = 0.0
        self.dropped = 0   # frames the relay received but never sent (link too slow);
                           # only counted when max_rate is 0, since a rate cap skips on purpose
        self._last_seq: int | None = None
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)

    # Same names as client.CameraFeed so the viewer can use either.
    @property
    def topic_name(self) -> str:
        return self.topic

    @property
    def msg_type(self) -> str:
        return self.type or "?"

    def _on_frame(self, header: dict, payload: bytes) -> None:
        """Reader-thread callback: count drops from ``seq`` gaps, decode, store, notify."""
        now = time.time()
        seq = header.get("seq")
        with self._lock:
            self.type = header.get("type", self.type)
            self.state = "subscribed"
            if (not self.max_rate and seq is not None and self._last_seq is not None
                    and seq > self._last_seq + 1):
                self.dropped += seq - self._last_seq - 1
            self._last_seq = seq
        if not self.decode:
            data: np.ndarray | dict | None
            data, enc = None, header.get("enc", "")
        else:
            try:
                data, enc = decode_payload(header, payload)
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                return
        frame = Frame(data, enc, header.get("stamp") or 0.0, now, header)
        with self._new:
            self._latest = frame
            self._arrivals.append(now)
            self.count += 1
            self._new.notify_all()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def wait(self, timeout: float = 5.0) -> Frame | None:
        """Block until a message newer than the current one arrives.

        Parameters
        ----------
        timeout : float
            Seconds to wait.

        Returns
        -------
        Frame or None
            The newest frame (possibly the old one if the timeout expired), or None
            if nothing has arrived yet.
        """
        with self._new:
            start = self.count
            self._new.wait_for(lambda: self.count > start, timeout)
            return self._latest

    def fps(self) -> float:
        """Arrival rate over the last 30 messages (Hz); 0 if nothing for 2 s.

        Notes
        -----
        ``fps = (N - 1) / (t_N - t_1)`` over the N most recent arrival times.
        """
        with self._lock:
            t = list(self._arrivals)
        if len(t) < 2 or time.time() - t[-1] > 2.0:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])

    def error(self) -> tuple[int, str]:
        """(number of errors, last error message); relay errors count as one."""
        with self._lock:
            if self.state == "error":
                return max(self._errors, 1), self.status_error
            return self._errors, self._last_error

    def close(self) -> None:
        self.link.unsubscribe(self.topic)


class GlideLink:
    """Connection to a glide_link relay on the robot or next to a simulator."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORT,
                 timeout: float = 8.0) -> None:
        self.host, self.port = host, port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.hello, _ = recv_frame(self.sock)
        if self.hello.get("op") != "hello":
            raise ConnectionError(f"unexpected greeting {self.hello!r}")
        self.sock.settimeout(None)
        self.feeds: dict[str, TopicFeed] = {}
        self.listeners: list[Callable[[dict, bytes], None]] = []
        self.bytes_received = 0
        self.connected = True
        self.disconnect_reason = ""
        self._send_lock = threading.Lock()
        self._topics_reply: dict | None = None
        self._topics_event = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    @property
    def available_topics(self) -> dict[str, str]:
        """Topics the relay saw when this client connected (name -> type)."""
        return dict(self.hello.get("topics", {}))

    def _send(self, header: dict) -> None:
        with self._send_lock:
            self.sock.sendall(pack(header))

    def subscribe(self, topic: str, max_rate: float = 0.0, lossless: bool = False,
                  decode: bool = True) -> TopicFeed:
        """Subscribe to a topic (re-subscribing updates its options).

        Parameters
        ----------
        topic : str
            ROS topic name.
        max_rate : float
            Rate cap in Hz applied on the relay; 0 = every message.
        lossless : bool
            PNG instead of JPEG for color images.
        decode : bool
            Decode payloads into ``Frame.image``; False leaves it None (raw bytes
            go to listeners only).

        Returns
        -------
        TopicFeed
            Feed holding the newest message.
        """
        feed = self.feeds.get(topic)
        if feed is None:
            feed = self.feeds[topic] = TopicFeed(self, topic, decode)
        feed.max_rate = max_rate
        self._send({"op": "subscribe",
                    "topics": [{"topic": topic, "max_rate": max_rate, "lossless": lossless}]})
        return feed

    def unsubscribe(self, topic: str) -> None:
        self.feeds.pop(topic, None)
        if self.connected:
            self._send({"op": "unsubscribe", "topics": [topic]})

    def add_listener(self, fn: Callable[[dict, bytes], None]) -> None:
        """Call fn(header, payload) for every message, before decoding (used by the recorder)."""
        self.listeners.append(fn)

    def list_topics(self, timeout: float = 5.0) -> dict[str, str]:
        """Ask the relay for the current topic table (name -> type); {} on timeout."""
        self._topics_event.clear()
        self._send({"op": "list"})
        self._topics_event.wait(timeout)
        return dict(self._topics_reply or {})

    def _read_loop(self) -> None:
        """Reader thread: dispatch msg/status/topics frames until the link drops."""
        try:
            while True:
                header, payload = recv_frame(self.sock)
                self.bytes_received += HDR.size + len(payload)
                op = header.get("op")
                if op == "msg":
                    for fn in self.listeners:
                        fn(header, payload)
                    feed = self.feeds.get(header["topic"])
                    if feed is not None:
                        feed._on_frame(header, payload)
                elif op == "status":
                    feed = self.feeds.get(header.get("topic"))
                    if feed is not None:
                        feed.state = header.get("state", feed.state)
                        feed.type = header.get("type", feed.type)
                        feed.status_error = header.get("error", "")
                elif op == "topics":
                    self._topics_reply = header.get("topics", {})
                    self._topics_event.set()
        except (OSError, ConnectionError, ValueError) as exc:
            self.disconnect_reason = str(exc)
        finally:
            self.connected = False

    def close(self) -> None:
        self.connected = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
