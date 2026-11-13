#!/usr/bin/env python3
"""glide_link relay: serve ROS 2 topics to a laptop over one TCP connection.

Runs on the Glide robot (ROS 2 Humble, Python 3.11) or next to a simulator.
Needs only rclpy, rosidl_runtime_py, numpy and OpenCV, which the robot already
has, so nothing is installed on the robot.

Wire format (both directions): every frame is
    !II  header_len, payload_len
    header_len bytes of UTF-8 JSON
    payload_len bytes of payload
Client -> relay headers:
    {"op": "subscribe", "topics": [{"topic": "/x", "max_rate": 6, "lossless": false}]}
    {"op": "unsubscribe", "topics": ["/x"]}
    {"op": "list"}                         -> relay answers {"op": "topics", ...}
Relay -> client headers:
    {"op": "hello", "proto": 1, "host": ..., "topics": {name: type}}
    {"op": "msg", "topic", "type", "seq", "stamp", "recv", "enc", ...}
      enc "jpeg"     payload is a JPEG (color, RGB order once decoded)
      enc "png"      payload is a PNG (8-bit mono/labels or 8-bit color)
      enc "png16"    payload is a 16-bit PNG; header "unit" says "mm" for depth
      enc "json"     payload is the message as JSON
      enc "passthru" payload is a CompressedImage's bytes; header "format"
    {"op": "status", "topic", "state": "waiting"|"subscribed", ...}
Only the newest message per topic is kept for each client, so a slow Wi-Fi
link drops old frames instead of building a backlog. "seq" counts messages the
relay received on that topic, so gaps show where frames were dropped.

Usage on the robot: run via run_relay.sh, which sets the same DDS environment
as glide_launch.service.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message

PROTO = 1
HDR = struct.Struct("!II")
MAX_HEADER = 1 << 20
COLOR_ENCODINGS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
# Message types that are too large to send as JSON unless asked for explicitly.
HEAVY_TYPES = {"sensor_msgs/msg/PointCloud2", "sensor_msgs/msg/LaserScan"}

# Best effort + volatile is compatible with every publisher QoS.
SUB_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)


# --------------------------------------------------------------------- encoding

def _jsonable(o):
    if isinstance(o, (bytes, bytearray)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    try:
        return list(o)
    except TypeError:
        return str(o)


def _stamp(msg) -> float | None:
    h = getattr(msg, "header", None)
    if h is None:
        return None
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _image_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> array (H, W[, C]), honouring row padding."""
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        dtype, ch = np.dtype(np.uint16), 1
    elif enc == "32fc1":
        dtype, ch = np.dtype(np.float32), 1
    elif enc in COLOR_ENCODINGS:
        dtype, ch = np.dtype(np.uint8), COLOR_ENCODINGS[enc]
    elif enc in ("mono8", "8uc1"):
        dtype, ch = np.dtype(np.uint8), 1
    elif enc == "8uc3":
        dtype, ch = np.dtype(np.uint8), 3
    else:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")
    h, w = msg.height, msg.width
    row = np.frombuffer(bytes(msg.data), np.uint8).reshape(h, msg.step)
    row = row[:, : w * ch * dtype.itemsize]
    arr = row.copy().view(dtype).reshape((h, w, ch) if ch > 1 else (h, w))
    return arr.astype(dtype.newbyteorder("="), copy=False)


def encode_image(msg, lossless: bool, quality: int) -> tuple[dict, bytes]:
    enc = msg.encoding.lower()
    arr = _image_array(msg)
    meta = {"width": msg.width, "height": msg.height, "src_encoding": msg.encoding,
            "frame_id": msg.header.frame_id}
    if enc in COLOR_ENCODINGS:
        if enc.startswith("rgb"):
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        else:
            bgr = arr[..., :3]
        if lossless:
            ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            meta["enc"] = "png"
        else:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            meta["enc"] = "jpeg"
    elif enc == "8uc3":
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        meta["enc"] = "png"
    elif arr.dtype == np.uint8:
        # mono / IR / segmentation labels: always lossless so labels survive.
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        meta["enc"] = "png"
    else:
        if arr.dtype == np.float32:      # metres -> millimetres
            mm = np.nan_to_num(arr * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(mm, 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        meta["enc"] = "png16"
        meta["unit"] = "mm" if enc in ("16uc1", "32fc1") else "raw"
    if not ok:
        raise ValueError("image encoding failed")
    return meta, buf.tobytes()


def encode_message(msg, type_name: str, lossless: bool, quality: int) -> tuple[dict, bytes]:
    if type_name == "sensor_msgs/msg/Image":
        return encode_image(msg, lossless, quality)
    if type_name == "sensor_msgs/msg/CompressedImage":
        return ({"enc": "passthru", "format": msg.format, "frame_id": msg.header.frame_id},
                bytes(msg.data))
    d = message_to_ordereddict(msg, truncate_length=None)
    return {"enc": "json"}, json.dumps(d, default=_jsonable).encode()


# ---------------------------------------------------------------------- framing

def pack(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(header, default=_jsonable).encode()
    return HDR.pack(len(h), len(payload)) + h + payload


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    hl, pl = HDR.unpack(recv_exact(sock, HDR.size))
    if hl > MAX_HEADER:
        raise ConnectionError("header too large")
    header = json.loads(recv_exact(sock, hl))
    payload = recv_exact(sock, pl) if pl else b""
    return header, payload


# ---------------------------------------------------------------------- clients

class Client:
    def __init__(self, sock: socket.socket, addr, relay: "Relay"):
        self.sock, self.addr, self.relay = sock, addr, relay
        self.subs: dict[str, dict] = {}        # topic -> {max_rate, lossless, last_sent}
        self.pending: "OrderedDict[str, list[bytes]]" = OrderedDict()
        self.control: list[bytes] = []
        self.cv = threading.Condition()
        self.alive = True
        self.sent_bytes = 0

    def wants(self, topic: str, now: float) -> dict | None:
        s = self.subs.get(topic)
        if s is None:
            return None
        rate = s.get("max_rate") or 0
        if rate > 0 and now - s["last_sent"] < 1.0 / rate * 0.95:
            return None
        return s

    def offer(self, topic: str, frame: bytes, now: float, keep_all: bool = False):
        """Queue a frame. Images keep only the newest; small messages (keep_all) queue
        up to 500 so odometry and IMU arrive complete."""
        with self.cv:
            sub = self.subs.get(topic)
            if sub is None:          # unsubscribed meanwhile
                return
            sub["last_sent"] = now
            q = self.pending.get(topic)
            if keep_all and q is not None:
                q.append(frame)
                del q[:-500]
            else:
                self.pending[topic] = [frame]
            self.pending.move_to_end(topic)
            self.cv.notify()

    def send_control(self, frame: bytes):
        with self.cv:
            self.control.append(frame)
            self.cv.notify()

    def writer(self):
        try:
            while self.alive:
                with self.cv:
                    while self.alive and not self.pending and not self.control:
                        self.cv.wait(1.0)
                    batch = self.control + [f for q in self.pending.values() for f in q]
                    self.control, self.pending = [], OrderedDict()
                for frame in batch:
                    self.sock.sendall(frame)
                    self.sent_bytes += len(frame)
        except OSError:
            pass
        finally:
            self.close()

    def reader(self):
        try:
            while self.alive:
                header, _ = recv_frame(self.sock)
                self.relay.handle_request(self, header)
        except (OSError, ConnectionError, ValueError):
            pass
        finally:
            self.close()

    def close(self):
        if not self.alive:
            return
        self.alive = False
        with self.cv:
            self.cv.notify_all()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.relay.drop_client(self)


# ------------------------------------------------------------------------ relay

class Relay(Node):
    def __init__(self, args):
        super().__init__("glide_link_relay")
        self.args = args
        self.lock = threading.Lock()
        self.peers: list[Client] = []
        self.ros_subs: dict[str, object] = {}  # topic -> rclpy subscription
        self.types: dict[str, str] = {}
        self.seq: dict[str, int] = {}
        self.wanted: set[str] = set()
        self.errors: dict[str, str] = {}
        self.create_timer(1.0, self._resolve_pending)
        self.create_timer(args.stats_every, self._log_stats) if args.stats_every > 0 else None
        self._last_stats = (time.time(), 0)

    # --- ROS side
    def topic_table(self) -> dict[str, str]:
        return {name: types[0] for name, types in self.get_topic_names_and_types()}

    def _resolve_pending(self):
        with self.lock:
            missing = [t for t in self.wanted if t not in self.ros_subs]
        if not missing:
            return
        table = self.topic_table()
        for topic in missing:
            if topic in table:
                self._subscribe_ros(topic, table[topic])

    def _subscribe_ros(self, topic: str, type_name: str):
        try:
            cls = get_message(type_name)
        except Exception as exc:  # message package not on this machine
            self.errors[topic] = f"cannot import {type_name}: {exc}"
            self._broadcast_status(topic, "error", error=self.errors[topic])
            with self.lock:
                self.wanted.discard(topic)
            return
        sub = self.create_subscription(cls, topic, lambda m, t=topic: self._on_msg(t, m),
                                       self._qos_for(topic, type_name))
        with self.lock:
            self.ros_subs[topic] = sub
            self.types[topic] = type_name
            self.seq.setdefault(topic, 0)
        self.get_logger().info(f"subscribed {topic} [{type_name}]")
        self._broadcast_status(topic, "subscribed", type=type_name)

    def _qos_for(self, topic: str, type_name: str) -> QoSProfile:
        """Best effort for images; otherwise match the publishers, so latched topics
        such as /tf_static (reliable + transient local) still deliver their last message."""
        if type_name.startswith("sensor_msgs/msg/") and "Image" in type_name:
            return SUB_QOS
        try:
            infos = self.get_publishers_info_by_topic(topic)
        except Exception:
            return SUB_QOS
        if not infos:
            return SUB_QOS
        reliable = all(i.qos_profile.reliability == ReliabilityPolicy.RELIABLE for i in infos)
        latched = all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for i in infos)
        return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10 if latched else 5,
                          reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE)

    def _on_msg(self, topic: str, msg):
        now = time.time()
        with self.lock:
            self.seq[topic] += 1
            seq = self.seq[topic]
            targets = [(c, c.wants(topic, now)) for c in self.peers]
        targets = [(c, s) for c, s in targets if s is not None]
        if not targets:
            return
        type_name = self.types[topic]
        encoded: dict[bool, bytes] = {}
        for client, sub in targets:
            lossless = bool(sub.get("lossless"))
            if lossless not in encoded:
                try:
                    meta, payload = encode_message(msg, type_name, lossless, self.args.jpeg_quality)
                except Exception as exc:
                    if self.errors.get(topic) != str(exc):
                        self.errors[topic] = str(exc)
                        self.get_logger().warning(f"{topic}: {exc}")
                    return
                header = {"op": "msg", "topic": topic, "type": type_name, "seq": seq,
                          "stamp": _stamp(msg), "recv": now}
                header.update(meta)
                encoded[lossless] = pack(header, payload)
            client.offer(topic, encoded[lossless], now, keep_all=meta["enc"] == "json")

    # --- client side
    def add_client(self, sock, addr):
        c = Client(sock, addr, self)
        with self.lock:
            self.peers.append(c)
        self.get_logger().info(f"client connected {addr[0]}:{addr[1]}")
        c.send_control(pack({"op": "hello", "proto": PROTO, "host": socket.gethostname(),
                             "topics": self.topic_table()}))
        threading.Thread(target=c.writer, daemon=True).start()
        threading.Thread(target=c.reader, daemon=True).start()

    def drop_client(self, c: Client):
        with self.lock:
            if c in self.peers:
                self.peers.remove(c)
            still = {t for cl in self.peers for t in cl.subs}
            idle = [t for t in self.ros_subs if t not in still]
            self.wanted &= still
            subs = [self.ros_subs.pop(t) for t in idle]
        for s in subs:   # stop receiving topics nobody wants (saves robot CPU)
            self.destroy_subscription(s)
        self.get_logger().info(f"client disconnected {c.addr[0]}:{c.addr[1]} "
                               f"({c.sent_bytes / 1e6:.1f} MB sent)")

    def handle_request(self, c: Client, req: dict):
        op = req.get("op")
        if op == "subscribe":
            table = None
            for spec in req.get("topics", []):
                if isinstance(spec, str):
                    spec = {"topic": spec}
                topic = spec["topic"]
                with self.lock:
                    c.subs[topic] = {"max_rate": float(spec.get("max_rate") or 0),
                                     "lossless": bool(spec.get("lossless")), "last_sent": 0.0}
                    self.wanted.add(topic)
                    known = topic in self.ros_subs
                if known:
                    c.send_control(pack({"op": "status", "topic": topic, "state": "subscribed",
                                         "type": self.types[topic]}))
                    continue
                table = table if table is not None else self.topic_table()
                if topic in table:
                    if table[topic] in HEAVY_TYPES and not spec.get("allow_heavy"):
                        with self.lock:
                            c.subs.pop(topic, None)
                        c.send_control(pack({"op": "status", "topic": topic, "state": "error",
                                             "error": f"{table[topic]} is too large to stream; "
                                                      "pass allow_heavy"}))
                        continue
                    self._subscribe_ros(topic, table[topic])
                else:
                    c.send_control(pack({"op": "status", "topic": topic, "state": "waiting",
                                         "error": "no publisher yet; will subscribe when it appears"}))
        elif op == "unsubscribe":
            with self.lock:
                for t in req.get("topics", []):
                    c.subs.pop(t, None)
        elif op == "list":
            c.send_control(pack({"op": "topics", "topics": self.topic_table()}))
        elif op == "ping":
            c.send_control(pack({"op": "pong", "t": req.get("t"), "relay_time": time.time()}))

    def _broadcast_status(self, topic, state, **extra):
        with self.lock:
            clients = [c for c in self.peers if topic in c.subs]
        for c in clients:
            c.send_control(pack(dict({"op": "status", "topic": topic, "state": state}, **extra)))

    def _log_stats(self):
        with self.lock:
            sent = sum(c.sent_bytes for c in self.peers)
            n = len(self.peers)
        t0, s0 = self._last_stats
        now = time.time()
        if n:
            self.get_logger().info(f"{n} client(s), {(sent - s0) / (now - t0) / 1e6:.2f} MB/s, "
                                   f"{len(self.ros_subs)} topic(s)")
        self._last_stats = (now, sent)


def serve(relay: Relay, host: str, port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    relay.get_logger().info(f"glide_link relay listening on {host}:{port}")
    while True:
        sock, addr = srv.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        relay.add_client(sock, addr)


def main():
    p = argparse.ArgumentParser(description="Serve ROS 2 topics to glide_bridge clients over TCP")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--stats-every", type=float, default=10.0,
                   help="seconds between throughput log lines (0 = off)")
    args = p.parse_args()

    rclpy.init()
    relay = Relay(args)
    threading.Thread(target=serve, args=(relay, args.host, args.port), daemon=True).start()
    try:
        rclpy.spin(relay)
    except KeyboardInterrupt:
        pass
    finally:
        relay.destroy_node()
        rclpy.try_shutdown() if hasattr(rclpy, "try_shutdown") else rclpy.shutdown()


if __name__ == "__main__":
    main()
