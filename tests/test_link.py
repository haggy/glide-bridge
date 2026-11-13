import io
import json
import socket
import threading

import numpy as np
from PIL import Image

from glide_bridge.link import decode_payload, pack, recv_frame


def png(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()


def test_framing_roundtrip_over_socket():
    a, b = socket.socketpair()
    payload = bytes(range(256)) * 1000
    data = pack({"op": "msg", "topic": "/x"}, payload) + pack({"op": "list"})
    t = threading.Thread(target=a.sendall, args=(data,)); t.start()   # larger than the socket buffer
    h1, p1 = recv_frame(b)
    h2, p2 = recv_frame(b)
    assert h1 == {"op": "msg", "topic": "/x"} and p1 == payload
    assert h2 == {"op": "list"} and p2 == b""
    t.join()
    a.close(); b.close()


def test_png16_depth_mm_to_metres():
    mm = np.array([[0, 1000], [2500, 65535]], np.uint16)
    out, enc = decode_payload({"enc": "png16", "unit": "mm", "src_encoding": "16UC1"}, png(mm))
    assert out.dtype == np.float32 and enc == "16UC1"
    np.testing.assert_allclose(out, mm / 1000.0)


def test_png_labels_are_lossless():
    labels = np.array([[0, 1, 2], [2, 1, 0]], np.uint8)
    out, _ = decode_payload({"enc": "png", "src_encoding": "mono8"}, png(labels))
    np.testing.assert_array_equal(out, labels)


def test_jpeg_color_is_rgb():
    img = np.zeros((16, 16, 3), np.uint8); img[..., 0] = 255   # pure red
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG", quality=95)
    out, _ = decode_payload({"enc": "jpeg", "src_encoding": "rgb8"}, buf.getvalue())
    assert out.shape == (16, 16, 3) and out[..., 0].mean() > 240 and out[..., 2].mean() < 15


def test_json_payload():
    out, enc = decode_payload({"enc": "json"}, json.dumps({"pose": {"x": 1.5}}).encode())
    assert enc == "json" and out["pose"]["x"] == 1.5
