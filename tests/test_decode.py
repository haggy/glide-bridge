import base64
import io
import struct

import numpy as np
from PIL import Image

from glide_bridge.decode import decode, decode_raw
from glide_bridge.topics import PRESETS


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def raw_msg(arr: np.ndarray, encoding: str, pad: int = 0) -> dict:
    h, w = arr.shape[:2]
    row = arr.reshape(h, -1).view(np.uint8)
    step = row.shape[1] + pad
    padded = np.zeros((h, step), np.uint8)
    padded[:, : row.shape[1]] = row
    return {"height": h, "width": w, "encoding": encoding, "is_bigendian": 0,
            "step": step, "data": b64(padded.tobytes())}


def test_raw_rgb8_roundtrip() -> None:
    img = np.random.default_rng(0).integers(0, 255, (4, 5, 3), dtype=np.uint8)
    out, enc = decode_raw(raw_msg(img, "rgb8"))
    assert enc == "rgb8"
    np.testing.assert_array_equal(out, img)


def test_raw_bgr8_is_swapped_to_rgb() -> None:
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0] = 255  # blue channel in BGR
    out, _ = decode_raw(raw_msg(img, "bgr8"))
    assert (out[..., 2] == 255).all() and (out[..., 0] == 0).all()


def test_raw_16uc1_depth_to_metres_with_row_padding() -> None:
    depth_mm = np.array([[0, 1000], [2500, 65535]], np.uint16)
    out, _ = decode_raw(raw_msg(depth_mm, "16UC1", pad=4))
    np.testing.assert_allclose(out, depth_mm / 1000.0)
    assert out.dtype == np.float32


def test_compressed_png_color() -> None:
    img = np.random.default_rng(1).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    msg = {"format": "rgb8; png compressed bgr8", "data": b64(buf.getvalue())}
    out, _ = decode(msg, "sensor_msgs/CompressedImage")
    np.testing.assert_array_equal(out, img)


def test_compressed_jpeg_shape() -> None:
    img = np.full((16, 24, 3), 128, np.uint8)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG")
    msg = {"format": "rgb8; jpeg compressed bgr8", "data": b64(buf.getvalue())}
    out, _ = decode(msg, "sensor_msgs/CompressedImage")
    assert out.shape == (16, 24, 3) and abs(int(out.mean()) - 128) <= 2


def test_compressed_depth_16uc1() -> None:
    depth_mm = np.array([[0, 500], [1234, 4000]], np.uint16)
    buf = io.BytesIO(); Image.fromarray(depth_mm).save(buf, format="PNG")
    payload = struct.pack("<iff", 0, 0.0, 0.0) + buf.getvalue()
    msg = {"format": "16UC1; compressedDepth png", "data": b64(payload)}
    out, _ = decode(msg, "sensor_msgs/CompressedImage")
    np.testing.assert_allclose(out, depth_mm / 1000.0, rtol=1e-6)


def test_compressed_depth_32fc1_inverse_quantization() -> None:
    quant_a, quant_b = 100.0, 0.0      # value = a / depth + b
    depth = np.array([[0.0, 1.0], [2.0, 4.0]], np.float32)
    values = np.zeros(depth.shape, np.uint16)
    valid = depth > 0
    values[valid] = np.round(quant_a / depth[valid] + quant_b)
    buf = io.BytesIO(); Image.fromarray(values).save(buf, format="PNG")
    payload = struct.pack("<iff", 0, quant_a, quant_b) + buf.getvalue()
    msg = {"format": "32FC1; compressedDepth png", "data": b64(payload)}
    out, _ = decode(msg, "sensor_msgs/CompressedImage")
    np.testing.assert_allclose(out, depth, rtol=1e-3)


def test_presets_topic_names() -> None:
    assert PRESETS["sim"]["color"].topic("compressed") == (
        "/color/image_raw/compressed", "sensor_msgs/CompressedImage")
    assert PRESETS["glide"]["up_color"].base_topic == "/camera_up/color/image_raw"
    assert PRESETS["glide"]["down_depth"].info_topic == "/camera_down/depth/camera_info"
