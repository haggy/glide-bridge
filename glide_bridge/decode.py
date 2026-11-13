"""Decode ROS image messages received as rosbridge JSON into numpy arrays.

rosbridge sends ``uint8[]`` fields (the image bytes) as base64 strings. Three
message shapes are handled:

* ``sensor_msgs/Image`` (raw): pixels laid out by ``encoding``, ``height``, ``step``.
* ``sensor_msgs/CompressedImage`` from the ``compressed`` plugin: a JPEG or PNG.
* ``sensor_msgs/CompressedImage`` from the ``compressedDepth`` plugin: a 12-byte
  header followed by a PNG (16-bit depth, or quantized inverse depth for 32FC1).

Color images are returned as ``uint8`` RGB (H, W, 3) or grayscale (H, W).
Depth images are returned as ``float32`` metres (H, W), with 0 meaning no reading.
"""

from __future__ import annotations

import base64
import io
import struct

import numpy as np
from PIL import Image as PILImage

# encoding -> (numpy dtype, channels)
_RAW_LAYOUTS = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "8UC3": (np.uint8, 3),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}

DEPTH_ENCODINGS = {"16UC1", "mono16", "32FC1"}


def _bytes(data) -> bytes:
    """rosbridge gives base64 text for uint8[]; accept raw bytes or int lists too."""
    if isinstance(data, str):
        return base64.b64decode(data)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return bytes(bytearray(data))


def depth_to_metres(img: np.ndarray, encoding: str) -> np.ndarray:
    """16-bit depth is millimetres (ROS convention); 32FC1 is already metres."""
    if encoding == "32FC1":
        out = img.astype(np.float32, copy=True)
        out[~np.isfinite(out)] = 0.0
        return out
    return img.astype(np.float32) / 1000.0


def decode_raw(msg: dict) -> tuple[np.ndarray, str]:
    """Decode a ``sensor_msgs/Image``. Returns (array, encoding)."""
    encoding = msg["encoding"]
    if encoding not in _RAW_LAYOUTS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    dtype, channels = _RAW_LAYOUTS[encoding]
    h, w, step = msg["height"], msg["width"], msg["step"]
    buf = np.frombuffer(_bytes(msg["data"]), dtype=np.uint8)
    rows = buf[: h * step].reshape(h, step)
    row_bytes = w * channels * np.dtype(dtype).itemsize
    pixels = np.ascontiguousarray(rows[:, :row_bytes]).view(dtype)
    if msg.get("is_bigendian") and np.dtype(dtype).itemsize > 1:
        pixels = pixels.byteswap()
    img = pixels.reshape(h, w, channels) if channels > 1 else pixels.reshape(h, w)

    if encoding in DEPTH_ENCODINGS:
        return depth_to_metres(img, encoding), encoding
    if encoding.startswith("bgr"):
        img = img[..., [2, 1, 0]]
    elif encoding.startswith("rgba"):
        img = img[..., :3]
    return np.ascontiguousarray(img), encoding


def decode_compressed(msg: dict) -> tuple[np.ndarray, str]:
    """Decode a ``sensor_msgs/CompressedImage`` (JPEG/PNG). Returns (array, format)."""
    fmt = msg.get("format", "")
    if "compressedDepth" in fmt:
        return decode_compressed_depth(msg), fmt
    with PILImage.open(io.BytesIO(_bytes(msg["data"]))) as im:
        # The compressed plugin encodes color as BGR order in the format string
        # but JPEG/PNG data itself is decoded to RGB by PIL, so no swap needed.
        if im.mode in ("I;16", "I;16B", "I"):
            return np.asarray(im, dtype=np.uint16), fmt
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        return np.asarray(im), fmt


def decode_compressed_depth(msg: dict) -> np.ndarray:
    """Decode the ``compressedDepth`` plugin's output to float32 metres.

    Payload: struct ConfigHeader { int32 format; float depthQuantA, depthQuantB; }
    then a PNG. For 16UC1 the PNG holds millimetres. For 32FC1 it holds quantized
    inverse depth: depth = depthQuantA / (value - depthQuantB), with 0 = invalid.
    """
    raw = _bytes(msg["data"])
    fmt = msg.get("format", "")
    _, quant_a, quant_b = struct.unpack("<iff", raw[:12])
    with PILImage.open(io.BytesIO(raw[12:])) as im:
        values = np.asarray(im).astype(np.float32)
    if "32FC1" in fmt:
        out = np.zeros_like(values)
        valid = values > 0
        out[valid] = quant_a / (values[valid] - quant_b)
        return out
    return values / 1000.0


def decode(msg: dict, msg_type: str) -> tuple[np.ndarray, str]:
    if msg_type.endswith("CompressedImage"):
        return decode_compressed(msg)
    return decode_raw(msg)
