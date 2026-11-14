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
from typing import Any

import numpy as np
from PIL import Image as PILImage

# encoding -> (numpy dtype, channels)
_RAW_LAYOUTS: dict[str, tuple[type[np.generic], int]] = {
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


def _bytes(data: str | bytes | bytearray | list[int]) -> bytes:
    """rosbridge gives base64 text for uint8[]; accept raw bytes or int lists too."""
    if isinstance(data, str):
        return base64.b64decode(data)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return bytes(bytearray(data))


def depth_to_metres(img: np.ndarray, encoding: str) -> np.ndarray:
    """Convert a depth image to float32 metres.

    Parameters
    ----------
    img : np.ndarray
        (H, W) uint16 millimetres (16UC1/mono16) or float32 metres (32FC1).
    encoding : str
        ROS image encoding of ``img``.

    Returns
    -------
    np.ndarray
        (H, W) float32 metres, 0 = no reading.

    Notes
    -----
    ``depth_m = raw_mm / 1000`` for 16-bit (ROS convention). For 32FC1, non-finite
    values (NaN/inf = no return) are set to 0.
    """
    if encoding == "32FC1":
        out = img.astype(np.float32, copy=True)
        out[~np.isfinite(out)] = 0.0
        return out
    return img.astype(np.float32) / 1000.0


def decode_raw(msg: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Decode a rosbridge ``sensor_msgs/Image`` dict.

    Parameters
    ----------
    msg : dict
        rosbridge JSON message (``data`` as base64).

    Returns
    -------
    image : np.ndarray
        RGB uint8 (H, W, 3), mono (H, W), or depth float32 metres (H, W).
    encoding : str
        The message's encoding.

    Raises
    ------
    ValueError
        If the encoding is not supported.
    """
    encoding = msg["encoding"]
    if encoding not in _RAW_LAYOUTS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    dtype, channels = _RAW_LAYOUTS[encoding]
    h, w, step = msg["height"], msg["width"], msg["step"]
    buf: np.ndarray = np.frombuffer(_bytes(msg["data"]), dtype=np.uint8)
    rows: np.ndarray = buf[: h * step].reshape(h, step)   # (H, step) bytes, rows may be padded
    row_bytes = w * channels * np.dtype(dtype).itemsize
    pixels: np.ndarray = np.ascontiguousarray(rows[:, :row_bytes]).view(dtype)
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


def decode_compressed(msg: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Decode a ``sensor_msgs/CompressedImage`` (JPEG/PNG, or compressedDepth).

    Parameters
    ----------
    msg : dict
        Message with ``format`` and ``data`` (base64 text or raw bytes).

    Returns
    -------
    image : np.ndarray
        RGB uint8 (H, W, 3), mono uint8 (H, W), uint16 (H, W) for 16-bit PNG, or
        float32 metres for compressedDepth.
    format : str
        The message's format string.
    """
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


def decode_compressed_depth(msg: dict[str, Any]) -> np.ndarray:
    """Decode the ``compressedDepth`` plugin's output to float32 metres.

    Parameters
    ----------
    msg : dict
        Message with ``format`` and ``data``.

    Returns
    -------
    np.ndarray
        (H, W) float32 metres, 0 = invalid.

    Notes
    -----
    Payload: ``struct ConfigHeader { int32 format; float depthQuantA, depthQuantB; }``
    then a PNG. For 16UC1 the PNG holds millimetres, ``depth = value / 1000``. For
    32FC1 it holds quantized inverse depth,
    ``depth = depthQuantA / (value - depthQuantB)``, with value 0 = invalid.
    """
    raw = _bytes(msg["data"])
    fmt = msg.get("format", "")
    _, quant_a, quant_b = struct.unpack("<iff", raw[:12])
    with PILImage.open(io.BytesIO(raw[12:])) as im:
        values: np.ndarray = np.asarray(im).astype(np.float32)   # (H, W)
    if "32FC1" in fmt:
        out = np.zeros_like(values)
        valid = values > 0
        out[valid] = quant_a / (values[valid] - quant_b)
        return out
    return values / 1000.0


def decode(msg: dict[str, Any], msg_type: str) -> tuple[np.ndarray, str]:
    """Dispatch on ``msg_type``: CompressedImage -> :func:`decode_compressed`, else
    :func:`decode_raw`. Returns (array, encoding or format)."""
    if msg_type.endswith("CompressedImage"):
        return decode_compressed(msg)
    return decode_raw(msg)
