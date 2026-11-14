"""Convert between ``sensor_msgs`` image messages and numpy arrays, without cv_bridge."""

from __future__ import annotations

import io

import numpy as np
from builtin_interfaces.msg import Time
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage, Image

# ROS encoding -> (numpy dtype, channels)
ENCODINGS: dict[str, tuple[type[np.generic], int]] = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "8UC3": (np.uint8, 3),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def image_to_numpy(msg: Image, to_rgb: bool = True) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` into an array view of its pixels.

    Parameters
    ----------
    msg : sensor_msgs.msg.Image
        Raw image. Rows may be padded; ``msg.step`` (bytes per row) is honoured.
    to_rgb : bool, optional
        Reorder ``bgr8``/``bgra8`` channels to RGB(A). Default True.

    Returns
    -------
    np.ndarray
        ``(H, W, C)`` for multi-channel encodings, ``(H, W)`` otherwise, with the
        dtype from :data:`ENCODINGS` in native byte order. 16-bit depth stays in mm.

    Raises
    ------
    ValueError
        If ``msg.encoding`` is not in :data:`ENCODINGS`.
    """
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    dtype, ch = ENCODINGS[msg.encoding]
    dt: np.dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    buf: np.ndarray = np.frombuffer(msg.data, dtype=np.uint8)
    rows: np.ndarray = buf[: msg.height * msg.step].reshape(msg.height, msg.step)
    pixels: np.ndarray = rows[:, : msg.width * ch * dt.itemsize].copy().view(dt)
    arr: np.ndarray = pixels.reshape(msg.height, msg.width, ch) if ch > 1 else pixels.reshape(
        msg.height, msg.width)
    arr = arr.astype(dt.newbyteorder("="), copy=False)
    if to_rgb and msg.encoding in ("bgr8", "bgra8"):
        arr = arr[..., [2, 1, 0]] if ch == 3 else arr[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(arr)


def numpy_to_image(arr: np.ndarray, encoding: str, stamp: Time | None = None,
                   frame_id: str = "") -> Image:
    """Pack a numpy array into a ``sensor_msgs/Image``.

    Parameters
    ----------
    arr : np.ndarray
        ``(H, W)`` or ``(H, W, C)`` array whose dtype and channel count match
        ``encoding`` (e.g. ``uint8 (H, W, 3)`` for ``rgb8``).
    encoding : str
        ROS encoding string; must be in :data:`ENCODINGS`.
    stamp : builtin_interfaces.msg.Time, optional
        Header stamp. Zero if omitted.
    frame_id : str, optional
        Header frame id.

    Returns
    -------
    sensor_msgs.msg.Image
        Little-endian, unpadded rows: ``step = W * C * itemsize``.
    """
    dtype, ch = ENCODINGS[encoding]
    a: np.ndarray = np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<"))
    got_ch = a.shape[2] if a.ndim == 3 else 1
    if got_ch != ch:
        raise ValueError(f"{encoding} needs {ch} channel(s), array has {got_ch}")
    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(a.shape[0]), int(a.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * ch * a.itemsize
    msg.data = a.tobytes()
    return msg


def compressed_to_image(msg: CompressedImage) -> Image:
    """Decode a JPEG/PNG ``sensor_msgs/CompressedImage`` into a raw ``Image``.

    Parameters
    ----------
    msg : sensor_msgs.msg.CompressedImage
        Output of the ``compressed`` image_transport plugin (not ``compressedDepth``).

    Returns
    -------
    sensor_msgs.msg.Image
        ``rgb8`` for color, ``mono8`` for 8-bit gray, ``16UC1`` for 16-bit PNG.
        Header is copied from ``msg``.

    Notes
    -----
    The ``compressed`` plugin's ``format`` string says ``bgr8; jpeg compressed bgr8``
    for color, but the JPEG itself decodes to RGB, so no channel swap is needed.
    """
    with PILImage.open(io.BytesIO(bytes(msg.data))) as im:
        if im.mode in ("I;16", "I;16B", "I"):
            arr: np.ndarray = np.asarray(im, dtype=np.uint16)
            encoding = "16UC1"
        elif im.mode == "L":
            arr = np.asarray(im)
            encoding = "mono8"
        else:
            arr = np.asarray(im.convert("RGB"))
            encoding = "rgb8"
    out = numpy_to_image(arr, encoding, frame_id=msg.header.frame_id)
    out.header.stamp = msg.header.stamp
    return out


def stamp_to_sec(stamp: Time) -> float:
    """``builtin_interfaces/Time`` to float seconds: ``sec + nanosec * 1e-9``."""
    return stamp.sec + stamp.nanosec * 1e-9
