"""
Utilities and services for image object detection
"""
import numpy as np
import numpy.typing as npt
import torch
import ultralytics
from ultralytics import YOLO
import cv2


class ObjectDetection:

    def __init__(self, verbose_logging: bool = False) -> None:
        self._model = YOLO("yolo26n.pt")
        self._verbose_logging = verbose_logging

    def apply_bounding_boxes(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # Ultralytics treats numpy input as BGR (OpenCV order); the viewer sends RGB.
        bgr = np.ascontiguousarray(image[..., ::-1])
        results = self._model(bgr, verbose=self._verbose_logging)

        if len(results) == 0:
            return image

        canvas = image.copy()

        for result in results:
            # boxes.xyxy: tensor of shape (NUM_BOXES, 4)
            boxes_xyxy = result.boxes.xyxy
            for box in boxes_xyxy:
                # box is a tensor with shape [4]
                # First pair is top-left (x,y)
                # Second pair is bottom right (x,y)
                # Round to int32 pixels: uint8 would wrap coordinates above 255.
                box = box.round().to(torch.int32)
                top_left = tuple(box[:2].tolist())
                bottom_right = tuple(box[2:].tolist())
                cv2.rectangle(canvas, top_left, bottom_right, color=(0, 255, 0), thickness=3)

        return canvas