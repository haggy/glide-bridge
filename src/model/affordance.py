from dataclasses import dataclass, field

import torch
from ultralytics import YOLO


@dataclass
class HeadOutput:
    boxes: torch.Tensor  # (B, )
    scores: torch.Tensor
    classes: torch.Tensor
    # Pyramid feature network layer outputs
    # p3: (B, C, H/8, W/8)
    # p4: (B, C, H/16, W/16)
    # p5: (B, C, H/32, W/32)
    p3: torch.Tensor
    p4: torch.Tensor
    p5: torch.Tensor


    @classmethod
    def create(cls, 
               y: torch.Tensor, 
               pyramid_feature_output: dict[str, list[torch.Tensor]], 
               score_threshold: float = 0.25
    ) -> "HeadOutput":
        """
        Pyramid output dict is expected to have this structure:

        {
            "one2many": {
                "boxes":  torch.Tensor,  # shape: (B, 4, 8400) — flattened & concatenated across P3, P4, P5
                "scores": torch.Tensor,  # shape: (B, nc, 8400) — flattened & concatenated across P3, P4, P5
                "feats":  [              # list of 3 input neck feature maps [P3, P4, P5]
                    torch.Tensor,        # P3 input shape: (B, ch[0], 80, 80)
                    torch.Tensor,        # P4 input shape: (B, ch[1], 40, 40)
                    torch.Tensor,        # P5 input shape: (B, ch[2], 20, 20)
                ],
            },
            "one2one": {
                "boxes":  torch.Tensor,  # shape: (B, 4, 8400)
                "scores": torch.Tensor,  # shape: (B, nc, 8400)
                "feats":  [              # list of 3 input neck feature maps [P3, P4, P5]
                    torch.Tensor,        # P3 input shape: (B, ch[0], 80, 80)
                    torch.Tensor,        # P4 input shape: (B, ch[1], 40, 40)
                    torch.Tensor,        # P5 input shape: (B, ch[2], 20, 20)
                ],
            },
        }
        """
        # y shape: (B, 300, 6)
        # 300: is the max number of detections
        # size 6 is the data: first 4 is box, 5 is the score, 6 is the class
        scores = y[..., 4]
        score_filter = scores > score_threshold

        y_thresh = y[score_filter]
        p_feat_thresh = pyramid_feature_output["one2one"]["feats"]

        boxes = y_thresh[..., :4]
        classes = y_thresh[..., 5].long()
        p3, p4, p5 = p_feat_thresh

        return cls(
            boxes=boxes,
            scores=scores,
            classes=classes,
            p3=p3,
            p4=p4,
            p5=p5,
        )

    def _get_pyramid_level(self, level: int) -> torch.Tensor:
        if level < 3 or level > 5:
            raise ValueError(f"Level most be one of: 3, 4, 5 -- got {level}")
        
        outputs = self.one2one or self.one2many

        if outputs is None:
            raise ValueError("No head outputs have been set")

        return outputs["feats"][level - 3]

    def get_p3(self) -> torch.Tensor:
        return self._get_pyramid_level(3)

    def get_p4(self) -> torch.Tensor:
        return self._get_pyramid_level(4)

    def get_p5(self) -> torch.Tensor:
        return self._get_pyramid_level(5)


class AffordGlide(torch.nn.Module):
    """
    The Glide affordance model

    Uses a YOLO backbone to produce feature maps via
    Feature Pyramid Networks (FPN). 
    """
    def __init__(self):
        super().__init__()

        # Lock into eval mode. Frozen and used only for feature extraction
        self._feature_extractor = YOLO("yolo26n.pt").model.eval()

        # Enabling end to end results in roughly 1 prediction per observation
        # i.e. Each detection keeps only its winning class
        self._feature_extractor.model[-1].end2end = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        detect_head = self._feature_extractor.model[-1]
        print("Pyramid strides (P3, P4, P5):", detect_head.stride)  # Usually tensor([ 8., 16., 32.])
        print("Input layers from neck:", detect_head.f)
        
        # Run a dummy forward pass
        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            y, preds = self._feature_extractor(x)

        ho = HeadOutput.create(y, preds)

        print(f"HO P5 shape: {ho.p5.shape}")
        print(f"HO P5 channel 0: {ho.p5[:, 0, ::].shape}")


        return torch.zeros((B, H, W), dtype=torch.float32)