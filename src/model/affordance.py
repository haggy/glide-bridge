from dataclasses import dataclass, field

import torch
from ultralytics import YOLO


@dataclass
class HeadOutput:
    one2one: list[torch.Tensor] | None = None
    one2many: list[torch.Tensor] | None = None

    @classmethod
    def create(cls, pyramid_feature_output: dict[str, list[torch.Tensor]]) -> "HeadOutput":
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
        return cls(
            one2one=pyramid_feature_output.get("one2one"),
            one2many=pyramid_feature_output.get("one2many"),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        pyramid_features = {}
        
        def hook_fn(module, inputs, outputs):
            # inputs[0] is a list of 3 tensors corresponding to [P3, P4, P5] from the neck
            pyramid_features["neck_inputs"] = [f.detach() for f in inputs[0]]
            pyramid_features["head_outputs"] = outputs


        detect_head = self._feature_extractor.model[-1]
        print("Pyramid strides (P3, P4, P5):", detect_head.stride)  # Usually tensor([ 8., 16., 32.])
        print("Input layers from neck:", detect_head.f)
        handle = detect_head.register_forward_hook(hook_fn)
        
        # Run a dummy forward pass
        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            _ = self._feature_extractor(x)
        
        for i, feat in enumerate(pyramid_features["neck_inputs"]):
            print(f"P{i+3} input shape: {feat.shape}")  # Strides 8, 16, 32
        

        ho = HeadOutput.create(pyramid_features["head_outputs"])
        print(f"HO P5 shape: {ho.get_p5().shape}")
        print(f"HO P5 channel 0: {ho.get_p5()[:, 0, ::].shape}")
        
        handle.remove()

        return torch.zeros((B, H, W), dtype=torch.float32)