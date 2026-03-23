import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_CovMatch(nn.Module):
    """
    SegFormer-B0 wrapper tailored for the FedCovMatch algorithm.
    Utilizes a forward pre-hook to extract deep features from the decoder 
    just before the final classification layer, which are essential for 
    computing covariance alignment across domains.
    """
    def __init__(self, num_classes: int):
        super(SegFormerB0_CovMatch, self).__init__()
        
        self.config = SegformerConfig(
            num_labels=num_classes,
            widths=[32, 64, 160, 256],
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_dropout_prob=0.1,
            decoder_hidden_size=256, 
            semantic_loss_ignore_index=255
        )

        self.model = SegformerForSemanticSegmentation(self.config)
        
        self._extracted_features = None
        self.model.decode_head.classifier.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input_args: tuple):
        """
        Callback function triggered automatically during the forward pass.
        Intercepts the forward pass right before the final Conv2d layer.
        `input_args` is a tuple where the 0th element is the feature map.
        """
        self._extracted_features = input_args[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for inference or evaluation.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            
        Returns:
            upsampled_logits (torch.Tensor): Predictions of shape [B, num_classes, H, W].
        """
        outputs = self.model(x)
        logits = outputs.logits  # Shape: [B, num_classes, H/4, W/4]

        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        return upsampled_logits

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both the segmentation logits and the deep features.
        Used explicitly during FedCovMatch training to calculate the Covariance Loss.
        
        Returns:
            upsampled_logits: Predictions of shape [B, num_classes, H, W]
            features: Extracted feature maps of shape [B, 256, H/4, W/4]
        """
        outputs = self.model(x)
        logits = outputs.logits               # [B, num_classes, H/4, W/4]
        features = self._extracted_features   # [B, 256, H/4, W/4] extracted via hook

        # Upsample to match original image resolution
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return upsampled_logits, features