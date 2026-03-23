import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_Avg_GA(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the SegFormer-B0 model variant for FedAvg with Generalization Adjustment (GA).

        Args:
            num_classes (int): Number of target semantic classes.
        """
        super(SegFormerB0_Avg_GA, self).__init__()
        
        # Standard SegFormer-B0 configuration
        # Architectures parameters (widths, layers, heads) follow the B0 specification
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

    def forward(self, x):
        """
        Forward pass for semantic segmentation prediction.

        Args:
            x (torch.Tensor): Input image batch of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Upsampled logits of shape [B, num_classes, H, W].
        """
        outputs = self.model(x)

        # SegFormer decoder outputs logits at 1/4 of the input resolution (e.g., 128x128 for 512x512 input)
        logits = outputs.logits  

        # Bi-linear interpolation to upscale logits back to the original input spatial dimensions.
        # This ensures the output resolution matches the ground truth mask for loss computation.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits