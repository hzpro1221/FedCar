import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_Avg(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the SegFormer-B0 model for Federated Learning.

        Args:
            num_classes (int): Number of semantic categories for classification.
        """
        super(SegFormerB0_Avg, self).__init__()
        
        # Configure SegFormer-B0 parameters
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
        Forward pass for semantic segmentation.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Logits upsampled to the original input resolution [B, C, H, W].
        """
        outputs = self.model(x)

        # SegFormer decoder outputs logits at H/4, W/4 resolution
        # For SegFormer-B0, this results in [B, num_classes, 128, 128] if input is 512x512
        logits = outputs.logits  

        # Bi-linear up-sampling back to original input resolution [B, num_classes, 512, 512]
        # This aligns predictions with ground truth masks for loss calculation
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits