import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_DG(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the SegFormer-B0 architecture for Federated Domain Generalization (FedDG).

        Args:
            num_classes (int): Number of semantic categories for the segmentation task.
        """
        super(SegFormerB0_DG, self).__init__()
        
        # Widths, layers, and heads follow the B0 standard to balance efficiency and performance.
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
        Forward pass of the SegFormer-B0 model.

        Args:
            x (torch.Tensor): Input image batch of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Prediction logits upsampled to the original input resolution [B, C, H, W].
        """
        outputs = self.model(x)

        # Segformer's MLP decoder outputs logits at 1/4 of the original input resolution.
        # For a 512x512 input, the raw logits will be [B, num_classes, 128, 128].
        logits = outputs.logits  

        # Bi-linear interpolation to upscale the logits back to the original spatial dimensions.
        # This step is required to align predictions with the ground truth masks during training/evaluation.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits