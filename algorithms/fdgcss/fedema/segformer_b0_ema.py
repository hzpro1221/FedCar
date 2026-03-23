import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_EMA(nn.Module):
    """
    SegFormer-B0 wrapper for Semantic Segmentation.
    This version is tailored for FedEMA, utilizing a lightweight MLP-based decoder.
    """
    def __init__(self, num_classes):
        super(SegFormerB0_EMA, self).__init__()
        
        # Define SegFormer-B0 Configuration
        # B0 is the most lightweight variant, ideal for Federated Learning on edge devices.
        self.config = SegformerConfig(
            num_labels=num_classes,
            widths=[32, 64, 160, 256],      # Feature dimensions at each of the 4 stages
            num_layers=[2, 2, 2, 2],        # Number of transformer blocks per stage
            num_heads=[1, 2, 5, 8],         # Multi-head attention settings
            mlp_ratios=[4, 4, 4, 4],        # Expansion ratio in MLP layers
            hidden_dropout_prob=0.1,
            decoder_hidden_size=256,        # Embedding dimension for the MLP decoder head
            semantic_loss_ignore_index=255  # Typically used for background/void in Cityscapes/BDD
        )

        # Initialize the model with the specified B0 configuration
        self.model = SegformerForSemanticSegmentation(self.config)

    def forward(self, x):
        """
        Forward pass for semantic segmentation.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            
        Returns:
            upsampled_logits (torch.Tensor): Pixel-wise class predictions [B, num_classes, H, W].
        """
        outputs = self.model(x)

        # SegFormer's decoder output is 1/4 of the original input resolution (e.g., 128x128 for a 512x512 input).
        # This is a design choice by the authors to keep the MLP decoder efficient.
        logits = outputs.logits  # Shape: [B, num_classes, H/4, W/4]

        # Bilinear interpolation to bring logits back to the original image resolution.
        # This allows for direct loss calculation against full-resolution ground truth masks.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:],        # Target size (H, W) from the input image
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits