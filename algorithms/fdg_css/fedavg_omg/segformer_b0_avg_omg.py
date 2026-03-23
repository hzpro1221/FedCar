import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_Avg_OMG(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the SegFormer-B0 model for the FedOMG framework.

        Args:
            num_classes (int): Number of target semantic categories.
        """
        super(SegFormerB0_Avg_OMG, self).__init__()
        
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
            torch.Tensor: Bilinearly upsampled logits of shape [B, num_classes, H, W].
        """
        outputs = self.model(x)

        # SegFormer decoder naturally outputs logits at 1/4 of the input resolution
        # (e.g., [B, num_classes, 128, 128] for a 512x512 input)
        logits = outputs.logits  

        # Bi-linear up-sampling to restore the original input resolution [B, num_classes, 512, 512].
        # This aligns the output with ground truth mask dimensions for pixel-wise loss calculation.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits