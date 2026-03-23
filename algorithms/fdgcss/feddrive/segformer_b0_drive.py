import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_Drive(nn.Module):
    """
    SegFormer-B0 wrapper tailored for the FedDrive semantic segmentation task.
    This implementation uses a lightweight MLP-based decoder and a Mix Transformer encoder.
    """
    def __init__(self, num_classes):
        super(SegFormerB0_Drive, self).__init__()
        
        # SegFormer-B0 Configuration
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
        Forward pass for the segmentation model.
        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W].
        Returns:
            upsampled_logits (torch.Tensor): Logits upsampled to original [H, W].
        """
        outputs = self.model(x)

        # Segformer's decoder output resolution is typically 1/4 of the input.
        # Shape: [B, num_classes, H/4, W/4]
        logits = outputs.logits  

        # Upsampling: Segformer deliberately outputs at 1/4 resolution to maintain 
        # efficiency. We use bilinear interpolation to match the ground truth size.
        # Target shape: [B, num_classes, H, W]
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits