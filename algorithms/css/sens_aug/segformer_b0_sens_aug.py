import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_SensAug(nn.Module):
    def __init__(self, num_classes):
        """
        Implementation of the SegFormer-B0 architecture for Sensitivity-Aware Augmentation (SensAug).
        Utilizes a hierarchical Mix Transformer (MiT) backbone and a lightweight All-MLP decoder.
        
        Args:
            num_classes (int): Number of semantic target categories.
        """
        super(SegFormerB0_SensAug, self).__init__()
        
        # SegFormer-B0 configuration
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
        Forward pass with Bilinear Interpolation to restore spatial resolution.
        
        Args:
            x (torch.Tensor): Input batch of shape [B, 3, H, W].
            
        Returns:
            upsampled_logits (torch.Tensor): Output logits upscaled to the original 
                                             input resolution [B, num_classes, H, W].
        """
        outputs = self.model(x)

        logits = outputs.logits  

        # Bi-linear interpolation is applied to upscale the predictions back to the 
        # original input dimensions. This ensures alignment with the ground truth 
        # mask for pixel-wise loss calculation and evaluation.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits