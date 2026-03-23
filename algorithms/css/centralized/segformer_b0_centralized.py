import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_Centralized(nn.Module):
    def __init__(self, num_classes):
        """
        Implementation of the SegFormer-B0 architecture for Centralized Semantic Segmentation.
        This model utilizes a hierarchical Mix Transformer (MiT) backbone and a lightweight 
        All-MLP decoder.
        
        Args:
            num_classes (int): Number of target semantic categories for the segmentation task.
        """
        super(SegFormerB0_Centralized, self).__init__()
        
        # Configure SegFormer-B0 parameterss
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
        Forward pass for the SegFormer model.
        
        Args:
            x (torch.Tensor): Input image batch with shape [B, 3, H, W].
            
        Returns:
            upsampled_logits (torch.Tensor): Prediction logits bilinearly interpolated 
                                             to match the original input resolution [B, num_classes, H, W].
        """
        outputs = self.model(x)

        # SegFormer's MLP decoder produces logits at 1/4 of the input spatial resolution 
        # (e.g., [B, num_classes, 128, 128] for a 512x512 input). 
        logits = outputs.logits  

        # Bi-linear interpolation is applied to restore the spatial resolution to the original input size.
        # This alignment is crucial for pixel-wise loss computation during training and mIoU evaluation.
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits