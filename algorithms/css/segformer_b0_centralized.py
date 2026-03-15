import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_Centralized(nn.Module):
    def __init__(self, num_classes):
        super(SegFormerB0_Avg, self).__init__()
        
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
        outputs = self.model(x)

        # [B, num_classes, 128, 128]
        logits = outputs.logits  

        # Up-sampling to [B, num_classes, 512, 512] (because label is this shape).
        # Seem like author of this paper deliberately shrink the dimension to H/4 :vv..
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits
