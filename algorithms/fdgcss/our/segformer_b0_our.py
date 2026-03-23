import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_CovMatch(nn.Module):
    def __init__(self, num_classes):
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

    def _hook_fn(self, module, input_args):
        self._extracted_features = input_args[0]

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]

        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        return upsampled_logits

    def forward_features(self, x):
        outputs = self.model(x)
        logits = outputs.logits               # [B, C, H/4, W/4]
        features = self._extracted_features   # [B, 256, H/4, W/4]

        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return upsampled_logits, features