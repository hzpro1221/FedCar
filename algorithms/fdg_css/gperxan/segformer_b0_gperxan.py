import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class XON_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps)

        channels = normalized_shape[0] if isinstance(normalized_shape, (tuple, list)) else normalized_shape
        self.instancenorm = nn.InstanceNorm1d(channels, affine=True)
        
        self.score_ln = nn.Parameter(torch.tensor([1.0]))
        self.score_in = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        out_ln = self.layernorm(x)

        orig_shape = x.shape
        if len(orig_shape) == 4:
            x_in = x.reshape(orig_shape[0], orig_shape[1], -1) 
        else:
            x_in = x.transpose(1, 2)

        out_in = self.instancenorm(x_in)

        if len(orig_shape) == 4:
            out_in = out_in.reshape(orig_shape)
        else:
            out_in = out_in.transpose(1, 2)

        scores = F.softmax(torch.stack([self.score_ln, self.score_in]), dim=0)
        
        return scores[0] * out_ln + scores[1] * out_in


def replace_layernorm_with_xon(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, XON_LayerNorm(child.normalized_shape, eps=child.eps))
        else:
            replace_layernorm_with_xon(child)


class SegFormerB0_gPerXAN(nn.Module):
    def __init__(self, num_classes):
        super(SegFormerB0_gPerXAN, self).__init__()
        
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
        
        replace_layernorm_with_xon(self.model)

    def forward(self, x, return_features=False):
        outputs = self.model.segformer(x, output_hidden_states=True)
        features = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs[0]        
        
        logits = self.model.decode_head(features)

        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        if return_features:
            return upsampled_logits, features
        return upsampled_logits