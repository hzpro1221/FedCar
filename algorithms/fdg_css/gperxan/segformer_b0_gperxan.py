import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class XON_LayerNorm(nn.Module):
    """
    Cross-domain Adaptive Normalization (XON) layer.
    
    This layer dynamically blends Layer Normalization (LN) and Instance Normalization (IN).
    LN captures global structural information, while IN filters out domain-specific 
    style variations. The blending weights are learnable parameters.
    
    The output is computed as:
    $$y = \text{softmax}(w_{ln}, w_{in})_0 \cdot LN(x) + \text{softmax}(w_{ln}, w_{in})_1 \cdot IN(x)$$
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps)

        # InstanceNorm1d expects the number of channels (C)
        channels = normalized_shape[0] if isinstance(normalized_shape, (tuple, list)) else normalized_shape
        self.instancenorm = nn.InstanceNorm1d(channels, affine=True)
        
        # Learnable scores for the dynamic blending mechanism
        self.score_ln = nn.Parameter(torch.tensor([1.0]))
        self.score_in = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Supports 3D [B, N, C] or 4D [B, C, H, W].
        """
        out_ln = self.layernorm(x)

        orig_shape = x.shape
        # InstanceNorm1d requires [B, C, L]. Handle 4D [B, C, H, W] or 3D [B, L, C]
        if len(orig_shape) == 4:
            # Flatten spatial dims: [B, C, H, W] -> [B, C, H*W]
            x_in = x.reshape(orig_shape[0], orig_shape[1], -1) 
        else:
            # Transpose sequence dim: [B, N, C] -> [B, C, N]
            x_in = x.transpose(1, 2)

        out_in = self.instancenorm(x_in)

        # Reshape back to original input format
        if len(orig_shape) == 4:
            out_in = out_in.reshape(orig_shape)
        else:
            out_in = out_in.transpose(1, 2)

        # Compute dynamic blending weights via Softmax
        scores = F.softmax(torch.stack([self.score_ln, self.score_in]), dim=0)
        
        return scores[0] * out_ln + scores[1] * out_in


def replace_layernorm_with_xon(module):
    """
    Recursively replaces all nn.LayerNorm layers in a model with XON_LayerNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, XON_LayerNorm(child.normalized_shape, eps=child.eps))
        else:
            replace_layernorm_with_xon(child)


class SegFormerB0_gPerXAN(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes SegFormer-B0 with Generalized Personalized Cross-domain 
        Adaptive Normalization (gPerXAN).
        
        Args:
            num_classes (int): Number of target semantic categories.
        """
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
        
        # Inject XON layers into the SegFormer backbone and decoder
        replace_layernorm_with_xon(self.model)

    def forward(self, x, return_features=False):
        """
        Forward pass for semantic segmentation.
        
        Args:
            x (torch.Tensor): Input batch [B, 3, H, W].
            return_features (bool): If True, returns hidden states for regularization.
            
        Returns:
            upsampled_logits (torch.Tensor): Prediction map at input resolution.
            features (list, optional): Multi-scale hidden states if return_features is True.
        """
        # Extract features from the hierarchical Transformer encoder
        outputs = self.model.segformer(x, output_hidden_states=True)
        features = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs[0]        
        
        # Decode multi-scale features into class logits
        logits = self.model.decode_head(features)

        # Bi-linear up-sampling to recover original spatial resolution
        upsampled_logits = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        if return_features:
            return upsampled_logits, features
        return upsampled_logits