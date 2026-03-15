import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerB0_SR(nn.Module):
    def __init__(
        self, 
        num_classes,
        z_dim=128
    ):
        super(SegFormerB0_SR, self).__init__()
        
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
        self.z_dim = z_dim
        self.model = SegformerForSemanticSegmentation(self.config)

        self.model.decode_head.classifier = nn.Identity()

        self.to_z_params = nn.Conv2d(
            self.config.decoder_hidden_size, 
            z_dim * 2, 
            kernel_size=1
        )

        self.fc_class = nn.Conv2d(z_dim, num_classes, kernel_size=1)        

    def forward(self, x, return_dist=False):
        outputs = self.model(x)
        features = outputs.logits 
        
        # split mu và sigma
        z_params = self.to_z_params(features)
        z_mu = z_params[:, :self.z_dim, :, :]
        z_sigma = F.softplus(z_params[:, self.z_dim:, :, :]) 

        if self.training and return_dist:
            # z = mu + sigma * epsilon
            eps = torch.randn_like(z_sigma)
            z = z_mu + z_sigma * eps
        else:
            # z = mu, this is for inference
            z = z_mu 

        logits = self.fc_class(z)

        # Upsampling
        upsampled_logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        if return_dist:
            return upsampled_logits, z, z_mu, z_sigma

        return upsampled_logits
