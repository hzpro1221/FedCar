import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

def calc_mu_sig(x, eps=1e-6):
    mu = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    sig = torch.sqrt(var + eps)
    return mu, sig

class StyleRepresentation(nn.Module):
    def __init__(self, num_prototype=2, channel_size=64):
        super().__init__()
        self.num_prototype = num_prototype
        self.channel_size = channel_size
        
        self.style_mu = nn.Parameter(torch.zeros(num_prototype, channel_size))
        self.style_sig = nn.Parameter(torch.zeros(num_prototype, channel_size))

    def was_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)
        
        distance = (cur_mu - proto_mu).pow(2) + cur_sig.pow(2) + \
                   proto_sig.pow(2) - 2 * cur_sig * proto_sig
        return distance

    def forward(self, fea):
        batch = fea.size(0)
        cur_mu, cur_sig = calc_mu_sig(fea)

        distance = self.was_distance(cur_mu, cur_sig, self.style_mu, self.style_sig, batch)
        distance = distance.mean(dim=2) 

        alpha = 1.0 / (1.0 + distance)
        alpha = F.softmax(alpha, dim=1) # [B, num_prototype]

        # Mix styles based on alpha
        mixed_mu = torch.matmul(alpha, self.style_mu).view(batch, self.channel_size, 1, 1)
        mixed_sig = torch.matmul(alpha, self.style_sig).view(batch, self.channel_size, 1, 1)

        # normalize & projecting
        fea = ((fea - cur_mu) / cur_sig) * mixed_sig + mixed_mu
        return fea

class SegFormerB0_SPC_Net(nn.Module):
    def __init__(self, num_classes=19, num_datasets=2):
        super(SegFormerB0_SPC_Net, self).__init__()
        self.num_classes = num_classes
        self.num_datasets = num_datasets
        self.feature_dim = 256 

        self.config = SegformerConfig(
            num_labels=num_classes,
            widths=[32, 64, 160, 256],
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_dropout_prob=0.1,
            decoder_hidden_size=256,
        )
        self.segformer = SegformerModel(self.config)

        # 2. Style Representation (Scale 1/4 & 1/8)
        self.style_adain1 = StyleRepresentation(num_prototype=num_datasets, channel_size=32)
        self.style_adain2 = StyleRepresentation(num_prototype=num_datasets, channel_size=64)

        # 3. All-MLP Decoder (Replace for ASPP)
        self.linear_c4 = nn.Conv2d(256, self.feature_dim, 1)
        self.linear_c3 = nn.Conv2d(160, self.feature_dim, 1)
        self.linear_c2 = nn.Conv2d(64, self.feature_dim, 1)
        self.linear_c1 = nn.Conv2d(32, self.feature_dim, 1)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(self.feature_dim * 4, self.feature_dim, 1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True)
        )

        # 4. Projected Clustering 
        self.prototypes = nn.Parameter(torch.zeros(num_classes, num_datasets, self.feature_dim))
        self.feat_norm = nn.LayerNorm(self.feature_dim, eps=1e-5)
        self.mask_norm = nn.LayerNorm(num_classes, eps=1e-5)

    def forward(self, x):
        size = x.shape[2:] 
        
        outputs = self.segformer(x, output_hidden_states=True)
        features = list(outputs.hidden_states)

        features[0] = self.style_adain1(features[0])
        features[1] = self.style_adain2(features[1])

        c1, c2, c3, c4 = features

        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1)

        _fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # [B, 256, H/4, W/4]

        B, C, H, W = _fused.shape
        fea_out = _fused.permute(0, 2, 3, 1).reshape(-1, C) # Flatten H, W to N = B*H*W
        fea_out = self.feat_norm(fea_out)
        fea_out = F.normalize(fea_out, p=2, dim=1)

        norm_protos = F.normalize(self.prototypes, p=2, dim=2)

        # n: pixels, d: feature_dim, k: classes, m: datasets
        masks = torch.einsum('nd,kmd->nmk', fea_out, norm_protos) # [N, M, K]
        
        masks_max, _ = torch.max(masks, dim=1) # [N, K]
        main_out = self.mask_norm(masks_max)

        main_out = main_out.view(B, H, W, self.num_classes).permute(0, 3, 1, 2) # [B, 19, H/4, W/4]
        
        upsampled_logits = F.interpolate(main_out, size=size, mode='bilinear', align_corners=False)

        return upsampled_logits