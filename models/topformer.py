import math
import torch
from torch import nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class ConvModule(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        bias = norm_cfg is None
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if norm_cfg is not None:
            self.add_module('bn', nn.BatchNorm2d(out_channels))
        if act_cfg is not None:
            if act_cfg.get('type') == 'ReLU':
                self.add_module('act', nn.ReLU(inplace=True))
            elif act_cfg.get('type') == 'ReLU6':
                self.add_module('act', nn.ReLU6(inplace=True))
            else:
                self.add_module('act', act_cfg.get('type')())

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, norm_cfg=None):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, ks: int, stride: int, expand_ratio: int, activations=None, norm_cfg=dict(type='BN', requires_grad=True)):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class TokenPyramidModule(nn.Module):
    def __init__(self, cfgs, out_indices, inp_channel=16, activation=nn.ReLU, norm_cfg=dict(type='BN', requires_grad=True), width_mult=1.):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
            activation()
        )
        self.cfgs = cfgs
        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg, activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4, activation=None, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__() 
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads 
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, self.nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, self.nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):
        B, C, H, W = get_shape(x)
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)
        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

class Block(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0., norm_cfg=dict(type='BN2d', requires_grad=True), act_layer=None):
        super().__init__()
        self.block_num = block_num
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg, act_layer=act_layer))

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class InjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, norm_cfg=dict(type='BN', requires_grad=True), activations=None):
        super(InjectionMultiSum, self).__init__()
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        return local_feat * sig_act + global_feat

SIM_BLOCK = {
    "muli_sum": InjectionMultiSum,
}

class Topformer(nn.Module):
    def __init__(self, 
                 cfgs=[
                     [3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1], 
                     [5, 3, 64, 2], [5, 3, 64, 1], [3, 3, 128, 2], 
                     [3, 3, 128, 1], [5, 6, 160, 2], [5, 6, 160, 1], [3, 6, 160, 1]
                 ],
                 channels=[32, 64, 128, 160], 
                 out_channels=[256, 256, 256, 256], 
                 embed_out_indice=[2, 4, 6, 9], 
                 decode_out_indices=[1, 2, 3],
                 depths=4, key_dim=16, num_heads=8, attn_ratios=2, mlp_ratios=2, c2t_stride=2, drop_path_rate=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True), act_layer=nn.ReLU6, injection_type="muli_sum",
                 injection=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices

        self.tpm = TokenPyramidModule(cfgs=cfgs, out_indices=embed_out_indice, norm_cfg=norm_cfg)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.trans = BasicLayer(
            block_num=depths, embedding_dim=self.embed_dim, key_dim=key_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratios, attn_ratio=attn_ratios, drop=0, attn_drop=0, drop_path=dpr,
            norm_cfg=norm_cfg, act_layer=act_layer)
        
        self.SIM = nn.ModuleList()
        inj_module = SIM_BLOCK[injection_type]
        if self.injection:
            for i in range(len(channels)):
                if i in decode_out_indices:
                    self.SIM.append(inj_module(channels[i], out_channels[i], norm_cfg=norm_cfg, activations=act_layer))
                else:
                    self.SIM.append(nn.Identity())

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (n if n > 0 else 1)))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        checkpoint_path = "models/topformer-B-224-75.3.pth"
        
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict_ema' in state_dict:
            state_dict = state_dict['state_dict_ema']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded default pretrained weights from {checkpoint_path}")
    
    def forward(self, x):
        ouputs = self.tpm(x)
        out = self.ppa(ouputs)
        out = self.trans(out)

        if self.injection:
            xx = out.split(self.channels, dim=1)
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    local_tokens = ouputs[i]
                    global_semantics = xx[i]
                    out_ = self.SIM[i](local_tokens, global_semantics)
                    results.append(out_)
            return results
        else:
            ouputs.append(out)
            return ouputs

class SegmentationHead(nn.Module):
    def __init__(self, in_channels_list, num_classes):
        super().__init__()
        head_channels = in_channels_list[0] 

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True)
        )

        self.cls_seg = nn.Conv2d(head_channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        target_size = inputs[0].shape[2:]
        feat = inputs[0]
        for i in range(1, len(inputs)):
            feat = feat + F.interpolate(inputs[i], size=target_size, mode='bilinear', align_corners=False)

        feat_head = self.fusion_conv(feat)
        logits = self.cls_seg(feat_head)

        return logits

class TopformerSeg(nn.Module):
    def __init__(self, num_classes=19, **kwargs):
        super().__init__()
        self.n_classes = num_classes
        self.backbone = Topformer(**kwargs)
        
        head_in_channels = [self.backbone.out_channels[i] for i in self.backbone.decode_out_indices]

        self.decode_head = SegmentationHead(
            in_channels_list=head_in_channels, 
            num_classes=num_classes
        )

    def forward(self, x, return_features=False):
        input_shape = x.shape[2:]
        
        features = self.backbone(x)

        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
        
        if return_features:
            return logits, features
            
        return logits