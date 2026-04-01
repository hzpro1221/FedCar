import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import ray
from algorithms.fdg_css.fedavg.fedavg_client import Base_FedAvg_Client

class XAN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(XAN, self).__init__()
        self.num_features = num_features
        
        self.inorm = nn.InstanceNorm2d(num_features, affine=True, eps=eps)
        self.bnorm = nn.BatchNorm2d(num_features, affine=True, eps=eps, momentum=momentum)
        
        self.w_in = nn.Parameter(torch.ones(1))
        self.w_bn = nn.Parameter(torch.ones(1))

    def forward(self, x):
        _, _, h, w = x.shape
        
        if h * w <= 1:
            return self.w_bn * self.bnorm(x)
            
        return self.w_in * self.inorm(x) + self.w_bn * self.bnorm(x)
        
def replace_bn_with_xan(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)):
            xan = XAN(child.num_features, eps=child.eps, momentum=child.momentum)
            
            if child.affine:
                xan.inorm.weight.data = child.weight.data.clone()
                xan.inorm.bias.data = child.bias.data.clone()
                xan.bnorm.weight.data = child.weight.data.clone()
                xan.bnorm.bias.data = child.bias.data.clone()
                
            xan.bnorm.running_mean.data = child.running_mean.data.clone()
            xan.bnorm.running_var.data = child.running_var.data.clone()
            if hasattr(child, 'num_batches_tracked') and child.num_batches_tracked is not None:
                xan.bnorm.num_batches_tracked.data = child.num_batches_tracked.data.clone()

            setattr(module, name, xan)
        else:
            replace_bn_with_xan(child)

@ray.remote(num_gpus=0.2)
class gPerXAN_Client(Base_FedAvg_Client):
    def __init__(self, reg_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        
        self.reg_weight = reg_weight
        
        if hasattr(self.local_model, 'n_classes'):
            self.num_classes = self.local_model.n_classes
        elif hasattr(self.local_model, 'num_classes'):
            self.num_classes = self.local_model.num_classes
        else:
            self.num_classes = 19 

        replace_bn_with_xan(self.local_model)
        self.local_model.to(self.device)

    def train(self, global_parameters, data_domain, client_id):
        self.load_dataset(data_domain)

        local_state = self.local_model.state_dict()
        for k, v in global_parameters.items():
            if ".bnorm." not in k and "score_" not in k.lower():
                local_state[k] = v
        
        self.local_model.load_state_dict(local_state)
        self.local_model.to(self.device)
        self.local_model.train()

        if hasattr(self.local_model, 'decode_head'): 
            server_head = copy.deepcopy(self.local_model.decode_head)
        elif hasattr(self.local_model, 'head'): 
            server_head = copy.deepcopy(self.local_model.head)
        elif hasattr(self.local_model, 'model') and hasattr(self.local_model.model, 'decode_head'):
            server_head = copy.deepcopy(self.local_model.model.decode_head)
        else:
            raise AttributeError("[gPerXAN Error] Could not find Main Head (decode_head or head) in local_model")
            
        server_head.to(self.device)
        server_head.eval()
        for param in server_head.parameters():
            param.requires_grad = False

        self.optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=self.init_lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=self.num_epoch * self.max_steps_per_epch, 
            power=self.power
        )

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
                    break

                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                
                local_logits, features = self.local_model(images, return_features=True)
                loss_local = self.criterion(local_logits, masks)

                server_logits_raw = server_head(features)
                server_logits = F.interpolate(
                    server_logits_raw, 
                    size=images.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                loss_reg = self.criterion(server_logits, masks)

                loss = loss_local + self.reg_weight * loss_reg
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        local_weights = {}
        for k, v in self.local_model.state_dict().items():
            if ".bnorm." not in k:
                local_weights[k] = v.cpu()

        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained, client_id