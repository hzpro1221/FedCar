import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from algorithms.fdg_css.fedavg.fedavg_client import Base_FedAvg_Client

@ray.remote(num_gpus=0.2)
class FedSR_Client(Base_FedAvg_Client):
    def __init__(
        self, 
        num_classes, 
        z_dim=128, 
        alpha=0.01, 
        beta=0.001, 
        feat_dim=128,       
        hook_layer_name='bga', 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta
        self.feat_dim = feat_dim
        self.hook_layer_name = hook_layer_name

        def build_prob_bottleneck(in_dim, out_dim):
            mid_dim = max(in_dim // 2, out_dim) 
            return nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_dim, out_dim, kernel_size=1)
            )

        self.mu_net = build_prob_bottleneck(self.feat_dim, self.z_dim).to(self.device)
        self.logvar_net = build_prob_bottleneck(self.feat_dim, self.z_dim).to(self.device)

        self.r_mu = nn.Parameter(torch.randn(self.num_classes, self.z_dim).to(self.device))
        self.r_logvar = nn.Parameter(torch.zeros(self.num_classes, self.z_dim).to(self.device))

    def get_extra_parameters(self):
        return list(self.mu_net.parameters()) + \
               list(self.logvar_net.parameters()) + \
               [self.r_mu, self.r_logvar]

    def train(self, global_parameters, data_domain, client_id):
        self.load_dataset(data_domain)
        
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()
        
        all_params = list(self.local_model.parameters()) + self.get_extra_parameters()
        self.optimizer = torch.optim.AdamW(all_params, lr=self.init_lr, weight_decay=self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=self.num_epoch * self.max_steps_per_epch, 
            power=self.power
        )
        
        self.current_feat_head = None 

        def hook_fn(module, input, output):
            if isinstance(output, (list, tuple)):
                self.current_feat_head = output[-1]
            else:
                self.current_feat_head = output

        target_layer = dict(self.local_model.named_modules()).get(self.hook_layer_name, None)
        if target_layer is None:
            raise ValueError(f"Not found layer '{self.hook_layer_name}' in {self.local_model.__class__.__name__}")
        
        hook_handle = target_layer.register_forward_hook(hook_fn)

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch: break
                
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.local_model(images)
                
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                feat_head = self.current_feat_head
                
                mu = self.mu_net(feat_head)
                logvar = self.logvar_net(feat_head)
                
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                loss_ce = self.criterion(logits, masks)
                
                with torch.no_grad():
                    small_masks = F.interpolate(masks.unsqueeze(1).float(), size=z.shape[2:], mode='nearest').squeeze(1).long()
                    valid_mask = (small_masks != 255)
                
                if valid_mask.any():
                    z_v = z.permute(0, 2, 3, 1)[valid_mask]      
                    mu_p = mu.permute(0, 2, 3, 1)[valid_mask]    
                    logvar_p = logvar.permute(0, 2, 3, 1)[valid_mask]
                    y_v = small_masks[valid_mask]

                    # L2R Loss
                    loss_l2r = (z_v ** 2).sum(dim=1).mean()

                    # CMI Loss
                    mu_r = self.r_mu[y_v]
                    logvar_r = self.r_logvar[y_v]
                    var_p = torch.exp(logvar_p)
                    var_r = torch.exp(logvar_r)
                    
                    loss_cmi = 0.5 * (logvar_r - logvar_p + (var_p + (mu_p - mu_r)**2) / var_r - 1)
                    loss_cmi = loss_cmi.sum(dim=1).mean()
                else:
                    loss_l2r, loss_cmi = 0.0, 0.0

                total_loss = loss_ce + self.alpha * loss_l2r + self.beta * loss_cmi
                
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        hook_handle.remove()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained, client_id