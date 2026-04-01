import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import ray

from algorithms.fdg_css.fedavg.fedavg_client import Base_FedAvg_Client

_BN_STAT_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

def is_bn_statistic(key: str) -> bool:
    return any(key.endswith(s) for s in _BN_STAT_SUFFIXES)

def lab_transform(images: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    B = images.shape[0]
    num_translate = int(B * p)
    
    if num_translate < 1 or B < 2:
        return images

    out_images = images.clone()
    idx_to_translate = torch.randperm(B, device=images.device)[:num_translate]
    
    src_images = images[idx_to_translate]
    src_mu  = src_images.mean(dim=(2, 3), keepdim=True)             
    src_sig = src_images.std(dim=(2, 3),  keepdim=True).clamp(min=1e-6)

    perm    = torch.randint(0, B, (num_translate,), device=images.device)
    ref_images = images[perm]
    
    ref_mu  = ref_images.mean(dim=(2, 3), keepdim=True)
    ref_sig = ref_images.std(dim=(2, 3),  keepdim=True).clamp(min=1e-6) 

    alpha   = torch.rand(num_translate, 1, 1, 1, device=images.device)    

    mix_mu  = (1 - alpha) * src_mu  + alpha * ref_mu
    mix_sig = (1 - alpha) * src_sig + alpha * ref_sig

    translated_images = (src_images - src_mu) / src_sig * mix_sig + mix_mu
    out_images[idx_to_translate] = translated_images

    return out_images

@ray.remote(num_gpus=0.2)
class SiloBN_LAB_Client(Base_FedAvg_Client):
    def __init__(self, hnm_perc=0.25, **kwargs):
        super().__init__(**kwargs)
        self.hnm_perc = hnm_perc

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def _silobn_load(self, global_parameters: dict):
        local_state = self.local_model.state_dict()
        local_bn_stats = {
            k: v.clone() for k, v in local_state.items() if is_bn_statistic(k)
        }
        self.local_model.load_state_dict(global_parameters, strict=False)
        if local_bn_stats:
            self.local_model.load_state_dict(local_bn_stats, strict=False)
        
    def train(self, global_parameters, data_domain, client_id):
        self.load_dataset(data_domain)
        self._silobn_load(global_parameters)
        
        self.local_model.to(self.device)
        self.local_model.train()
        
        self.optimizer = optim.AdamW(
            self.local_model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=self.num_epoch * self.max_steps_per_epch, power=self.power
        )

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
                    break
                    
                images, masks = images.to(self.device), masks.to(self.device)
                
                images = lab_transform(images)
                
                self.optimizer.zero_grad()
                outputs = self.local_model(images)
                
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                    
                pixel_losses = self.criterion(outputs, masks)
                valid_mask   = masks != 255
                valid_losses = pixel_losses[valid_mask]

                if valid_losses.numel() > 0:
                    k = max(1, int(self.hnm_perc * valid_losses.numel()))
                    topk_losses, _ = torch.topk(valid_losses, k)
                    loss = topk_losses.mean()
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        local_weights = {
            k: v.cpu() for k, v in self.local_model.state_dict().items() if not is_bn_statistic(k) 
        }        
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained, client_id