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

@ray.remote(num_gpus=0.2)
class FedDG_GA_Client(Base_FedAvg_Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_loss(self):
        self.local_model.eval()
        total_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
                    break
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.local_model(images)
                
                if isinstance(outputs, (tuple, list)):
                    loss = sum(self.criterion(out, masks) for out in outputs)
                else:
                    loss = self.criterion(outputs, masks)
                    
                total_loss += loss.item()
                num_steps += 1

        return total_loss / max(num_steps, 1)

    def train(self, global_parameters, prev_local_parameters, data_domain, client_id):
        self.load_dataset(data_domain)
        self.local_model.to(self.device)
        
        self.local_model.load_state_dict(prev_local_parameters)
        local_loss_prev = self._compute_loss()
        
        self.local_model.load_state_dict(global_parameters)
        global_loss = self._compute_loss()
        
        generalization_gap = global_loss - local_loss_prev

        self.local_model.train()
        
        self.optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=self.init_lr, 
            weight_decay=self.weight_decay
        )

        actual_steps_per_epoch = min(len(self.train_dataloader), self.max_steps_per_epch)
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=self.num_epoch * actual_steps_per_epoch, 
            power=self.power
        )

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
                    break
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.local_model(images)
                
                if isinstance(outputs, (tuple, list)):
                    loss = 0.0
                    for out in outputs:
                        loss += self.criterion(out, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        return local_weights, generalization_gap, client_id