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
class FedEMA_Client(Base_FedAvg_Client):
    def __init__(self, lambda_ent=0.01, beta=0.99, num_rounds=120, **kwargs):
        super().__init__(**kwargs)
        
        self.lambda_ent = lambda_ent
        self.beta = beta
        self.num_rounds = num_rounds

    def _negative_entropy(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        p     = torch.softmax(logits, dim=1)           
        log_p = torch.log_softmax(logits, dim=1)

        neg_ent = (p * log_p).sum(dim=1)

        valid_mask   = (masks != 255).float()
        valid_pixels = valid_mask.sum()

        if valid_pixels > 0:
            return (neg_ent * valid_mask).sum() / valid_pixels
        return torch.tensor(0.0, device=logits.device)

    def train(self, global_parameters: dict, data_domain, client_id):
        self.load_dataset(data_domain)
        
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()

        self.optimizer = optim.AdamW(
            self.local_model.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay,
        )

        total_iters = self.num_rounds * self.num_epoch * self.max_steps_per_epch
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=total_iters,
            power=self.power,
        )

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
                    break

                images = images.to(self.device)
                masks  = masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.local_model(images)
                
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                    ce_loss = sum(self.criterion(out, masks) for out in outputs)
                else:
                    logits = outputs
                    ce_loss = self.criterion(logits, masks)

                neg_ent = self._negative_entropy(logits, masks)

                loss = ce_loss - self.lambda_ent * neg_ent

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  
                
        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained, client_id