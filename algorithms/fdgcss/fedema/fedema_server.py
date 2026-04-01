import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import copy

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .fedema_client import FedEMA_Client

class FedEMA_Server(FedAvg_Server):
    def __init__(self, beta=0.99, lambda_ent=0.01, **kwargs):
        self.beta = beta
        self.lambda_ent = lambda_ent
        super().__init__(**kwargs)

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} FedEMA workers via Ray ActorPool...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedEMA_Client.remote(
                    local_model=copy.deepcopy(self.backbone_model),
                    num_sample=self.num_sample,
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    max_steps_per_epch=self.max_steps_per_epch,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    lambda_ent=self.lambda_ent,
                    beta=self.beta,
                    num_rounds=self.num_rounds,
                    **kwargs
                )
            )
        return workers

    def update_global_model(self, aggregated_weights):
        ema_weights = {k: v.cpu() for k, v in self.backbone_model.state_dict().items()}
        new_ema = {}
        
        for key in ema_weights:
            new_ema[key] = (
                self.beta * ema_weights[key].float() 
                + (1 - self.beta) * aggregated_weights[key].to("cpu").float()
            ).to(ema_weights[key].dtype)
            
        self.backbone_model.load_state_dict(new_ema)