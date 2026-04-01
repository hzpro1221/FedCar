import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import random
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
import ray
from ray.util.actor_pool import ActorPool
import wandb 

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .feddg_ga_client import FedDG_GA_Client

class FedDG_GA_Server(FedAvg_Server):
    def __init__(self, ga_step_size, **kwargs):
        self.ga_step_size = ga_step_size
        self.current_round = 0  
        
        super().__init__(**kwargs)
        
        self.num_clients = len(self.source_domains)
        self.client_weights = [1.0 / self.num_clients] * self.num_clients

        initial_weights = {k: v.cpu() for k, v in self.backbone_model.state_dict().items()}
        self.all_local_weights = {domain: copy.deepcopy(initial_weights) for domain in self.source_domains}

    def _build_model(self):
        if self.model_name == 'segformer_b0_avg_ga':
            from .segformer_b0_avg_ga import SegFormerB0_Avg_GA
            return SegFormerB0_Avg_GA(num_classes=self.num_classes)
        return super()._build_model()

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} FedDG_GA workers via Ray ActorPool...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedDG_GA_Client.remote(
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
                    **kwargs
                )
            )
        return workers

    def aggregate(self, local_weights_list, gaps_list):
        print(f"[Server] Generalization Gaps: {[f'{g:.4f}' for g in gaps_list]}")
        
        mu = sum(gaps_list) / self.num_clients
        max_diff = max([abs(g - mu) for g in gaps_list]) + 1e-8  
        
        d_r = (1.0 - (self.current_round + 1) / self.num_rounds) * self.ga_step_size
        print(f"[Server] Round {self.current_round + 1}/{self.num_rounds}, Decay factor d_r = {d_r:.6f}")
        
        a_temp = []
        for i in range(self.num_clients):
            new_w = self.client_weights[i] + d_r * ((gaps_list[i] - mu) / max_diff)
            a_temp.append(max(new_w, 1e-10)) 
            
        sum_a_temp = sum(a_temp)
        
        if sum_a_temp < 1e-8:
            print("[WARNING] All weights near zero! Resetting to uniform distribution.")
            self.client_weights = [1.0 / self.num_clients] * self.num_clients
        else:
            self.client_weights = [w / sum_a_temp for w in a_temp]
            
        print(f"[Server] FedGA - Updated Weights: {[f'{w:.4f}' for w in self.client_weights]}")

        print("[Server] Starting FedGA aggregation...")
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])

        for i in range(self.num_clients):
            local_w = local_weights_list[i]
            weight_factor = self.client_weights[i]
            for key in avg_weights.keys():
                avg_weights[key] += (local_w[key] * weight_factor).to(avg_weights[key].dtype)
                
        print("[Server] Aggregation complete.")
        self.current_round += 1
        return avg_weights

    def train(self, target_domain, checkpoint_path):
        print(f"\n[Server] Commencing FedDG-GA with State Persistence.")
        global_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="Round")

        for round_idx in round_pbar:
            self.current_round = round_idx
            tasks = []
            
            for i, domain in enumerate(self.source_domains):
                tasks.append({
                    "global_weights": global_weights,
                    "prev_local_weights": self.all_local_weights[domain], 
                    "data_domain": domain,
                    "client_id": i
                })
            
            results = list(self.actor_pool.map(
                lambda actor, task: actor.train.remote(
                    global_parameters=task["global_weights"],
                    prev_local_parameters=task["prev_local_weights"],
                    data_domain=task["data_domain"],
                    client_id=task["client_id"]
                ),
                tasks
            ))

            local_weights_list = [r[0] for r in results]
            gaps_list = [r[1] for r in results]
            
            for i, domain in enumerate(self.source_domains):
                self.all_local_weights[domain] = local_weights_list[i]

            aggregated_weights = self.aggregate(local_weights_list, gaps_list)
            self.update_global_model(aggregated_weights)
            global_weights = self.backbone_model.state_dict()
            
            miou, _, _ = self.evaluate(target_domain=target_domain)
            wandb.log({"Round": round_idx + 1, "mIoU": miou * 100})
            round_pbar.set_postfix(mIoU=f"{miou * 100:.2f}%")

        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model