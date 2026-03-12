import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import ray

# The num_gpus for each worker is fixed as 0.2, because there are totally only 5 domains, thus 5 clients maximum :vv...
@ray.remote(num_gpus=0.2)
class FedAvg_Server:
    def __init__(
        self,
        data,
        local_model,

        num_epoch,
        batch_size,

        init_lr,
        min_lr,
        power,
        weight_decay,
    ):
        """
        data: domain object used in this client.
        local_model: model saved in client.

        num_epoch.
        batch_size.
        
        init_lr & min_lr & power: used to schedule learning rate.
        weight_decay: Used in AdamW optimizer (in this work, by default the optimizer will be AdamW optimizer)
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_epoch = num_epoch
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay

        # Init optimizer & scheduler
        self.optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=init_lr, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=num_epoch, 
            power=0.9
        )

    def train(
        
    )