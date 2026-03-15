import sys
import os
project_root = "/root/KhaiDD/FedCar"
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import ray
import time 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

# The num_gpus for each worker is fixed as 0.2, because there are totally only 5 domains, thus 5 clients maximum :vv...
@ray.remote(num_gpus=0.2)
class FedEMA_Client:
    def __init__(
        self,
        data,
        client_id,
        local_model,

        num_epoch,
        batch_size,

        init_lr,
        min_lr,
        power,
        weight_decay,

        lambda_ent=0.01, # used in Negative Entropy
        max_steps_per_epch=10
    ):
        """
        data: domain object used in this client.
        client_id.
        local_model: model saved in client.

        num_epoch.
        batch_size.
        
        init_lr & min_lr & power: used to schedule learning rate.
        weight_decay: Used in AdamW optimizer (in this work, by default the optimizer will be AdamW optimizer)
        max_steps_per_epch.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        
        self.local_model = local_model.to(self.device)

        self.num_epoch = num_epoch
        self.batch_size = batch_size

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.lambda_ent = lambda_ent
        self.max_steps_per_epch = max_steps_per_epch

        # Init dataloader & optimizer & scheduler
        if self.data == 'cityscape':
            self.dataset = CityscapesDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/cityscape/leftImg8bit/train",
                labels_dir="/root/KhaiDD/FedCar/dataset/cityscape/gtFine/train"
            )
        elif self.data == "bdd100":
            self.dataset = BDD100KDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/bdd100/10k/train",
                labels_dir="/root/KhaiDD/FedCar/dataset/bdd100/labels/train"
            )
        elif self.data == "gta5":
            self.dataset = GTA5Dataset(
                list_of_paths=[
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part1",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part2",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part3",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part4",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part5",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part6",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part7",
                ]
            )
        elif self.data == "mapillary":
            self.dataset = MapillaryDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/mapillary/training"
            ) 
        elif self.data == "synthia":
            self.dataset = SynthiaDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/synthia/RAND_CITYSCAPES",
                start_index=0,
                end_index=6580
            )
        
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.num_samples = len(self.dataset)

        self.optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=init_lr, 
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=num_epoch, 
            power=self.power
        )

    def calc_negative_entropy(self, logits, masks):
        """
        Negative Entropy - E[\sum p * log_p].
        Penalize model if it is overconfident.
        """
        p = torch.softmax(logits, dim=1)
        log_p = torch.log_softmax(logits, dim=1)
        
        neg_ent = p * log_p
        neg_ent = torch.sum(neg_ent, dim=1) # Sum over classes: [B, H, W]
        
        valid_mask = (masks != 255).float()
        neg_ent = neg_ent * valid_mask
        
        valid_pixels = valid_mask.sum()
        if valid_pixels > 0:
            return neg_ent.sum() / valid_pixels
        return torch.tensor(0.0, device=logits.device)

    def train(
        self,
        global_parameters
    ):
        """
        global_parameters: New parameters after aggregated from server.
        """
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()
        
        for epoch in range(self.num_epoch):
            epoch_loss = 0
            
            for step, (images, masks) in enumerate(self.train_dataloader):
                if (step > self.max_steps_per_epch):
                    break
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.local_model(images)
                
                ce_loss = self.criterion(outputs, masks)

                neg_ent = self.calc_negative_entropy(outputs, masks)

                loss = ce_loss + self.lambda_ent * neg_ent
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

            self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        return local_weights, self.num_samples