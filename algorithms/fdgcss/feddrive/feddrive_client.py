import sys
import os

project_root = os.getcwd()
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

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

@ray.remote(num_gpus=0.2)
class FedDrive_Client:
    def __init__(
        self,
        data,
        client_id,
        local_model,
        num_sample,
        num_epoch,
        batch_size,
        num_workers,
        init_lr,
        min_lr,
        power,
        weight_decay,
        hnm_perc,
        max_steps_per_epch
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        
        self.local_model = local_model.to(self.device)

        self.num_sample = num_sample  
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.hnm_perc = hnm_perc
        self.max_steps_per_epch = max_steps_per_epch

        if self.data == 'cityscape':
            self.dataset = CityscapesDataset(
                images_dir="dataset/cityscape/leftImg8bit/train",
                labels_dir="dataset/cityscape/gtFine/train",
                num_sample=self.num_sample
            )
        elif self.data == "bdd100":
            self.dataset = BDD100KDataset(
                images_dir="dataset/bdd100/10k/train",
                labels_dir="dataset/bdd100/labels/train",
                num_sample=self.num_sample
            )
        elif self.data == "gta5":
            self.dataset = GTA5Dataset(
                list_of_paths=[f"dataset/gta5/gta5_part{i}" for i in range(1, 8)],
                num_sample=self.num_sample
            )
        elif self.data == "mapillary":
            self.dataset = MapillaryDataset(
                root_dir="dataset/mapillary/training",
                num_sample=self.num_sample
            ) 
        elif self.data == "synthia":
            self.dataset = SynthiaDataset(
                root_dir="dataset/synthia/RAND_CITYSCAPES",
                start_index=0,
                end_index=6580,
                num_sample=self.num_sample
            )
        else:
            raise ValueError(f"Unknown dataset domain: {self.data}")
        
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.num_samples = len(self.dataset)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

        self.optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=init_lr, 
            weight_decay=weight_decay
        )

    def train(self, global_parameters):
        local_state = self.local_model.state_dict()
        local_norms = {k: v for k, v in local_state.items() if 'norm' in k or 'bn' in k}
        
        self.local_model.load_state_dict(global_parameters, strict=False)
        self.local_model.load_state_dict(local_norms, strict=False)

        self.local_model.to(self.device)
        self.local_model.train()
        
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=self.num_epoch * self.max_steps_per_epch, 
            power=self.power
        )

        for epoch in range(self.num_epoch):
            epoch_loss = 0.0
            
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch: 
                    break
                    
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.local_model(images)
                
                pixel_losses = self.criterion(outputs, masks)
                
                valid_mask = (masks != 255)
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
                
                epoch_loss += loss.item()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        
        return local_weights, self.num_samples