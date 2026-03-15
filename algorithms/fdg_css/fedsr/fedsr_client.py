import sys
import os
project_root = "/root/KhaiDD/FedCar"
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import copy
import ray
import time 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

# The num_gpus for each worker is fixed as 0.2, because there are totally only 5 domains, thus 5 clients maximum :vv...
@ray.remote(num_gpus=0.2)
class FedSR_Client:
    def __init__(
        self,
        data,
        client_id,
        local_model,

        num_epoch,
        batch_size,
        num_classes,

        init_lr,
        min_lr,
        power,
        weight_decay,
        max_steps_per_epch=10,

        z_dim=128, 
        alpha=0.01, 
        beta=0.001
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
        self.max_steps_per_epch = max_steps_per_epch

        self.alpha = alpha
        self.beta = beta

        self.r_mu = nn.Parameter(torch.zeros(num_classes, z_dim).to(self.device))
        self.r_sigma = nn.Parameter(torch.ones(num_classes, z_dim).to(self.device))

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
            list(self.local_model.parameters()) + [self.r_mu, self.r_sigma], 
            lr=init_lr, weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=num_epoch, 
            power=self.power
        )

    def train(self, global_parameters):
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()
        
        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if (step > self.max_steps_per_epch): 
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                
                logits, z, z_mu, z_sigma = self.local_model(
                    images, 
                    return_dist=True
                )
                
                loss = self.criterion(logits, masks)
                
                # Shrink the size of mask (for GPU effieciency)
                small_masks = F.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=z.shape[2:], 
                    mode='nearest'
                ).squeeze(1).long()

                valid_mask = (small_masks != 255) 
                
                z_valid = z.permute(0, 2, 3, 1)[valid_mask]
                z_mu_valid = z_mu.permute(0, 2, 3, 1)[valid_mask]
                z_sigma_valid = z_sigma.permute(0, 2, 3, 1)[valid_mask]
                y_valid = small_masks[valid_mask]
                
                # L2 Regularization
                loss_l2r = z_valid.norm(dim=1).mean()
                
                # CMI Loss
                r_sigma_softplus = F.softplus(self.r_sigma)
                
                r_mu_batch = self.r_mu[y_valid]       
                r_sigma_batch = r_sigma_softplus[y_valid] 
                
                loss_cmi = torch.log(r_sigma_batch) - torch.log(z_sigma_valid) + \
                           (z_sigma_valid**2 + (z_mu_valid - r_mu_batch)**2) / (2 * r_sigma_batch**2) - 0.5
                loss_cmi = loss_cmi.sum(dim=1).mean()

                # Total loss 
                total_loss = loss + self.alpha * loss_l2r + self.beta * loss_cmi
                
                total_loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        return local_weights, self.num_samples