import sys
import os

project_root = os.getcwd()
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

# The num_gpus for each worker is fixed at 0.2 because there are only 5 domains total, allowing up to 5 concurrent clients.
@ray.remote(num_gpus=0.2)
class FedSR_Client:
    def __init__(
        self,
        data,
        client_id,
        local_model,

        num_sample,   
        num_epoch,
        batch_size,
        num_workers,  
        num_classes,

        init_lr,
        min_lr,
        power,
        weight_decay,
        max_steps_per_epch,

        z_dim, 
        alpha, 
        beta
    ):
        """
        Initializes the Federated Learning Client for FedSR (Federated Structural Regularization).

        Args:
            data (str): Target domain/dataset for this client.
            client_id (int/str): Unique client identifier.
            local_model (nn.Module): The local PyTorch model.
            num_sample (int): Number of training samples to load.
            num_epoch (int): Local training epochs.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of data loading workers.
            num_classes (int): Number of semantic classes.
            init_lr (float): Initial learning rate.
            min_lr (float): Minimum learning rate.
            power (float): Power for the PolynomialLR.
            weight_decay (float): Weight decay for AdamW.
            max_steps_per_epch (int): Max batches to process per epoch.
            z_dim (int): Dimension of the latent representation 'z'.
            alpha (float): Weight coefficient for L2 Regularization (L2R).
            beta (float): Weight coefficient for Conditional Mutual Information (CMI) loss.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        
        self.local_model = local_model.to(self.device)

        self.num_sample = num_sample
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.max_steps_per_epch = max_steps_per_epch

        self.alpha = alpha
        self.beta = beta
        self.z_dim = z_dim

        # Init dataloader
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
                list_of_paths=[
                    "dataset/gta5/gta5_part1",
                    "dataset/gta5/gta5_part2",
                    "dataset/gta5/gta5_part3",
                    "dataset/gta5/gta5_part4",
                    "dataset/gta5/gta5_part5",
                    "dataset/gta5/gta5_part6",
                    "dataset/gta5/gta5_part7",
                ],
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
        self.total_samples = len(self.dataset)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def train(self, global_parameters):
        """
        Executes local training with Structural Regularization (L2R & CMI).

        Args:
            global_parameters (dict): Aggregated model weights from the server.

        Returns:
            tuple: 
                - local_weights (dict): Updated local weights (moved to CPU).
                - total_samples (int): Client's dataset size.
        """
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()
        
        self.optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=self.init_lr, 
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=self.num_epoch * self.max_steps_per_epch, 
            power=self.power
        )

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch: 
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                
                logits, z, z_mu, z_sigma = self.local_model(
                    images, 
                    return_dist=True
                )
                
                loss = self.criterion(logits, masks)
                
                # Shrink the size of mask to match latent space 'z' (for GPU efficiency)
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
                
                # Conditional Mutual Information (CMI) Loss
                r_sigma_softplus = F.softplus(self.local_model.r_sigma)
                r_mu_batch = self.local_model.r_mu[y_valid]       
                r_sigma_batch = r_sigma_softplus[y_valid]
                
                loss_cmi = torch.log(r_sigma_batch) - torch.log(z_sigma_valid) + \
                           (z_sigma_valid**2 + (z_mu_valid - r_mu_batch)**2) / (2 * r_sigma_batch**2) - 0.5
                loss_cmi = loss_cmi.sum(dim=1).mean()

                # Total loss combination
                total_loss = loss + self.alpha * loss_l2r + self.beta * loss_cmi
                
                total_loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}

        return local_weights, self.total_samples