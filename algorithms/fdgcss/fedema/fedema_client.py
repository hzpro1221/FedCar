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
import time 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

# The num_gpus for each worker is fixed at 0.2 because there are only 5 domains total, allowing up to 5 concurrent clients.
@ray.remote(num_gpus=0.2)
class FedEMA_Client:
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

        lambda_ent, 
        max_steps_per_epch
    ):
        """
        Initializes the Federated Learning Client for the FedEMA algorithm.

        Args:
            data (str): The name of the domain/dataset used by this client (e.g., 'cityscape', 'gta5').
            client_id (int/str): Unique identifier for the client.
            local_model (nn.Module): The PyTorch model assigned to this client.
            num_sample (int): Number of training samples to load from the dataset. Loads all if None.
            num_epoch (int): Number of local training epochs per round.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of subprocesses to use for data loading.
            init_lr (float): Initial learning rate for the optimizer.
            min_lr (float): Minimum learning rate for the learning rate scheduler.
            power (float): Power factor for the PolynomialLR scheduler.
            weight_decay (float): Weight decay coefficient for the AdamW optimizer.
            lambda_ent (float): Weight of the Negative Entropy penalty term.
            max_steps_per_epch (int): Maximum number of batches to process per epoch.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        self.num_sample = num_sample
        
        self.local_model = local_model.to(self.device)

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.lambda_ent = lambda_ent
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
        self.total_samples = len(self.dataset)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def calc_negative_entropy(self, logits, masks):
        """
        Calculates the Negative Entropy of the predictions: E[\sum p * log_p].
        Used to penalize the model if it becomes overconfident.
        
        Args:
            logits: Raw unnormalized predictions from the model [B, C, H, W]
            masks: Ground truth labels [B, H, W]
            
        Returns:
            Mean negative entropy computed only over valid pixels.
        """
        p = torch.softmax(logits, dim=1)
        log_p = torch.log_softmax(logits, dim=1)
        
        neg_ent = p * log_p
        neg_ent = torch.sum(neg_ent, dim=1) 
        
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
        Executes the local training process using the aggregated global weights.

        Args:
            global_parameters (dict): The latest aggregated model state_dict from the central server.

        Returns:
            tuple: 
                - local_weights (dict): The updated state_dict of the local model (moved to CPU).
                - num_samples_trained (int): The actual number of samples processed during this round.
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
            epoch_loss = 0.0
            
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
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
        
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained