import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ray
import gc

from algorithms.dataset_pytorch import (
    BDD100KDataset, CityscapesDataset, GTA5Dataset, 
    MapillaryDataset, SynthiaDataset
)

class Base_FedAvg_Client:
    def __init__(
        self,
        local_model,
        num_sample,
        num_epoch,
        batch_size,
        num_workers,
        init_lr,
        min_lr,
        power,
        weight_decay,
        max_steps_per_epch
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model = local_model.to(self.device)
        self.num_sample = num_sample
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.max_steps_per_epch = max_steps_per_epch
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        self.data_domain = None
        self.dataset = None
        self.train_dataloader = None
        self.total_samples = 0

    @staticmethod
    def _build_dataset(data, num_sample):
        if data == 'cityscape':
            return CityscapesDataset("dataset/cityscape/leftImg8bit/train", "dataset/cityscape/gtFine/train", num_sample=num_sample)
        elif data == "bdd100":
            return BDD100KDataset("dataset/bdd100/10k/train", "dataset/bdd100/labels/train", num_sample=num_sample)
        elif data == "gta5":
            return GTA5Dataset(list_of_paths=[f"dataset/gta5/gta5_part{i}" for i in range(1, 8)], num_sample=num_sample)
        elif data == "mapillary":
            return MapillaryDataset("dataset/mapillary/training", num_sample=num_sample) 
        elif data == "synthia":
            return SynthiaDataset("dataset/synthia/RAND_CITYSCAPES", start_index=0, end_index=6580, num_sample=num_sample)
        else:
            raise ValueError(f"Unknown dataset domain: {data}")

    def load_dataset(self, data_domain):
        if self.data_domain != data_domain:
            self.data_domain = data_domain
            if self.dataset is not None:
                del self.dataset
                del self.train_dataloader
                gc.collect()
            
            print(f"[Ray Worker] Loading dataset for domain: {data_domain}...")
            self.dataset = self._build_dataset(data_domain, self.num_sample)
            self.total_samples = len(self.dataset)
            self.train_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )

    def train(self, global_parameters, data_domain, client_id):
        self.load_dataset(data_domain)
        
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
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained, client_id

@ray.remote(num_gpus=0.2)
class FedAvg_Client(Base_FedAvg_Client):
    pass