import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import ray

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

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
        beta,
        max_steps_per_epch
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        self.num_sample = num_sample
        
        self.local_model = local_model.to(self.device)
        
        self.teacher_model = copy.deepcopy(self.local_model).to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.beta = beta

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
            self.dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        self.total_samples = len(self.dataset)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def calc_negative_entropy(self, logits, masks):
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

    def train(self, global_parameters):
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()
        
        self.teacher_model.load_state_dict(global_parameters)
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
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
                ce_loss = self.criterion(outputs, masks)
                neg_ent = self.calc_negative_entropy(outputs, masks)

                loss = ce_loss + self.lambda_ent * neg_ent
                
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    for s_param, t_param in zip(self.local_model.parameters(), self.teacher_model.parameters()):
                        t_param.data = self.beta * t_param.data + (1.0 - self.beta) * s_param.data
                
                self.scheduler.step()

        local_ema_weights = {k: v.cpu() for k, v in self.teacher_model.state_dict().items()}
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_ema_weights, num_samples_trained