import sys
import os
project_root = "/root/KhaiDD/FedCar"
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import ray
import time 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

# The num_gpus for each worker is fixed as 0.2, because there are totally only 5 domains, thus 5 clients maximum :vv...
@ray.remote(num_gpus=0.2)
class FedDG_Client:
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

    def extract_amp_spectrum(self, img_tensor):
        fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
        amp = torch.abs(fft)
        return amp

    def frequency_swap(self, img_tensor, amp_trg, L=0.01):
        fft_src = torch.fft.fft2(img_tensor, dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

        amp_src_shift = torch.fft.fftshift(amp_src, dim=(-2, -1))
        amp_trg_shift = torch.fft.fftshift(amp_trg, dim=(-2, -1))

        _, _, h, w = amp_src_shift.shape
        b = int(np.floor(min(h, w) * L))
        c_h, c_w = h // 2, w // 2

        h1, h2 = c_h - b, c_h + b + 1
        w1, w2 = c_w - b, c_w + b + 1

        amp_src_shift[:, :, h1:h2, w1:w2] = amp_trg_shift[:, :, h1:h2, w1:w2]

        amp_src_final = torch.fft.ifftshift(amp_src_shift, dim=(-2, -1))
        fft_src_final = amp_src_final * torch.exp(1j * pha_src)
        src_in_trg = torch.fft.ifft2(fft_src_final, dim=(-2, -1))
        
        return torch.real(src_in_trg).float()    
    
    def train(self, global_parameters, global_amp_bank):
        """
        global_parameters: Weights từ Server
        global_amp_bank: List các tensor amp [B, C, H, W] collected from other clients
        """
        self.local_model.load_state_dict(global_parameters)
        self.local_model.train()
        
        local_amp_collection = [] 
        meta_step_size = self.init_lr

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step > self.max_steps_per_epch:
                    break
                images, masks = images.to(self.device), masks.to(self.device)
                local_amp_collection.append(self.extract_amp_spectrum(images).cpu())

                self.optimizer.zero_grad()

                # --- STAGE 1: Meta-Train (Original Domain) ---
                pred_inner = self.local_model(images)
                loss_inner = self.criterion(pred_inner, masks)

                grads = torch.autograd.grad(
                    loss_inner, 
                    self.local_model.parameters(),
                    retain_graph=True
                )

                fast_weights = {name: param - meta_step_size * grad for (name, param), grad in zip(self.local_model.named_parameters(), grads)}
                fast_state = {
                    **fast_weights, 
                    **dict(self.local_model.named_buffers())
                }

                # --- STAGE 2: Meta-Test (Simulated Continuous Domains) ---
                num_domain_used = 2
                # In the first loop, global_amp_bank is empty -> this is to prevent crash
                loss_outer = torch.tensor(0.0).to(self.device)
                if len(global_amp_bank) > 0:
                    for target_amp in random.sample(global_amp_bank, num_domain_used):
                        target_amp = target_amp.to(self.device)
                        
                        curr_b = images.shape[0]
                        target_amp = target_amp[:curr_b] if target_amp.shape[0] >= curr_b else target_amp.repeat(curr_b, 1, 1, 1)[:curr_b]

                        L = random.uniform(0.01, 0.05) 
                        images_aug = self.frequency_swap(images, target_amp, L=L)
                        
                        pred_outer = functional_call(self.local_model, fast_state, (images_aug,))
                        loss_outer += self.criterion(pred_outer, masks)
                    loss_outer = loss_outer / num_domain_used

                # --- STAGE 3: Episodic Update ---
                total_loss = loss_inner + loss_outer
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        
        if not local_amp_collection:
             local_amp_collection.append(self.extract_amp_spectrum(images).cpu())
        
        mean_local_amp = torch.mean(torch.stack(local_amp_collection), dim=0)

        return local_weights, self.num_samples, mean_local_amp