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

# The num_gpus for each worker is fixed at 0.2 because there are only 5 domains total, allowing up to 5 concurrent clients.
@ray.remote(num_gpus=0.2)
class FedDG_Client:
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
        
        meta_lr,
        num_domains_used,
        freq_l_min,
        freq_l_max,
        max_steps_per_epch=10
    ):
        """
        Initializes the Federated Learning Client for FedDG (Federated Domain Generalization).
        Utilizes Episodic Meta-Learning with Continuous Frequency Space Interpolation.

        Args:
            data (str): Target domain/dataset for this client.
            client_id (int/str): Unique client identifier.
            local_model (nn.Module): The local PyTorch model.
            num_sample (int): Number of training samples to load.
            num_epoch (int): Local training epochs.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of data loading workers.
            init_lr (float): Initial learning rate for the outer loop optimizer.
            min_lr (float): Minimum learning rate.
            power (float): Power for the PolynomialLR.
            weight_decay (float): Weight decay for AdamW.
            max_steps_per_epch (int): Max batches to process per epoch.
            meta_lr (float): Inner-loop (fast adaptation) learning rate for Meta-Learning.
            num_domains_used (int): Number of external domain amplitude spectra to sample for Meta-Test.
            freq_l_min (float): Minimum interpolation ratio (L) in frequency space.
            freq_l_max (float): Maximum interpolation ratio (L) in frequency space.
        """
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
        self.max_steps_per_epch = max_steps_per_epch
        
        self.meta_lr = meta_lr
        self.num_domains_used = num_domains_used
        self.freq_l_min = freq_l_min
        self.freq_l_max = freq_l_max

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

    def extract_amp_spectrum(self, img_tensor):
        """Extracts the amplitude spectrum from an image tensor using 2D FFT."""
        fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
        amp = torch.abs(fft)
        return amp

    def frequency_swap(self, img_tensor, amp_trg, L=0.01):
        """
        Interpolates the low-frequency amplitude of the source image with a target amplitude.
        This simulates a new domain style while preserving semantic structure (phase).
        """
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
        Executes local Episodic Meta-Learning using Frequency Space Interpolation.

        Args:
            global_parameters (dict): Aggregated model weights from the server.
            global_amp_bank (list): List of amplitude spectrum tensors [B, C, H, W] collected from other clients.

        Returns:
            tuple: 
                - local_weights (dict): Updated local weights (moved to CPU).
                - total_samples (int): Client's dataset size.
                - mean_local_amp (torch.Tensor): Mean amplitude spectrum of the client's local data (CPU).
        """
        self.local_model.load_state_dict(global_parameters)
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

        local_amp_collection = [] 

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch:
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                local_amp_collection.append(self.extract_amp_spectrum(images).cpu())

                self.optimizer.zero_grad()

                # --- STAGE 1: Meta-Train (Original Domain) ---
                pred_inner = self.local_model(images)
                loss_inner = self.criterion(pred_inner, masks)

                # Compute gradients for inner update
                grads = torch.autograd.grad(
                    loss_inner, 
                    self.local_model.parameters(),
                    retain_graph=True
                )

                # Fast adaptation (Inner loop update)
                fast_weights = {name: param - self.meta_lr * grad for (name, param), grad in zip(self.local_model.named_parameters(), grads)}
                fast_state = {
                    **fast_weights, 
                    **dict(self.local_model.named_buffers())
                }

                # --- STAGE 2: Meta-Test (Simulated Continuous Domains) ---
                loss_outer = torch.tensor(0.0).to(self.device)
                
                if len(global_amp_bank) > 0:
                    # Determine how many domains to sample (handle case where bank has fewer than requested)
                    actual_num_domains = min(self.num_domains_used, len(global_amp_bank))
                    
                    for target_amp in random.sample(global_amp_bank, actual_num_domains):
                        target_amp = target_amp.to(self.device)
                        
                        # Match batch dimensions
                        curr_b = images.shape[0]
                        target_amp = target_amp[:curr_b] if target_amp.shape[0] >= curr_b else target_amp.repeat(curr_b, 1, 1, 1)[:curr_b]

                        # Apply frequency swap to simulate domain shift
                        L = random.uniform(self.freq_l_min, self.freq_l_max) 
                        images_aug = self.frequency_swap(images, target_amp, L=L)
                        
                        # Forward pass with adapted weights on augmented data
                        pred_outer = functional_call(self.local_model, fast_state, (images_aug,))
                        loss_outer += self.criterion(pred_outer, masks)
                        
                    loss_outer = loss_outer / actual_num_domains

                # --- STAGE 3: Episodic Update ---
                # Combine losses to update original parameters (Outer loop update)
                total_loss = loss_inner + loss_outer
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        
        # Fallback in case loop broke without appending
        if not local_amp_collection:
            local_amp_collection.append(self.extract_amp_spectrum(images).cpu())
        
        # Calculate mean amplitude representing this client's domain
        mean_local_amp = torch.mean(torch.stack(local_amp_collection), dim=0)

        return local_weights, self.total_samples, mean_local_amp