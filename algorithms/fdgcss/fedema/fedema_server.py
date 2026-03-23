import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import random
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
import ray

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

from .fedema_client import FedEMA_Client
from .segformer_b0_ema import SegFormerB0_EMA

class FedEMA_Server:
    def __init__(
        self, 
        num_classes,
        backbone_model, 
        source_domains,
        num_rounds, 
        num_epochs, 
        batch_size,
        
        num_workers,
        num_sample,
        max_steps_per_epch,

        init_lr,
        min_lr,
        power,
        weight_decay,
        beta,           
        lambda_ent
    ):
        """
        Initializes the Federated Learning Server for the FedEMA algorithm.

        Args:
            num_classes (int): Number of classes for the segmentation task.
            backbone_model (nn.Module): The global backbone model (e.g., SegFormer-B0).
            source_domains (list): List of source domains/datasets used for training.
            num_rounds (int): Total number of communication rounds.
            num_epochs (int): Number of local epochs for each client per round.
            batch_size (int): Batch size for local client training and server evaluation.
            num_workers (int): Number of subprocesses for data loading.
            num_sample (int): Number of samples to load per dataset (loads all if None).
            max_steps_per_epch (int): Maximum number of training steps per epoch for clients.
            init_lr (float): Initial learning rate for the optimizer.
            min_lr (float): Minimum learning rate for the scheduler.
            power (float): Power factor for the polynomial learning rate scheduler.
            weight_decay (float): Weight decay coefficient for the AdamW optimizer.
            beta (float): Momentum parameter for the Exponential Moving Average (EMA).
            lambda_ent (float): Weight of the Negative Entropy penalty term.
        """
        print("\n" + "="*50)
        print("[Server] Initializing FedEMA Server...")
        self.num_classes = num_classes
        self.backbone_model = backbone_model
        self.source_domains = source_domains
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.num_workers = num_workers
        self.num_sample = num_sample
        self.max_steps_per_epch = max_steps_per_epch

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.beta = beta
        self.lambda_ent = lambda_ent
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device set to: {self.device}")
        print(f"[Server] Source domains registered: {self.source_domains}")
        print(f"[Server] Beta EMA: {self.beta} | Lambda Entropy: {self.lambda_ent}")

        # For each domain -> we init a client, and assign a domain to him
        self.clients = []
        print("[Server] Initializing remote clients via Ray...")
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                FedEMA_Client.remote(
                    data=domain,
                    client_id=i,
                    local_model=SegFormerB0_EMA(
                        num_classes=self.num_classes
                    ),
                    num_sample=self.num_sample,
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    
                    lambda_ent=self.lambda_ent,
                    max_steps_per_epch=self.max_steps_per_epch
                )
            )
        print(f"[Server] Successfully initialized {len(self.clients)} remote clients.")
        print("="*50 + "\n")
    
    def set_seed(self, seed): 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[Server] Global seed set to {seed}.")
    
    def aggregate(self, local_weights_list, total_samples_list):
        """
        Aggregates local model weights using FedAvg strategy:
        W_global = sum( (n_k / n_total) * W_k )
        """
        print("[Server] Starting FedAvg aggregation step...")
        total_samples = sum(total_samples_list)
        print(f"[Server] Total valid samples trained across all clients: {total_samples}")
        
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])

        for i in range(len(local_weights_list)):
            local_w = local_weights_list[i]
            n_k = total_samples_list[i]
            weight_factor = n_k / total_samples

            for key in avg_weights.keys():
                avg_weights[key] += (local_w[key] * weight_factor).to(avg_weights[key].dtype)
        
        print("[Server] Aggregation complete.")
        return avg_weights

    def update_ema(self, global_weights, aggregated_weights):
        """
        Updates the global model weights using Exponential Moving Average:
        \omega_EMA^r = \beta * \omega_EMA^{r-1} + (1 - \beta) * \omega_aggregated
        """
        for key in global_weights.keys():
            if torch.is_floating_point(global_weights[key]):
                global_weights[key] = self.beta * global_weights[key] + (1.0 - self.beta) * aggregated_weights[key]
            else:
                global_weights[key] = aggregated_weights[key]
        return global_weights

    def train(self, checkpoint_path):
        print(f"\n[Server] Commencing FedEMA Learning process for {self.num_rounds} rounds.")
        
        # Init \omega_EMA^0
        global_ema_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="Round", position=0)

        for round_idx in round_pbar:
            print(f"\n--- [Server] Starting Round {round_idx + 1}/{self.num_rounds} ---")
            
            # Distribute \omega_EMA^{r-1} to clients
            print("[Server] Broadcasting global EMA weights to all clients...")
            job_ids = [
                client.train.remote(global_parameters=global_ema_weights) 
                for client in self.clients
            ]

            print(f"[Server] Waiting for {len(self.clients)} clients to finish local training...")
            results = ray.get(job_ids)
            print("[Server] Received local updates from all clients.")
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]

            # Calculate \omega (Aggregated Weights via FedAvg)
            aggregated_weights = self.aggregate(local_weights_list, total_samples_list)

            # Update \omega_EMA^r (Apply EMA on the aggregated weights)
            print("[Server] Updating global model using EMA...")
            global_ema_weights = self.update_ema(global_ema_weights, aggregated_weights)

            # Load updated weights into server's backbone
            self.backbone_model.load_state_dict(global_ema_weights)
            round_pbar.set_description(f"Finished Round {round_idx + 1}")

        print(f"\n[Server] Training complete. Saving global EMA model to {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model
    
    def evaluate(self, target_domain, checkpoint_path):
        print("\n" + "="*50)
        print(f"[Server] Starting evaluation on Target Domain: {target_domain}")

        self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        print(f"[Server] Loaded checkpoint from {checkpoint_path}")

        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)
        print(f"[Server] Loading dataset for {target_domain}...")
        
        eval_num_sample = int(self.num_sample / 10) if self.num_sample is not None else None
        dataset = None

        if target_domain == 'cityscape':
            dataset = CityscapesDataset(
                images_dir="dataset/cityscape/leftImg8bit/val",
                labels_dir="dataset/cityscape/gtFine/val",
                num_sample=eval_num_sample
            )
        elif target_domain == "bdd100":
            dataset = BDD100KDataset(
                images_dir="dataset/bdd100/10k/val",
                labels_dir="dataset/bdd100/labels/val",
                num_sample=eval_num_sample
            )
        elif target_domain == "gta5":
            dataset = GTA5Dataset(
                list_of_paths=[
                    "dataset/gta5/gta5_part8",
                    "dataset/gta5/gta5_part9",
                    "dataset/gta5/gta5_part10"
                ],
                num_sample=eval_num_sample
            )
        elif target_domain == "mapillary":
            dataset = MapillaryDataset(
                root_dir="dataset/mapillary/validation",
                num_sample=eval_num_sample
            ) 
        elif target_domain == "synthia":
            dataset = SynthiaDataset(
                root_dir="dataset/synthia/RAND_CITYSCAPES",
                start_index=6580,
                end_index=None,
                num_sample=eval_num_sample
            )
        
        self.test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True
        )
        print(f"[Server] Dataset loaded. Total batches to evaluate: {len(self.test_dataloader)}")
        
        print("[Server] Starting inference loop...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.backbone_model(images) # Output: [B, 19, 512, 512]
                preds = torch.argmax(outputs, dim=1)  # Preds: [B, 512, 512]
                
                # Only calculate on pixels that are not 255 (ignore_index)
                mask_valid = (masks != 255)
                
                target = masks[mask_valid]
                predict = preds[mask_valid]
                
                indices = self.num_classes * target + predict
                conf_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

        print("[Server] Calculating metrics from confusion matrix...")
        tp = torch.diag(conf_matrix)
        fp = torch.sum(conf_matrix, dim=0) - tp
        fn = torch.sum(conf_matrix, dim=1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou = torch.mean(iou_per_class).item()

        total_correct_pixels = torch.sum(tp)
        total_valid_pixels = torch.sum(conf_matrix)
        pixel_acc = (total_correct_pixels / (total_valid_pixels + 1e-10)).item()

        print("\n" + "="*40)
        print(f"Evaluate result: \n- mIoU: {miou*100:.2f}%\n- Pixel Accuracy: {pixel_acc*100:.2f}%")
        print("="*40)
        for i, iou in enumerate(iou_per_class):
            print(f"Class {i:2d}: {iou.item()*100:.2f}%")
            
        return miou, pixel_acc, iou_per_class