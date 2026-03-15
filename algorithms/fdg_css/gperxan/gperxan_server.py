import sys
import os
project_root = "/root/KhaiDD/FedCar"
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

from .gperxan_client import gPerXAN_Client
from .segformer_b0_gperxan import SegFormerB0_gPerXAN

class gPerXAN_Server:
    def __init__(
        self, 
        num_classes,
        backbone_model, 
        source_domains,
        num_rounds, 
        num_epochs, 
        batch_size,

        init_lr,
        min_lr,
        power,
        weight_decay
    ):
        """
        1. num_classess: number of class to classify.
        1.1 backbone_model: The instance of backbone model's class. In this work, it's fixed as SegmentFormer-B0.
        2. source_domains: A list of source domains used to train (for simplicity, each client will correspond with a domain).
        3. num_rounds: Number of communication rounds.
        4. num_epochs: Number of epoch (used to train in server side).
        5. batch_size: Number of batch size (also used in server side).

        6. init_lr & min_lr & power: used to schedule learning rate.
        7. weight_decay: Used in AdamW optimizer (in this work, by default the optimizer will be AdamW optimizer)
        """
        print("\n" + "="*50)
        print("[Server] Initializing PerXAN Server...")
        self.num_classes = num_classes
        self.backbone_model = backbone_model
        self.source_domains = source_domains
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay

        # :vv trivial.. but this is for identifing device (in case you don't know)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device set to: {self.device}")
        print(f"[Server] Source domains registered: {self.source_domains}")
        print(f"[Server] Communication Rounds: {self.num_rounds} | Local Epochs: {self.num_epochs}")

        # For each domain -> we init a client, and assign a domain to him
        self.clients = []
        print("[Server] Initializing remote clients via Ray...")
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                gPerXAN_Client.remote(
                    data=domain,
                    client_id=i,
                    local_model=SegFormerB0_gPerXAN(
                        num_classes=self.num_classes
                    ),

                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,

                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay
                )
            )
        print(f"[Server] Successfully initialized {len(self.clients)} remote clients.")
        print("="*50 + "\n")
    
    def set_seed(self, seed): 
        # for python and numpy
        random.seed(seed)
        np.random.seed(seed)

        # pytorch cpu & gpu (this is only for single GPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[Server] Global seed set to {seed}.")
    
    def aggregate(self, local_weights_list, total_samples_list):
        """
        W_global = sum( (n_k / n_total) * W_k )
        """
        print("[Server] Starting PerXAN aggregation...")
        total_samples = sum(total_samples_list)
        print(f"[Server] Total samples across all clients: {total_samples}")
        
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

    def train(self, checkpoint_path):
        print(f"\n[Server] Commencing Federated Learning process for {self.num_rounds} rounds.")
        global_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="Round", position=0)

        for round_idx in round_pbar:
            print(f"\n--- [Server] Starting Round {round_idx + 1}/{self.num_rounds} ---")
            print("[Server] Broadcasting global weights to all clients...")
            job_ids = [
                client.train.remote(
                    global_parameters=global_weights
                ) 
                for i, client in enumerate(self.clients)
            ]

            print(f"[Server] Waiting for {len(self.clients)} clients to finish local training...")
            results = ray.get(job_ids)
            print(f"[Server] Received local weights from all clients.")
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]

            global_weights = self.aggregate(
                local_weights_list=local_weights_list, 
                total_samples_list=total_samples_list
            )

            print("[Server] Updating global backbone model with aggregated weights.")
            self.backbone_model.load_state_dict(global_weights)
            round_pbar.set_description(f"Num Finished Round {round_idx + 1}")

        # save model after training
        print(f"\n[Server] Training complete. Saving global model to {checkpoint_path}")
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
        dataset=None
        if target_domain == 'cityscape':
            dataset = CityscapesDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/cityscape/leftImg8bit/val",
                labels_dir="/root/KhaiDD/FedCar/dataset/cityscape/gtFine/val"
            )
        elif target_domain == "bdd100":
            dataset = BDD100KDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/bdd100/10k/val",
                labels_dir="/root/KhaiDD/FedCar/dataset/bdd100/labels/val"
            )
        elif target_domain == "gta5":
            dataset = GTA5Dataset(
                list_of_paths=[
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part8",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part9",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part10"
                ]
            )
        elif target_domain == "mapillary":
            dataset = MapillaryDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/mapillary/validation"
            ) 
        elif target_domain == "synthia":
            dataset = SynthiaDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/synthia/RAND_CITYSCAPES",
                start_index=6580
            )
        
        self.test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        print(f"[Server] Dataset loaded. Total batches to evaluate: {len(self.test_dataloader)}")
        
        print("[Server] Starting inference loop...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.backbone_model(images) # Output: [B, 19, 512, 512]

                preds = torch.argmax(outputs, dim=1) # Preds: [B, 512, 512]
                
                # Only caculate on pixel that is not 255 
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