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

from .fedcovmatch_client import FedCovMatch_Client

class FedCovMatch_Server:
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
        lam_cov=1.0,  
        lam_syn=1.0,  
        lam_cons=1.0
    ):
        print("\n" + "="*50)
        print("[Server] Initializing FedCovMatch Server...")
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
        
        self.lam_cov = lam_cov
        self.lam_syn = lam_syn
        self.lam_cons = lam_cons
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device set to: {self.device}")
        print(f"[Server] Source domains registered: {self.source_domains}")
        print(f"[Server] Communication Rounds: {self.num_rounds} | Local Epochs: {self.num_epochs}")
        
        self.feature_dim = 256 # D: dimension of original feature map 
        self.proj_dim = 64     # d: dimension of feature map after compressing (d << D)
        
        # Init global statistics and Random Projection Matrix
        self.global_stats = {} 
        self.P_matrix = torch.randn(self.feature_dim, self.proj_dim) / (self.proj_dim ** 0.5)
        
        self.clients = []
        print("[Server] Initializing remote clients via Ray...")
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                FedCovMatch_Client.remote(
                    data=domain, 
                    client_id=i, 
                    local_model=copy.deepcopy(self.backbone_model),
                    
                    num_rounds=self.num_rounds, 
                    num_epoch=self.num_epochs, 
                    batch_size=self.batch_size, 
                    num_workers=self.num_workers,       
                    max_steps_per_epch=self.max_steps_per_epch,
                    
                    init_lr=self.init_lr, 
                    min_lr=self.min_lr, 
                    power=self.power, 
                    weight_decay=self.weight_decay,
                    
                    lam_cov=self.lam_cov, 
                    lam_syn=self.lam_syn, 
                    lam_cons=self.lam_cons, 
                    proj_dim=self.proj_dim, 
                    num_classes=self.num_classes
                )
            )
        print(f"[Server] Successfully initialized {len(self.clients)} clients.")
        print("="*50 + "\n")

    def set_seed(self, seed): 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[Server] Global seed set to {seed}.")

    def aggregate(self, local_weights_list, total_samples_list):
        print("[Server] Starting FedCovMatch aggregation...")
        total_samples = sum(total_samples_list)
        print(f"[Server] Total valid samples trained across all clients: {total_samples}")
        
        avg_weights = copy.deepcopy(local_weights_list[0])
        
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            
        for i in range(len(local_weights_list)):
            weight_factor = total_samples_list[i] / total_samples
            for key in avg_weights.keys():
                avg_weights[key] += (local_weights_list[i][key] * weight_factor).to(avg_weights[key].dtype)
                
        print("[Server] Aggregation complete.")
        return avg_weights

    def update_global_stats(self, local_moments_list):
        N_c_total = torch.zeros(self.num_classes)
        S_c_total = torch.zeros(self.num_classes, self.proj_dim)
        Q_c_total = torch.zeros(self.num_classes, self.proj_dim, self.proj_dim)

        for moments in local_moments_list:
            N_c_total += moments['n_kc']
            S_c_total += moments['s_kc']
            Q_c_total += moments['Q_kc']

        for c in range(self.num_classes):
            if N_c_total[c] > 1: 
                mu_g = S_c_total[c] / N_c_total[c]
                
                # Sigma = (Q - N * mu * mu^T) / (N - 1)
                mu_outer = torch.outer(mu_g, mu_g)
                Sigma_g = (Q_c_total[c] - N_c_total[c] * mu_outer) / (N_c_total[c] - 1)
                
                self.global_stats[c] = {
                    'mu': mu_g,
                    'Sigma': Sigma_g
                }

    def train(self, checkpoint_path):
        print(f"\n[Server] Commencing Federated Learning process for {self.num_rounds} rounds.")
        global_weights = self.backbone_model.state_dict()
        
        round_pbar = tqdm(range(self.num_rounds), desc="Round", position=0)
        for round_idx in round_pbar:
            print(f"\n--- [Server] Starting Round {round_idx + 1}/{self.num_rounds} ---")
            print("[Server] Broadcasting global weights & stats to all clients...")
            
            job_ids = [
                client.train.remote(
                    global_parameters=global_weights,
                    global_stats=self.global_stats,
                    P_matrix=self.P_matrix
                ) for client in self.clients
            ]

            print(f"[Server] Waiting for {len(self.clients)} clients to finish local training...")
            results = ray.get(job_ids)
            print("[Server] Received updates from all clients.")
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]
            local_moments_list = [r[2] for r in results]

            global_weights = self.aggregate(
                local_weights_list=local_weights_list, 
                total_samples_list=total_samples_list
            )
            
            print("[Server] Updating global backbone model with aggregated weights.")
            self.backbone_model.load_state_dict(global_weights)
            
            print("[Server] Updating Global Covariance Statistics...")
            self.update_global_stats(local_moments_list)
            
            round_pbar.set_description(f"Finished Round {round_idx + 1}")

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
        else:
            raise ValueError(f"Unknown target domain: {target_domain}")
        
        self.test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,       # Đổi True thành False khi test cho chuẩn đánh giá
            num_workers=self.num_workers, # [FIT FEDAVG]
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