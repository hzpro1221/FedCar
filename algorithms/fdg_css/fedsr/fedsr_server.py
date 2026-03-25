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
import wandb 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset
from .fedsr_client import FedSR_Client
from .segformer_b0_sr import SegFormerB0_SR

class FedSR_Server:
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
        z_dim=128,
        alpha=0.01,
        beta=0.001
    ):
        print("\n" + "="*50)
        print("[Server] Initializing FedSR Server...")
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
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device: {self.device}")
        print(f"[Server] Registered Domains: {self.source_domains}")
        print(f"[Server] Config: z_dim={z_dim}, alpha={alpha}, beta={beta}")

        self.clients = []
        print("[Server] Launching remote clients...")
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                FedSR_Client.remote(
                    data=domain,
                    client_id=i,
                    local_model=SegFormerB0_SR(
                        num_classes=self.num_classes,
                        z_dim=self.z_dim
                    ),
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    num_sample=self.num_sample,
                    max_steps_per_epch=self.max_steps_per_epch,
                    num_classes=self.num_classes,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    z_dim=self.z_dim,
                    alpha=self.alpha,
                    beta=self.beta
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
        print("[Server] Starting weighted FedSR aggregation...")
        total_samples = sum(total_samples_list)
        
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

    def train(self, target_domain, checkpoint_path):
        print(f"\n[Server] Commencing FL for {self.num_rounds} rounds.")
        global_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="FL Round", position=0)

        for round_idx in round_pbar:
            print(f"\n--- Round {round_idx + 1}/{self.num_rounds} ---")
            
            job_ids = [
                client.train.remote(global_parameters=global_weights) 
                for client in self.clients
            ]

            results = ray.get(job_ids)
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]

            global_weights = self.aggregate(
                local_weights_list=local_weights_list, 
                total_samples_list=total_samples_list
            )

            self.backbone_model.load_state_dict(global_weights)
            
            print(f"[Server] Evaluating Round {round_idx + 1}...")
            miou, pixel_acc, _ = self.evaluate(target_domain=target_domain, checkpoint_path=None)
            
            wandb.log({
                "Round": round_idx + 1,
                "Round_Test_mIoU": miou * 100,
                "Round_Test_Pixel_Accuracy": pixel_acc * 100
            })

            round_pbar.set_description(f"Round {round_idx + 1} | mIoU: {miou*100:.2f}%")

        print(f"\n[Server] Training complete. Saving model to {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model
    
    def evaluate(self, target_domain, checkpoint_path=None):
        print("\n" + "="*50)
        print(f"[Server] Evaluating on Target Domain: {target_domain}")

        if checkpoint_path is not None:
            self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[Server] Loaded checkpoint from {checkpoint_path}")
            
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        if not hasattr(self, 'test_dataloader'):
            print(f"[Server] Loading {target_domain} evaluation dataset (First time only)...")
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
                    list_of_paths=["dataset/gta5/gta5_part10"],
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
                    num_sample=eval_num_sample
                )
            
            self.test_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False, 
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        print(f"[Server] Running inference on {len(self.test_dataloader)} batches...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Inference"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.backbone_model(images)
                preds = torch.argmax(outputs, dim=1)
                
                valid_mask = (masks != 255)
                target = masks[valid_mask]
                predict = preds[valid_mask]
                
                indices = self.num_classes * target + predict
                conf_matrix += torch.bincount(
                    indices, minlength=self.num_classes**2
                ).reshape(self.num_classes, self.num_classes)

        tp = torch.diag(conf_matrix)
        fp = torch.sum(conf_matrix, dim=0) - tp
        fn = torch.sum(conf_matrix, dim=1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou = torch.mean(iou_per_class).item()
        pixel_acc = (torch.sum(tp) / (torch.sum(conf_matrix) + 1e-10)).item()

        print("\n" + "="*40)
        print(f"Results for {target_domain}:")
        print(f"- mIoU: {miou*100:.2f}%")
        print(f"- Pixel Accuracy: {pixel_acc*100:.2f}%")
        print("="*40)
        
        return miou, pixel_acc, iou_per_class