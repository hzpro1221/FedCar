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
from ray.util.actor_pool import ActorPool
import wandb 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .silobn_lab_client import SiloBN_LAB_Client, is_bn_statistic

class SiloBN_LAB_Server(FedAvg_Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} SiloBN_LAB workers via Ray ActorPool...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                SiloBN_LAB_Client.remote(
                    local_model=copy.deepcopy(self.backbone_model),
                    num_sample=self.num_sample,
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    max_steps_per_epch=self.max_steps_per_epch,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    **kwargs 
                )
            )
        return workers

    def aggregate(self, local_weights_list, total_samples_list):
        print("[Server] Starting SiloBN aggregation...")
        total_samples = sum(total_samples_list)
        
        avg_weights = {k: v.cpu() for k, v in self.backbone_model.state_dict().items()}
        
        for key in avg_weights.keys():
            if not is_bn_statistic(key):
                avg_weights[key] = torch.zeros_like(avg_weights[key])

        for i in range(len(local_weights_list)):
            local_w = local_weights_list[i]
            n_k = total_samples_list[i]
            weight_factor = n_k / total_samples

            for key in avg_weights.keys():
                if not is_bn_statistic(key) and key in local_w:
                    target_device = avg_weights[key].device
                    avg_weights[key] += (local_w[key].to(target_device) * weight_factor).to(avg_weights[key].dtype)
        
        print("[Server] Aggregation complete.")
        return avg_weights

    def update_global_model(self, aggregated_weights):
        self.backbone_model.load_state_dict(aggregated_weights, strict=False)

    def warm_up_norm_layers(self, dataloader, warmup_steps=50):
        print(f"[Server] Warming up Batch Norm statistics for {warmup_steps} steps...")
        self.backbone_model.train() 
        
        with torch.no_grad(): 
            for step, (images, _) in enumerate(dataloader):
                if step >= warmup_steps:
                    break
                images = images.to(self.device)
                _ = self.backbone_model(images)
        print("[Server] Batch Norm Warm-up complete.")
    
    def evaluate(self, target_domain, checkpoint_path=None, apply_bn_warmup=True):
        print("\n" + "="*50)
        print(f"[Server] Starting evaluation on Target Domain: {target_domain}")

        if checkpoint_path is not None:
            self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[Server] Loaded checkpoint from {checkpoint_path}")
            
        self.backbone_model.to(self.device)

        if not hasattr(self, 'test_dataloader'):
            print(f"[Server] Loading dataset for {target_domain} (First time only)...")
            eval_num_sample = int(self.num_sample / 10) if self.num_sample is not None else None
            dataset = None
            
            if target_domain == 'cityscape':
                dataset = CityscapesDataset("dataset/cityscape/leftImg8bit/val", "dataset/cityscape/gtFine/val", num_sample=eval_num_sample)
            elif target_domain == "bdd100":
                dataset = BDD100KDataset("dataset/bdd100/10k/val", "dataset/bdd100/labels/val", num_sample=eval_num_sample)
            elif target_domain == "gta5":
                dataset = GTA5Dataset(list_of_paths=["dataset/gta5/gta5_part8", "dataset/gta5/gta5_part9", "dataset/gta5/gta5_part10"], num_sample=eval_num_sample)
            elif target_domain == "mapillary":
                dataset = MapillaryDataset("dataset/mapillary/validation", num_sample=eval_num_sample) 
            elif target_domain == "synthia":
                dataset = SynthiaDataset("dataset/synthia/RAND_CITYSCAPES", start_index=6580, end_index=None, num_sample=eval_num_sample)
            else:
                raise ValueError(f"Unknown target domain: {target_domain}")
            
            self.test_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True 
            )
            
        print(f"[Server] Total batches to evaluate: {len(self.test_dataloader)}")
        
        if apply_bn_warmup:
            warmup_steps = min(50, len(self.test_dataloader))
            self.warm_up_norm_layers(self.test_dataloader, warmup_steps)

        self.backbone_model.eval() 
        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        print("[Server] Starting inference loop...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.backbone_model(images) 
                
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                preds = torch.argmax(logits, dim=1) 

                mask_valid = (masks != 255)
                
                target = masks[mask_valid]
                predict = preds[mask_valid]
                
                indices = self.num_classes * target + predict
                conf_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

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
            
        return miou, pixel_acc, iou_per_class