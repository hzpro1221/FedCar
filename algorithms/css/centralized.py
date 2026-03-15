import sys
import os
project_root = "/root/KhaiDD/FedCar"
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
import copy
from tqdm import tqdm

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

from .segformer_b0_centralized import SegFormerB0_Centralized

class Centralized:
    def __init__(
        self, 
        num_classes,
        source_domains,
        num_epochs, 
        batch_size,
        init_lr,
        min_lr,
        power,
        weight_decay,
        max_steps_per_epch=50 # 10 * num_domain
    ):
        print("\n" + "="*50)
        print("[Trainer] Initializing Centralized (Deep All) Trainer...")
        self.num_classes = num_classes

        self.backbone_model = SegFormerB0_Centralized(num_classes=self.num_classes)
        
        self.source_domains = source_domains
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.max_steps_per_epch = max_steps_per_epch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Global device set to: {self.device}")
        print(f"[Trainer] Source domains to COMBINE: {self.source_domains}")
        print(f"[Trainer] Total Epochs: {self.num_epochs}")

        self.backbone_model.to(self.device)

        print("[Trainer] Loading and concatenating datasets...")
        datasets_list = []
        for domain in self.source_domains:
            if domain == 'cityscape':
                datasets_list.append(CityscapesDataset(
                    images_dir="/root/KhaiDD/FedCar/dataset/cityscape/leftImg8bit/train",
                    labels_dir="/root/KhaiDD/FedCar/dataset/cityscape/gtFine/train"
                ))
            elif domain == "bdd100":
                datasets_list.append(BDD100KDataset(
                    images_dir="/root/KhaiDD/FedCar/dataset/bdd100/10k/train",
                    labels_dir="/root/KhaiDD/FedCar/dataset/bdd100/labels/train"
                ))
            elif domain == "gta5":
                datasets_list.append(GTA5Dataset(
                    list_of_paths=[
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part1",
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part2",
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part3",
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part4",
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part5",
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part6",
                        "/root/KhaiDD/FedCar/dataset/gta5/gta5_part7",
                    ]
                ))
            elif domain == "mapillary":
                datasets_list.append(MapillaryDataset(
                    root_dir="/root/KhaiDD/FedCar/dataset/mapillary/training"
                ))
            elif domain == "synthia":
                datasets_list.append(SynthiaDataset(
                    root_dir="/root/KhaiDD/FedCar/dataset/synthia/RAND_CITYSCAPES",
                    start_index=0,
                    end_index=6580
                ))

        # Concatenate all dataset into one
        self.combined_dataset = ConcatDataset(datasets_list)
        
        self.train_dataloader = DataLoader(
            self.combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        print(f"[Trainer] Combined Dataset Size: {len(self.combined_dataset)} images.")
        print(f"[Trainer] Total batches per epoch: {len(self.train_dataloader)}")
        print("="*50 + "\n")

        self.optimizer = optim.AdamW(
            self.backbone_model.parameters(), 
            lr=self.init_lr, 
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        total_iters = self.num_epochs * len(self.train_dataloader)
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=total_iters, 
            power=self.power
        )

    def set_seed(self, seed): 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[Trainer] Global seed set to {seed}.")

    def train(self, checkpoint_path):
        print(f"\n[Trainer] Starting Centralized Training for {self.num_epochs} epochs.")
        self.backbone_model.train()

        epoch_pbar = tqdm(range(self.num_epochs), desc="Epochs", position=0)

        for epoch in epoch_pbar:
            running_loss = 0.0
            
            batch_pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
            
            for step, (images, masks) in enumerate(batch_pbar):
                if (step > self.max_steps_per_epch):
                    break
                    
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.backbone_model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                self.scheduler.step() 
                
                running_loss += loss.item()
                batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_epoch_loss = running_loss / min(self.max_steps_per_epch + 1, len(self.train_dataloader))
            epoch_pbar.set_postfix({"Avg Loss": f"{avg_epoch_loss:.4f}"})

        print(f"\n[Trainer] Training complete. Saving global model to {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model
    
    def evaluate(self, target_domain, checkpoint_path):
        print("\n" + "="*50)
        print(f"[Trainer] Starting evaluation on Target Domain: {target_domain}")

        self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        print(f"[Trainer] Loaded checkpoint from {checkpoint_path}")

        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        print(f"[Trainer] Loading dataset for {target_domain}...")
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
            num_workers=2,
            pin_memory=True
        )
        print(f"[Trainer] Dataset loaded. Total batches to evaluate: {len(self.test_dataloader)}")
        
        print("[Trainer] Starting inference loop...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.backbone_model(images)

                preds = torch.argmax(outputs, dim=1)
                
                mask_valid = (masks != 255)
                target = masks[mask_valid]
                predict = preds[mask_valid]
                
                indices = self.num_classes * target + predict
                conf_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

        print("[Trainer] Calculating metrics from confusion matrix...")
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