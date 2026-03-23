import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
from tqdm import tqdm

from segformer_b0_centralized import SegFormerB0_Centralized

from algorithms.dataset_pytorch import (
    BDD100KDataset, CityscapesDataset, GTA5Dataset, 
    MapillaryDataset, SynthiaDataset
)

class CentralizedTrainer:
    def __init__(
        self, 
        num_classes,
        source_domains,
        dataset_root,
        
        num_epochs, 
        batch_size,
        num_sample, 
        max_steps_per_epch,
        
        init_lr,
        min_lr,
        power,
        weight_decay,
        num_workers,
        seed
    ):
        """
        Centralized Training Baseline for Domain Generalization.
        Combines multiple source domains into a single training pipeline.
        """
        print("\n" + "="*50)
        print("[Trainer] Initializing Centralized Trainer...")
        
        self.num_classes = num_classes
        self.source_domains = source_domains
        self.dataset_root = dataset_root
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_sample = num_sample 
        self.max_steps_per_epch = max_steps_per_epch

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed:
            self.set_seed(seed)

        self.backbone_model = SegFormerB0_Centralized(
            num_classes=self.num_classes
        )
        self.backbone_model.to(self.device)

        self._prepare_datasets()
        
        print(f"[Trainer] Global device set to: {self.device}")
        print(f"[Trainer] Samples per domain: {self.num_sample if self.num_sample else 'Full'}")
        print("="*50 + "\n")

    def set_seed(self, seed):
        """Ensures deterministic behavior across runs."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print(f"[Trainer] Global seed set to {seed}.")

    def _prepare_datasets(self):
        """
        Initializes and concatenates training datasets from all source domains.
        Explicitly passes num_sample to each dataset for balanced training.
        """
        print(f"[Trainer] Aggregating source datasets...")
        datasets_list = []
        
        for domain in self.source_domains:
            path = os.path.join(self.dataset_root, domain)
            
            if domain == 'cityscape':
                datasets_list.append(CityscapesDataset(
                    images_dir=os.path.join(path, "leftImg8bit/train"),
                    labels_dir=os.path.join(path, "gtFine/train"),
                    num_sample=self.num_sample
                ))
            elif domain == "bdd100":
                datasets_list.append(BDD100KDataset(
                    images_dir=os.path.join(path, "10k/train"),
                    labels_dir=os.path.join(path, "labels/train"),
                    num_sample=self.num_sample
                ))
            elif domain == "gta5":
                datasets_list.append(GTA5Dataset(
                    list_of_paths=[os.path.join(path, f"gta5_part{i}") for i in range(1, 8)],
                    num_sample=self.num_sample
                ))
            elif domain == "mapillary":
                datasets_list.append(MapillaryDataset(
                    root_dir=os.path.join(path, "training"),
                    num_sample=self.num_sample
                ))
            elif domain == "synthia":
                datasets_list.append(SynthiaDataset(
                    root_dir=os.path.join(path, "RAND_CITYSCAPES"),
                    start_index=0,
                    end_index=6580,
                    num_sample=self.num_sample
                ))

        self.combined_dataset = ConcatDataset(datasets_list)
        self.train_dataloader = DataLoader(
            self.combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.optimizer = optim.AdamW(
            self.backbone_model.parameters(), 
            lr=self.init_lr, 
            weight_decay=self.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters= self.num_epochs * self.max_steps_per_epch, 
            power=self.power
        )

        print(f"[Trainer] Training pool size: {len(self.combined_dataset)} images.")

    def train(self, checkpoint_path):
        """Executes the full centralized training loop."""
        print(f"[Trainer] Starting training for {self.num_epochs} epochs.")
        
        for epoch in range(self.num_epochs):
            self.backbone_model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            for step, (images, masks) in enumerate(pbar):
                if step >= self.max_steps_per_epch:
                    break

                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.backbone_model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() 
                
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"[Trainer] Training finished. Model saved at: {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model
    
    def evaluate(self, target_domain, checkpoint_path):
        """
        Evaluates the trained model on a specific target domain.
        """
        print(f"\n[Trainer] Evaluating target domain: {target_domain}")
        
        self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.backbone_model.eval()

        path = os.path.join(self.dataset_root, target_domain)
        dataset = None
        
        if target_domain == 'cityscape':
            dataset = CityscapesDataset(
                images_dir=os.path.join(path, "leftImg8bit/val"),
                labels_dir=os.path.join(path, "gtFine/val"),
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "bdd100":
            dataset = BDD100KDataset(
                images_dir=os.path.join(path, "10k/val"),
                labels_dir=os.path.join(path, "labels/val"),
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "gta5":
            dataset = GTA5Dataset(
                list_of_paths=[
                    os.path.join(path, "gta5_part8"),
                    os.path.join(path, "gta5_part9"),
                    os.path.join(path, "gta5_part10")
                ],
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "mapillary":
            dataset = MapillaryDataset(
                root_dir=os.path.join(path, "validation"),
                num_sample=int(self.num_sample / 10)
            ) 
        elif target_domain == "synthia":
            dataset = SynthiaDataset(
                root_dir=os.path.join(path, "RAND_CITYSCAPES"),
                start_index=6580,
                num_sample=int(self.num_sample / 10)
            )

        test_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )
        
        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc=f"Eval {target_domain}"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.backbone_model(images)
                preds = torch.argmax(outputs, dim=1)
                
                valid = (masks != 255)
                # Calculate confusion matrix for mIoU
                indices = self.num_classes * masks[valid] + preds[valid]
                conf_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

        # Performance Metrics
        tp = torch.diag(conf_matrix)
        fp = torch.sum(conf_matrix, dim=0) - tp
        fn = torch.sum(conf_matrix, dim=1) - tp
        
        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou = torch.mean(iou_per_class).item()
        pixel_acc = (torch.sum(tp) / (torch.sum(conf_matrix) + 1e-10)).item()
        
        print(f"[Trainer] Result for {target_domain}: mIoU = {miou*100:.2f}%, Pixel Acc = {pixel_acc*100:.2f}%")
        return miou, pixel_acc