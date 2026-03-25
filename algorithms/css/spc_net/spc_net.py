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
import wandb 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset
from .segformer_b0_spc_net import SegFormerB0_SPC_Net

class SPC_Net:
    def __init__(
        self, 
        num_classes,
        source_domains,
        
        num_epochs, 
        batch_size,
        num_sample,          
        max_steps_per_epch,    
        num_workers,

        init_lr,
        min_lr,
        power,
        weight_decay,
        ema_decay=0.999, 
        dataset_root="dataset" 
    ):
        print("\n" + "="*50)
        print("[Trainer] Initializing SPC_Net Trainer...")
        self.num_classes = num_classes
        self.source_domains = source_domains
        self.dataset_root = dataset_root
        self.num_datasets = len(self.source_domains)
        self.ema_decay = ema_decay
        
        self.backbone_model = SegFormerB0_SPC_Net(
            num_classes=self.num_classes, 
            num_datasets=self.num_datasets
        )        
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_sample = num_sample
        self.max_steps_per_epch = max_steps_per_epch
        self.num_workers = num_workers
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_model.to(self.device)

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Domains: {self.source_domains} | Samples per domain: {self.num_sample}")

        self._prepare_datasets()

        self.optimizer = optim.AdamW(
            self.backbone_model.parameters(), 
            lr=self.init_lr, 
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        total_iters = self.num_epochs * self.max_steps_per_epch
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
        torch.backends.cudnn.deterministic = True
        print(f"[Trainer] Seed set to {seed}")

    def _prepare_datasets(self):
        print("[Trainer] Loading datasets...")
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
        print(f"[Trainer] Total combined images: {len(self.combined_dataset)}")
        print("="*50 + "\n")

    def train(self, target_domain, checkpoint_path):
        print(f"[Trainer] Starting training for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            self.backbone_model.train()
            running_loss = 0.0
            num_steps = 0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for step, (images, masks) in enumerate(pbar):
                if step >= self.max_steps_per_epch:
                    break
                    
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.backbone_model(images, masks)
                
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() 
                
                running_loss += loss.item()
                num_steps += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{self.scheduler.get_last_lr()[0]:.6f}"})

            avg_train_loss = running_loss / max(num_steps, 1)

            print(f"\n[Trainer] Evaluating Epoch {epoch + 1}...")
            miou, pixel_acc, _ = self.evaluate(target_domain=target_domain, checkpoint_path=None)
            
            wandb.log({
                "Epoch": epoch + 1,
                "Train_Loss": avg_train_loss,
                "Epoch_Test_mIoU": miou * 100,
                "Epoch_Test_Pixel_Accuracy": pixel_acc * 100
            })

        print(f"[Trainer] Saving model to {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model
    
    def evaluate(self, target_domain, checkpoint_path=None):
        print("\n" + "="*50)
        print(f"[Trainer] Starting evaluation on Target Domain: {target_domain}")

        if checkpoint_path is not None:
            self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[Trainer] Loaded checkpoint from {checkpoint_path}")
            
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

        if not hasattr(self, 'test_dataloader'):
            print(f"[Trainer] Loading dataset for {target_domain} (First time only)...")
            path = os.path.join(self.dataset_root, target_domain)
            dataset = None
            eval_n = int(self.num_sample / 10) if self.num_sample is not None else None
            
            if target_domain == 'cityscape':
                dataset = CityscapesDataset(
                    images_dir=os.path.join(path, "leftImg8bit/val"),
                    labels_dir=os.path.join(path, "gtFine/val"),
                    num_sample=eval_n
                )
            elif target_domain == "bdd100":
                dataset = BDD100KDataset(
                    images_dir=os.path.join(path, "10k/val"),
                    labels_dir=os.path.join(path, "labels/val"),
                    num_sample=eval_n
                )
            elif target_domain == "gta5":
                dataset = GTA5Dataset(
                    list_of_paths=[
                        os.path.join(path, "gta5_part8"),
                        os.path.join(path, "gta5_part9"),
                        os.path.join(path, "gta5_part10")
                    ],
                    num_sample=eval_n
                )
            elif target_domain == "mapillary":
                dataset = MapillaryDataset(
                    root_dir=os.path.join(path, "validation"),
                    num_sample=eval_n
                ) 
            elif target_domain == "synthia":
                dataset = SynthiaDataset(
                    root_dir=os.path.join(path, "RAND_CITYSCAPES"),
                    start_index=6580,
                    end_index=None,
                    num_sample=eval_n
                )
            
            self.test_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False, 
                num_workers=self.num_workers,
                pin_memory=True
            )
            
        print(f"[Trainer] Total batches to evaluate: {len(self.test_dataloader)}")
        
        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

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