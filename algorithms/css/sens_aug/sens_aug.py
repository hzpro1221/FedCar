import sys
import os
import random

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
import copy
from tqdm import tqdm

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

from segformer_b0_sens_aug import SegFormerB0_SensAug

# perturbation list
REDUCED_PERTURBATIONS = ["blur", "noise", "brightness", "saturation", "hue"]

def apply_perturbation(image, p_type, level):
    """
    Applies a specific visual perturbation to a tensor image [C, H, W].
    'level' typically scales from 1 (mild) to 5 (severe).
    """
    if p_type == "none": return image

    if p_type == "blur":
        k_size = int(level * 2 + 1)
        return TF.gaussian_blur(image, [k_size, k_size], [0.5 + level * 0.5, 0.5 + level * 0.5])
    
    elif p_type == "noise":
        return torch.clamp(image + torch.randn_like(image) * (level * 0.05), 0.0, 1.0)
    
    elif p_type == "brightness":
        # level 1-5 -> factor 0.7 to 1.3 (darker/lighter)
        factor = 1.0 + (level * 0.06) if level > 0 else 1.0 + (level * 0.06)
        return TF.adjust_brightness(image, factor)

    elif p_type == "saturation":
        return TF.adjust_saturation(image, 1.0 + (level * 0.4))

    elif p_type == "hue":
        return TF.adjust_hue(image, level * 0.05)

    return image

class SensAugDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper that applies dynamic perturbations based on a computed PDF."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.pdf_keys = [("none", 0)]
        self.pdf_weights = [1.0]

    def update_pdf(self, pdf_dict):
        self.pdf_keys = list(pdf_dict.keys())
        self.pdf_weights = list(pdf_dict.values())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        chosen_p = random.choices(self.pdf_keys, weights=self.pdf_weights, k=1)[0]
        p_type, level = chosen_p
        
        if p_type != "none":
            image = apply_perturbation(image, p_type, level)
        return image, mask

class SensAugTrainer:
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
        seed,

        sa_val_ratio, 
        max_sa_batches, 
        prob_clean
    ):
        """
        SensAug Trainer: Periodically analyzes model sensitivity to various image 
        perturbations and adjusts data augmentation probability accordingly.
        """
        print("\n" + "="*50)
        print("[SensAug] Initializing Sensitivity-Aware Trainer...")
        
        self.num_classes = num_classes
        self.source_domains = source_domains
        self.dataset_root = dataset_root
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_sample = num_sample
        self.max_steps_per_epch = max_steps_per_epch

        self.max_sa_batches = max_sa_batches
        self.prob_clean = prob_clean
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if seed: self.set_seed(seed)

        self.backbone_model = SegFormerB0_SensAug(num_classes=self.num_classes)
        self.backbone_model.to(self.device)

        # Dataset initialization with SA split
        self._prepare_datasets(sa_val_ratio)
        
        print(f"[SensAug] Setup Complete. Clean Image Prob: {self.prob_clean*100}%")
        print("="*50 + "\n")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _prepare_datasets(self, sa_val_ratio):
        """Prepares combined dataset and splits a portion for sensitivity analysis."""
        datasets = []
        for domain in self.source_domains:
            path = os.path.join(self.dataset_root, domain)
            if domain == 'cityscape':
                datasets.append(CityscapesDataset(
                    os.path.join(path, "leftImg8bit/train"), 
                    os.path.join(path, "gtFine/train"), 
                    num_sample=self.num_sample
                ))
            elif domain == 'bdd100':
                datasets.append(BDD100KDataset(
                    os.path.join(path, "10k/train"), 
                    os.path.join(path, "labels/train"), 
                    num_sample=self.num_sample
                ))
            elif domain == 'gta5':
                datasets.append(GTA5Dataset([
                    os.path.join(path, f"gta5_part{i}") for i in range(1, 8)], 
                    num_sample=self.num_sample
                ))
            elif domain == 'mapillary':
                datasets.append(MapillaryDataset(
                    os.path.join(path, "training"), 
                    num_sample=self.num_sample
                ))
            elif domain == 'synthia':
                datasets.append(SynthiaDataset(
                    os.path.join(path, "RAND_CITYSCAPES"), 
                    start_index=0, 
                    end_index=6580, 
                    num_sample=self.num_sample
                ))

        combined = ConcatDataset(datasets)
        val_size = int(sa_val_ratio * len(combined))
        train_size = len(combined) - val_size
        
        self.raw_train_set, self.sa_val_set = random_split(combined, [train_size, val_size])
        self.sa_train_wrapper = SensAugDatasetWrapper(self.raw_train_set)

        self.train_dataloader = DataLoader(self.sa_train_wrapper, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.sa_val_dataloader = DataLoader(self.sa_val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.backbone_model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.num_epochs * self.max_steps_per_epch, power=self.power)

    def run_sensitivity_analysis(self):
        """Evaluates model vulnerability to perturbations and updates augmentation PDF."""
        print("[SensAug] Analyzing model sensitivity...")
        self.backbone_model.eval()
        miou_record = {}
        levels = [1, 3, 5]

        with torch.no_grad():
            for p_type in REDUCED_PERTURBATIONS:
                for level in levels:
                    conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)
                    for step, (images, masks) in enumerate(self.sa_val_dataloader):
                        if step >= self.max_sa_batches: break
                        
                        images, masks = images.to(self.device), masks.to(self.device)
                        perturbed = apply_perturbation(images, p_type, level)
                        outputs = self.backbone_model(perturbed)
                        preds = torch.argmax(outputs, dim=1)
                        
                        valid = (masks != 255)
                        indices = self.num_classes * masks[valid] + preds[valid]
                        conf_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
                    
                    tp = torch.diag(conf_matrix)
                    iou = tp / (tp + torch.sum(conf_matrix, dim=0) + torch.sum(conf_matrix, dim=1) - tp + 1e-10)
                    miou_record[(p_type, level)] = torch.mean(iou).item()

        weights = np.array([(1.0 - m)**2 for m in miou_record.values()])
        pdf_vals = (weights / weights.sum()) * (1.0 - self.prob_clean)
        
        pdf_dict = {k: v for k, v in zip(miou_record.keys(), pdf_vals)}
        pdf_dict[("none", 0)] = self.prob_clean
        
        self.sa_train_wrapper.update_pdf(pdf_dict)
        print(f"[SensAug] Top Sensitivity: {max(miou_record, key=miou_record.get)} (mIoU: {min(miou_record.values()):.4f})")

    def train(self, checkpoint_path):
        """Main training loop with periodic sensitivity analysis."""
        for epoch in range(self.num_epochs):
            self.run_sensitivity_analysis()
            
            self.backbone_model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, (images, masks) in enumerate(pbar):
                if step >= self.max_steps_per_epch: break
                
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.backbone_model(images), masks)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

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
                labels_dir="/root/KhaiDD/FedCar/dataset/cityscape/gtFine/val",
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "bdd100":
            dataset = BDD100KDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/bdd100/10k/val",
                labels_dir="/root/KhaiDD/FedCar/dataset/bdd100/labels/val",
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "gta5":
            dataset = GTA5Dataset(
                list_of_paths=[
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part8",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part9",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part10",
                    num_sample=int(self.num_sample / 10)
                ]
            )
        elif target_domain == "mapillary":
            dataset = MapillaryDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/mapillary/validation",
                num_sample=int(self.num_sample / 10)
            ) 
        elif target_domain == "synthia":
            dataset = SynthiaDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/synthia/RAND_CITYSCAPES",
                start_index=6580,
                end_index=None,
                num_sample=int(self.num_sample / 10)
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