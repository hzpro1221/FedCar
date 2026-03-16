import sys
import os
import random

project_root = "/root/KhaiDD/FedCar"
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

from .segformer_b0_sens_aug import SegFormerB0_SensAug

PERTURBATIONS_14 = [
    "blur", "noise",
    "lighter_R", "lighter_G", "lighter_B",
    "darker_R", "darker_G", "darker_B",
    "lighter_H", "lighter_S", "lighter_V",
    "darker_H", "darker_S", "darker_V"
]

def apply_perturbation(image, p_type, level):
    if p_type == "none":
        return image
        
    # ==========================================
    # Group 1: Blur & Noise
    # ==========================================
    if p_type == "blur":
        kernel_size = int(level * 2 + 1)
        sigma = 0.5 + level * 0.5
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        
    elif p_type == "noise":
        std = level * 0.05
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)
        
    # ==========================================
    # Group 2: Interfere RGB 
    # ==========================================
    img_out = image.clone()
    intensity = level * 0.1 # Max shift = 0.5 at level 5
    
    if p_type == "lighter_R":
        img_out[0] = torch.clamp(img_out[0] + intensity, 0.0, 1.0)
        return img_out
    elif p_type == "lighter_G":
        img_out[1] = torch.clamp(img_out[1] + intensity, 0.0, 1.0)
        return img_out
    elif p_type == "lighter_B":
        img_out[2] = torch.clamp(img_out[2] + intensity, 0.0, 1.0)
        return img_out
    elif p_type == "darker_R":
        img_out[0] = torch.clamp(img_out[0] - intensity, 0.0, 1.0)
        return img_out
    elif p_type == "darker_G":
        img_out[1] = torch.clamp(img_out[1] - intensity, 0.0, 1.0)
        return img_out
    elif p_type == "darker_B":
        img_out[2] = torch.clamp(img_out[2] - intensity, 0.0, 1.0)
        return img_out
        
    # ==========================================
    # Nhóm 3: Interfere HSV space
    # ==========================================
    # Hue: in between [-0.5, 0.5]
    hue_shift = level * 0.1 
    if p_type == "lighter_H":
        return TF.adjust_hue(image, hue_shift)
    elif p_type == "darker_H":
        return TF.adjust_hue(image, -hue_shift)
        
    # Saturation 
    if p_type == "lighter_S":
        factor = 1.0 + (level * 0.4)
        return TF.adjust_saturation(image, factor)
    elif p_type == "darker_S":
        factor = max(0.0, 1.0 - (level * 0.2))
        return TF.adjust_saturation(image, factor)
        
    # Value 
    if p_type == "lighter_V":
        factor = 1.0 + (level * 0.3)
        return TF.adjust_brightness(image, factor)
    elif p_type == "darker_V":
        factor = max(0.0, 1.0 - (level * 0.15))
        return TF.adjust_brightness(image, factor)

    return image

class SensAugDatasetWrapper(torch.utils.data.Dataset):
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

class SensAug:
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
        max_steps_per_epch=10, 
        max_sa_batches=1
    ):
        print("\n" + "="*50)
        print("[Trainer] Initializing SensAug Trainer...")
        self.num_classes = num_classes

        self.backbone_model = SegFormerB0_SensAug(num_classes=self.num_classes)
        
        self.source_domains = source_domains
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.max_steps_per_epch = max_steps_per_epch * len(self.source_domains)
        self.max_sa_batches = max_sa_batches

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
        
        # Split 10% for validation
        total_size = len(self.combined_dataset)
        sa_val_size = int(0.1 * total_size)
        train_size = total_size - sa_val_size

        self.pure_train_dataset, self.sa_val_dataset = random_split(
            self.combined_dataset, 
            [train_size, sa_val_size]
        )

        print(f"[SensAug] Split dataset: {train_size} for Training, {sa_val_size} for SA Validation.")
        self.sensaug_train_dataset = SensAugDatasetWrapper(self.pure_train_dataset)

        self.train_dataloader = DataLoader(
            self.sensaug_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.sa_val_dataloader = DataLoader(
            self.sa_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
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

    def run_sensitivity_analysis(self):
        print("\n[SensAug] Running Sensitivity Analysis...")
        self.backbone_model.eval()
        
        levels = [1, 3, 5] 
        
        miou_record = {}        

        with torch.no_grad():
            for p_type in PERTURBATIONS_14:
                for level in levels:
                    conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)
                    
                    for step, (images, masks) in enumerate(self.sa_val_dataloader):
                        if step >= self.max_sa_batches:
                            break
                            
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        
                        perturbed_images = apply_perturbation(images, p_type, level)
                        
                        outputs = self.backbone_model(perturbed_images)
                        preds = torch.argmax(outputs, dim=1)
                        
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
                    
                    miou_record[(p_type, level)] = miou
                    print(f"  -> {p_type} (lvl {level}): mIoU = {miou:.4f}")

        # weight = (1.0 - mIoU)^2 
        pdf_dict = {}
        weight_sum = sum([(1.0 - m)**2 for m in miou_record.values()])
        
        # Keep 30% original data
        prob_clean = 0.3 
        prob_perturb = 1.0 - prob_clean
        
        for (p_type, level), miou in miou_record.items():
            weight = (1.0 - miou)**2
            pdf_dict[(p_type, level)] = (weight / weight_sum) * prob_perturb
            
        pdf_dict[("none", 0)] = prob_clean
        
        print("[SensAug] New Augmentation PDF computed:")
        for k, v in pdf_dict.items():
            if v > 0.05: 
                print(f"  - {k[0]} lvl {k[1]}: {v*100:.1f}%")

        self.sensaug_train_dataset.update_pdf(pdf_dict)
        self.backbone_model.train()

    def train(self, checkpoint_path):
        print(f"\n[Trainer] Starting Centralized Training for {self.num_epochs} epochs.")
        
        self.run_sensitivity_analysis()
        
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
            
            if epoch < self.num_epochs - 1:
                self.run_sensitivity_analysis()

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
                    # "/root/KhaiDD/FedCar/dataset/gta5/gta5_part8",
                    # "/root/KhaiDD/FedCar/dataset/gta5/gta5_part9",
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