import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AugmentedSegmentationDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        dataset_names=None,
        styles=None,       
        image_size=(512, 512), 
        image_transform=None, 
        mask_transform=None
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        self.samples = []
        
        if dataset_names is None:
            dataset_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
        print(f"Loading datasets: {dataset_names}")
        
        for ds_name in dataset_names:
            ds_dir = os.path.join(root_dir, ds_name)
            if not os.path.isdir(ds_dir):
                print(f"Warning: Dataset directory '{ds_dir}' not found. Skipping.")
                continue

            if styles is None:
                current_styles = [d for d in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, d))]
            else:
                current_styles = styles

            for style in current_styles:
                img_dir = os.path.join(ds_dir, style, "images")
                lbl_dir = os.path.join(ds_dir, style, "labels")
                orig_dir = os.path.join(ds_dir, style, "originals") 
                
                if not os.path.exists(img_dir) or not os.path.exists(lbl_dir) or not os.path.exists(orig_dir):
                    print(f"Warning: Missing images/labels/originals dir for {ds_name}/{style}. Skipping.")
                    continue
                    
                for img_name in os.listdir(img_dir):
                    if not img_name.endswith("_img.png"):
                        continue
                        
                    base_name = img_name.replace("_img.png", "")
                    lbl_name = f"{base_name}_label.png"
                    orig_name = f"{base_name}_orig.png" 
                    
                    syn_img_path = os.path.join(img_dir, img_name)
                    lbl_path = os.path.join(lbl_dir, lbl_name)
                    orig_path = os.path.join(orig_dir, orig_name) 
                    
                    if os.path.exists(lbl_path) and os.path.exists(orig_path):
                        self.samples.append((orig_path, syn_img_path, lbl_path))
                    else:
                        print(f"Warning: Missing label or original image for {ds_name}/{style}/{img_name}")

        print(f"Successfully loaded {len(self.samples)} paired samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        orig_path, syn_path, lbl_path = self.samples[idx]
        
        orig_image = Image.open(orig_path).convert("RGB") 
        syn_image = Image.open(syn_path).convert("RGB")   
        mask = Image.open(lbl_path)                       
        
        if self.image_size:
            orig_image = orig_image.resize(self.image_size, Image.BILINEAR)
            syn_image = syn_image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
            
        if self.image_transform:
            orig_image = self.image_transform(orig_image)
            syn_image = self.image_transform(syn_image)
        else:
            orig_image = transforms.ToTensor()(orig_image)
            syn_image = transforms.ToTensor()(syn_image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()
            
        return orig_image, syn_image, mask

def get_augmented_dataloader(
    root_dir="dataset/augment_data", 
    dataset_names=None, 
    styles=None, 
    batch_size=8, 
    shuffle=True, 
    num_workers=4
):
    dataset = AugmentedSegmentationDataset(
        root_dir=root_dir,
        dataset_names=dataset_names,
        styles=styles,
        image_size=(512, 512),
        image_transform=None, 
        mask_transform=None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True 
    )
    
    return dataloader
