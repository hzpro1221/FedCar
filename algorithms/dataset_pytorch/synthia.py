import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2

class SynthiaDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        image_size=(512, 512),
        start_index=None,
        end_index=None,
        num_sample=None 
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        
        self.images_dir = os.path.join(root_dir, 'RGB')
        self.labels_dir = os.path.join(root_dir, 'GT', 'LABELS')
        
        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Error: Didn't find RGB/ or GT/LABELS/ in {root_dir}")

        self.image_paths = []
        self.mask_paths = []

        all_filenames = sorted(os.listdir(self.images_dir))
        selected_filenames = all_filenames[start_index:end_index]

        for file_name in all_filenames:
            if file_name.endswith('.png'):
                img_path = os.path.join(self.images_dir, file_name)
                mask_path = os.path.join(self.labels_dir, file_name) 
                
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                
        self.image_paths = self.image_paths[:num_sample]
        self.mask_paths = self.mask_paths[:num_sample]

        self.mapping_256 = np.full(256, 255, dtype=np.int64) 
        
        self.mapping_256[3]  = 0  # road
        self.mapping_256[4]  = 1  # sidewalk
        self.mapping_256[2]  = 2  # building
        self.mapping_256[21] = 3  # wall
        self.mapping_256[5]  = 4  # fence
        self.mapping_256[7]  = 5  # pole
        self.mapping_256[15] = 6  # traffic light
        self.mapping_256[9]  = 7  # traffic sign
        self.mapping_256[6]  = 8  # vegetation
        self.mapping_256[1]  = 10 # sky 
        self.mapping_256[10] = 11 # person
        self.mapping_256[17] = 12 # rider
        self.mapping_256[8]  = 13 # car 
        self.mapping_256[19] = 15 # bus 
        self.mapping_256[12] = 17 # motorcycle
        self.mapping_256[11] = 18 # bicycle

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            raise ValueError(f"Cannot read mask image: {mask_path}")

        if mask.ndim == 3:
            mask = mask[:, :, 2]        

        image = Image.open(img_path).convert('RGB')
        mask = Image.fromarray(mask.astype(np.uint8))

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST) 

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_raw_np = np.array(mask, dtype=np.int64)
        
        mask_mapped_np = self.mapping_256[mask_raw_np]
        mask = torch.from_numpy(mask_mapped_np).long()

        return image, mask