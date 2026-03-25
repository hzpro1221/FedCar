import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class GTA5Dataset(Dataset):
    def __init__(
        self, 
        list_of_paths, 
        image_size = (512, 512),
        num_sample = None  
    ):
        self.list_of_paths = list_of_paths
        
        self.image_paths = []
        self.mask_paths = []
        self.image_size = image_size

        if isinstance(self.list_of_paths, list):
            self.list_of_paths.sort()

        for part_path in self.list_of_paths:
            img_dir = os.path.join(part_path, 'images')
            lbl_dir = os.path.join(part_path, 'labels')
            
            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                raise FileNotFoundError(f"Not found images or labels folder in: {part_path}")

            for file_name in sorted(os.listdir(img_dir)):
                if file_name.endswith('.png') or file_name.endswith('.jpg'):
                    img_path = os.path.join(img_dir, file_name)
                    
                    mask_path = os.path.join(lbl_dir, file_name)
                    
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        self.image_paths = self.image_paths[:num_sample]
        self.mask_paths = self.mask_paths[:num_sample]

        self.color2id = {
            (128, 64, 128): 0,   # road
            (244, 35, 232): 1,   # sidewalk
            (70, 70, 70): 2,     # building
            (102, 102, 156): 3,  # wall
            (190, 153, 153): 4,  # fence
            (153, 153, 153): 5,  # pole
            (250, 170, 30): 6,   # traffic light
            (220, 220, 0): 7,    # traffic sign
            (107, 142, 35): 8,   # vegetation
            (152, 251, 152): 9,  # terrain
            (70, 130, 180): 10,  # sky
            (220, 20, 60): 11,   # person
            (255, 0, 0): 12,     # rider
            (0, 0, 142): 13,     # car
            (0, 0, 70): 14,      # truck
            (0, 60, 100): 15,    # bus
            (0, 80, 100): 16,    # train
            (0, 0, 230): 17,     # motorcycle
            (119, 11, 32): 18    # bicycle
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST) 

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        mask_np = np.array(mask)
        id_mask = np.full((mask_np.shape[0], mask_np.shape[1]), 255, dtype=np.int64)
        
        for color, class_id in self.color2id.items():
            matches = (mask_np == color).all(axis=-1)
            id_mask[matches] = class_id
            
        mask = torch.from_numpy(id_mask).long()
        return image, mask