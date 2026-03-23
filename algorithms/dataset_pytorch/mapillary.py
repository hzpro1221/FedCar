import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class MapillaryDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        image_size=(512, 512),
        num_sample=None  
    ):
        """
            root_dir: path to data folder
            num_sample: number of samples to load (loads all if None).
        """
        self.root_dir = root_dir
        self.image_size = image_size
        
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        
        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Error: Cant find images or/ and labels in {root_dir}")

        self.image_paths = []
        self.mask_paths = []

        for file_name in sorted(os.listdir(self.images_dir)):
            if file_name.endswith('.jpg'):
                img_path = os.path.join(self.images_dir, file_name)
                
                mask_name = file_name.replace('.jpg', '.png')
                mask_path = os.path.join(self.labels_dir, mask_name)
                
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        self.image_paths = self.image_paths[:num_sample]
        self.mask_paths = self.mask_paths[:num_sample]

        self.mapping_256 = np.full(256, 255, dtype=np.int64) # By default, all irrelevant data are 255 (void/ignore)
        
        self.mapping_256[[7, 8, 13, 14, 23, 24]] = 0  # road
        self.mapping_256[[11, 15]]               = 1  # sidewalk
        self.mapping_256[17]                     = 2  # building
        self.mapping_256[6]                      = 3  # wall
        self.mapping_256[3]                      = 4  # fence
        self.mapping_256[[44, 45, 47]]           = 5  # pole
        self.mapping_256[48]                     = 6  # traffic light
        self.mapping_256[49]                     = 7  # traffic sign
        self.mapping_256[30]                     = 8  # vegetation
        self.mapping_256[29]                     = 9  # terrain
        self.mapping_256[27]                     = 10 # sky
        self.mapping_256[19]                     = 11 # person
        self.mapping_256[[20, 21, 22]]           = 12 # rider
        self.mapping_256[55]                     = 13 # car
        self.mapping_256[61]                     = 14 # truck
        self.mapping_256[54]                     = 15 # bus
        self.mapping_256[58]                     = 16 # train
        self.mapping_256[57]                     = 17 # motorcycle
        self.mapping_256[52]                     = 18 # bicycle

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST) 

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        mask_raw_np = np.array(mask, dtype=np.int64)
        mask_mapped_np = self.mapping_256[mask_raw_np]
        mask = torch.from_numpy(mask_mapped_np).long()

        return image, mask