import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class BDD100KDataset(Dataset):
    def __init__(
        self, 
        images_dir, 
        labels_dir, 
        image_size = (512, 512),
        num_sample = None  
    ):
        """
            images_dir: path to image.
            labels_dir: path to label.
            num_sample: number of samples to load (loads all if None).
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.image_filenames.sort() 
        
        self.image_filenames = self.image_filenames[:num_sample]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = f"{img_name.replace('.jpg', '')}_train_id.png"
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.labels_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).long()

        return image, mask