import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(
        self, 
        images_dir, 
        labels_dir, 
        image_size = (512, 512),
        num_sample = None  # Thêm tham số num_sample
    ):
        """
            images_dir: path to image.
            labels_dir: path to label.
            num_sample: number of samples to load (loads all if None).
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size

        self.image_paths = []
        self.mask_paths = []

        valid_cities = [c for c in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, c))]
        valid_cities.sort()

        for city in valid_cities:
            city_img_dir = os.path.join(images_dir, city)
            city_lbl_dir = os.path.join(labels_dir, city)
            
            for file_name in sorted(os.listdir(city_img_dir)):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_dir, file_name)
                    
                    mask_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    mask_path = os.path.join(city_lbl_dir, mask_name)
                    
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        self.image_paths = self.image_paths[:num_sample]
        self.mask_paths = self.mask_paths[:num_sample]

        self.mapping_256 = np.zeros(256, dtype=np.int64)
        self.mapping_256[:] = 255 # void/ignore
        
        self.mapping_256[7] = 0    # road
        self.mapping_256[8] = 1    # sidewalk
        self.mapping_256[11] = 2   # building
        self.mapping_256[12] = 3   # wall
        self.mapping_256[13] = 4   # fence
        self.mapping_256[17] = 5   # pole
        self.mapping_256[19] = 6   # traffic light
        self.mapping_256[20] = 7   # traffic sign
        self.mapping_256[21] = 8   # vegetation
        self.mapping_256[22] = 9   # terrain
        self.mapping_256[23] = 10  # sky
        self.mapping_256[24] = 11  # person
        self.mapping_256[25] = 12  # rider
        self.mapping_256[26] = 13  # car
        self.mapping_256[27] = 14  # truck
        self.mapping_256[28] = 15  # bus
        self.mapping_256[31] = 16  # train
        self.mapping_256[32] = 17  # motorcycle
        self.mapping_256[33] = 18  # bicycle

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        mask_raw_np = np.array(mask, dtype=np.int64)
        mask_mapped_np = self.mapping_256[mask_raw_np]
        mask = torch.from_numpy(mask_mapped_np).long()

        return image, mask