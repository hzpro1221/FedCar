
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
from diffusers.utils import load_image
import os

def rescale_image(img_tensor, height, width):
    return F.interpolate(
        img_tensor, 
        size=(height, width), 
        mode='bilinear', 
        align_corners=False, 
        antialias=True
    )

def get_pytorch_edges(img_tensor, k=15.0, threshold=0.1):
    gray = img_tensor.mean(dim=1, keepdim=True)
    
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img_tensor.device).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=img_tensor.device).view(1,1,3,3)
    
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    
    mag = torch.sqrt(gx**2 + gy**2 + 1e-4)
    
    mag = mag / (mag.max() + 1e-6)
    
    y_soft = torch.sigmoid(k * (mag - threshold)) 
    
    y_hard = (mag > threshold).float() 
    
    binary_edges = (y_hard - y_soft).detach() + y_soft 
    return binary_edges

def process_images_edges(pairs, h, w):
    results = [] 
    
    for prompt, path in pairs:
        raw_img = Image.open(path)
        img_np = np.array(raw_img)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to("cuda")
        
        img_rescaled = rescale_image(img_tensor, h, w)
        
        max_pool = F.max_pool2d(img_rescaled, kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-img_rescaled, kernel_size=3, stride=1, padding=1)
        
        edge_mask = (max_pool != min_pool).float() 
        
        edge_pil = to_pil_image((edge_mask.squeeze().cpu() * 255).byte().unsqueeze(0).repeat(3, 1, 1))
        
        results.append({
            'prompt': prompt,
            'edge_pil': edge_pil,              
            'target_edge_tensor': edge_mask   
        })
        
    return results