import sys, os
import gc
import traceback

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

class DatasetAugmenter:
    def __init__(
        self, 
        datasets, 
        dataset_names, 
        output_dir, 
        prompts,
        min_area,
        num_inference_steps,
        controlnet_conditioning_scale,
        max_samples=None,
        base_model="runwayml/stable-diffusion-v1-5", 
        controlnet_model="lllyasviel/sd-controlnet-canny",
    ):
        self.datasets = datasets 
        self.dataset_names = dataset_names 
        self.output_dir = output_dir
        self.prompts = prompts     
        self.min_area = min_area
        self.num_inference_steps = num_inference_steps
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.max_samples = max_samples

        print("Creating folder to save data...")        
        for ds_name in self.dataset_names:
            for key in self.prompts.keys():
                os.makedirs(os.path.join(self.output_dir, ds_name, key, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, ds_name, key, "labels"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, ds_name, key, "originals"), exist_ok=True)

        self.class_names = {
            0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
            5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
            9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
            14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    def _get_canny_edges(self, image_np, low_threshold=100, high_threshold=200):
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(image_cv, low_threshold, high_threshold)
        edges = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges)

    def _generator(self, prompts, control_images):
        return self.pipe(
            prompt=prompts,
            image=control_images,        
            num_inference_steps=self.num_inference_steps,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale
        ).images

    def _process_single_image(self, original_image_pil, mapped_mask_np, base_prompt, canny_condition):
        output_image_np = np.array(original_image_pil).astype(np.float32)
        unique_classes = np.unique(mapped_mask_np)
        
        batch_prompts = []
        batch_masks_3d = []
        
        for class_id in unique_classes:
            if class_id == 255 or class_id not in self.class_names:
                continue
            
            mask_2d = (mapped_mask_np == class_id).astype(np.float32)
            
            if np.sum(mask_2d) < self.min_area:
                continue
                
            class_name = self.class_names[class_id]
            segment_prompt = f"{class_name} in {base_prompt}"
            mask_3d = np.expand_dims(mask_2d, axis=-1)
            
            batch_prompts.append(segment_prompt)
            batch_masks_3d.append(mask_3d)

        if not batch_prompts:
            return original_image_pil

        batch_size = len(batch_prompts)
        batch_control_images = [canny_condition] * batch_size

        generated_images_pil = self._generator(
            prompts=batch_prompts, 
            control_images=batch_control_images
        )

        for mask_3d, generated_pil in zip(batch_masks_3d, generated_images_pil):
            generated_full_np = np.array(generated_pil).astype(np.float32)
            output_image_np = (output_image_np * (1.0 - mask_3d)) + (generated_full_np * mask_3d)
            
        output_image_np = np.clip(output_image_np, 0, 255).astype(np.uint8)
        return Image.fromarray(output_image_np)

    def run_augmentation(self):
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_name = self.dataset_names[dataset_idx] 
            print(f"\nProcessing Dataset {dataset_name} ({dataset_idx + 1}/{len(self.datasets)})...")
            
            total_samples = len(dataset) if self.max_samples is None else min(self.max_samples, len(dataset))
            
            for idx in tqdm(range(total_samples), desc=f"Augmenting {dataset_name}"):
                image_tensor, mask_tensor = dataset[idx]
                
                img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                original_pil = Image.fromarray(img_np)
                mask_np = mask_tensor.numpy()
                
                mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                canny_condition = self._get_canny_edges(img_np)
                
                for prompt_name, prompt_text in self.prompts.items():
                    augmented_pil = self._process_single_image(
                        original_image_pil=original_pil,
                        mapped_mask_np=mask_np,
                        base_prompt=prompt_text,
                        canny_condition=canny_condition
                    )
                    
                    file_prefix = f"sample_{idx:05d}"
                    
                    img_save_path = os.path.join(self.output_dir, dataset_name, prompt_name, "images", f"{file_prefix}_img.png")
                    lbl_save_path = os.path.join(self.output_dir, dataset_name, prompt_name, "labels", f"{file_prefix}_label.png")
                    orig_save_path = os.path.join(self.output_dir, dataset_name, prompt_name, "originals", f"{file_prefix}_orig.png")
                    
                    augmented_pil.save(img_save_path)
                    mask_pil.save(lbl_save_path)
                    original_pil.save(orig_save_path)

if __name__ == "__main__":
    OUTPUT_DIR="dataset/augment_data"

    IMAGE_SIZE = (512, 512)
    PROMPTS ={
    "spring": "springtime",
    "summer": "sunny summer day",
    "autumn": "autumn",
    "winter": "winter"
    }

    MIN_AREA=0
    NUM_INFERENCE_STEPS = 40
    CONTROLNET_CONDITIONING_SCALE = 1.2
    MAX_SAMPLES = 1

    try:
        from algorithms.dataset_pytorch import (
            BDD100KDataset, 
            CityscapesDataset, 
            GTA5Dataset, 
            MapillaryDataset, 
            SynthiaDataset
        )
    except ImportError as e:
        print(f"Error importing datasets: {e}")
        sys.exit(1)

    datasets_config = [
        {
            "name": "Cityscapes",
            "class_ref": CityscapesDataset,
            "kwargs": {
                "images_dir": "dataset/cityscape/leftImg8bit/train",
                "labels_dir": "dataset/cityscape/gtFine/train",
                "image_size": IMAGE_SIZE
            }
        },
        {
            "name": "BDD100K",
            "class_ref": BDD100KDataset,
            "kwargs": {
                "images_dir": "dataset/bdd100/10k/train",
                "labels_dir": "dataset/bdd100/labels/train",
                "image_size": IMAGE_SIZE
            }
        },
        {
            "name": "GTA5",
            "class_ref": GTA5Dataset,
            "kwargs": {
                "list_of_paths": [
                    "dataset/gta5/gta5_part1",
                    "dataset/gta5/gta5_part2",
                    "dataset/gta5/gta5_part3",
                    "dataset/gta5/gta5_part4",
                    "dataset/gta5/gta5_part5",
                    "dataset/gta5/gta5_part6",
                    "dataset/gta5/gta5_part7",
                ],
                "image_size": IMAGE_SIZE
            }
        },
        {
            "name": "Mapillary",
            "class_ref": MapillaryDataset,
            "kwargs": {
                "root_dir": "dataset/mapillary/training",
                "image_size": IMAGE_SIZE
            }
        },
        {
            "name": "Synthia",
            "class_ref": SynthiaDataset,
            "kwargs": {
                "root_dir": "dataset/synthia/RAND_CITYSCAPES",
                "image_size": IMAGE_SIZE,
                "start_index": 0,
                "end_index": 6580 
            }
        }
    ]

    loaded_datasets = []
    dataset_names = [] 
    
    for config in datasets_config:
        print(f"Loading {config['name']} dataset...")
        try:
            dataset_instance = config["class_ref"](**config["kwargs"])
            loaded_datasets.append(dataset_instance)
            dataset_names.append(config["name"]) 
        except Exception as e:
            print(f"Failed to load {config['name']}: {e}")

    if not loaded_datasets:
        print("No datasets were successfully loaded. Exiting.")
        sys.exit(1)

    print("\nInitializing DatasetAugmenter...")
    augmenter = DatasetAugmenter(
        datasets=loaded_datasets,
        dataset_names=dataset_names, 
        output_dir=OUTPUT_DIR,
        prompts=PROMPTS,
        min_area=MIN_AREA,                    
        num_inference_steps=NUM_INFERENCE_STEPS,            
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,  
        max_samples=MAX_SAMPLES
    )

    print("\nStarting augmentation pipeline...")
    augmenter.run_augmentation()
    print("\nAugmentation complete!")