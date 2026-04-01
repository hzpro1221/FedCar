import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import random
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
import ray
from ray.util.actor_pool import ActorPool
import wandb 

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset

from .fedavg_client import FedAvg_Client

from models.bisenet_v2 import BiSeNetV2
from models.topformer import TopformerSeg

class FedAvg_Server:
    def __init__(
        self, 
        num_classes,
        model_name,           
        source_domains,
        num_rounds, 
        num_epochs, 
        batch_size,
        num_workers,
        max_concurrent_clients, 
        num_sample,
        max_steps_per_epch,
        init_lr,
        min_lr,
        power,
        weight_decay,
        **kwargs              
    ):
        print("\n" + "="*50)
        print(f"[Server] Initializing {self.__class__.__name__}...")
        
        self.num_classes = num_classes
        self.model_name = model_name.lower()
        self.source_domains = source_domains
        
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_concurrent_clients = max_concurrent_clients
        self.num_sample = num_sample
        self.max_steps_per_epch = max_steps_per_epch

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device set to: {self.device}")
        
        self.backbone_model = self._build_model()
        
        self.workers = self._init_worker_pool(**kwargs)
        self.actor_pool = ActorPool(self.workers)
        
        print(f"[Server] Successfully initialized pool with {len(self.workers)} reusable remote workers.")
        print("="*50 + "\n")
    
    def _build_model(self):
        if self.model_name == 'bisenetv2':
            return BiSeNetV2(n_classes=self.num_classes)
        elif self.model_name == 'topformer':
            return TopformerSeg(num_classes=self.num_classes) 
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} base workers via Ray...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedAvg_Client.remote(
                    local_model=copy.deepcopy(self.backbone_model),
                    num_sample=self.num_sample,
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    max_steps_per_epch=self.max_steps_per_epch,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    **kwargs
                )
            )
        return workers

    def set_seed(self, seed): 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print(f"[Server] Global seed set to {seed}.")
    
    def aggregate(self, local_weights_list, total_samples_list):
        total_samples = sum(total_samples_list)
        avg_weights = {k: torch.zeros_like(v) for k, v in local_weights_list[0].items()}

        for local_w, n_k in zip(local_weights_list, total_samples_list):
            weight_factor = n_k / total_samples
            for key in avg_weights.keys():
                avg_weights[key] += (local_w[key].float() * weight_factor).to(avg_weights[key].dtype)
        
        return avg_weights

    def update_global_model(self, aggregated_weights):
        self.backbone_model.load_state_dict(aggregated_weights)

    def train(self, target_domain, checkpoint_path):
        print(f"\n[Server] Commencing Federated Learning process for {self.num_rounds} rounds.")
        global_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="Round", position=0)

        for round_idx in round_pbar:
            print(f"\n--- [Server] Starting Round {round_idx + 1}/{self.num_rounds} ---")
            
            tasks = []
            for i, domain in enumerate(self.source_domains):
                tasks.append({
                    "global_weights": global_weights,
                    "data_domain": domain,
                    "client_id": i
                })
            
            print(f"[Server] Pushing {len(tasks)} tasks to ActorPool ({self.max_concurrent_clients} workers)...")
            
            results = list(self.actor_pool.map(
                lambda actor, task: actor.train.remote(
                    global_parameters=task["global_weights"],
                    data_domain=task["data_domain"],
                    client_id=task["client_id"]
                ),
                tasks
            ))

            print(f"[Server] Received local weights from all domains.")
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]

            aggregated_weights = self.aggregate(local_weights_list, total_samples_list)
            self.update_global_model(aggregated_weights)
            global_weights = self.backbone_model.state_dict()
            
            print(f"[Server] Evaluating Round {round_idx + 1}...")
            miou, pixel_acc, _ = self.evaluate(target_domain=target_domain)
            
            wandb.log({
                "Round": round_idx + 1,
                "Round_Test_mIoU": miou * 100,
                "Round_Test_Pixel_Accuracy": pixel_acc * 100
            })

            round_pbar.set_postfix(mIoU=f"{miou * 100:.2f}%")

        print(f"\n[Server] Training complete. Saving global model to {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model

    def evaluate(self, target_domain, checkpoint_path=None):
        print("\n" + "="*50)
        print(f"[Server] Starting evaluation on Target Domain: {target_domain}")

        if checkpoint_path is not None:
            self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[Server] Loaded checkpoint from {checkpoint_path}")
            
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        if not hasattr(self, 'test_dataloader'):
            print(f"[Server] Loading dataset for {target_domain} (First time only)...")
            eval_num_sample = int(self.num_sample / 10) if self.num_sample is not None else None
            dataset = None
            
            if target_domain == 'cityscape':
                dataset = CityscapesDataset("dataset/cityscape/leftImg8bit/val", "dataset/cityscape/gtFine/val", num_sample=eval_num_sample)
            elif target_domain == "bdd100":
                dataset = BDD100KDataset("dataset/bdd100/10k/val", "dataset/bdd100/labels/val", num_sample=eval_num_sample)
            elif target_domain == "gta5":
                dataset = GTA5Dataset(list_of_paths=["dataset/gta5/gta5_part8", "dataset/gta5/gta5_part9", "dataset/gta5/gta5_part10"], num_sample=eval_num_sample)
            elif target_domain == "mapillary":
                dataset = MapillaryDataset("dataset/mapillary/validation", num_sample=eval_num_sample) 
            elif target_domain == "synthia":
                dataset = SynthiaDataset("dataset/synthia/RAND_CITYSCAPES", start_index=6580, end_index=None, num_sample=eval_num_sample)
            else:
                raise ValueError(f"Unknown target domain: {target_domain}")
            
            self.test_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,       
                num_workers=self.num_workers,
                pin_memory=True
            )
            
        print(f"[Server] Total batches to evaluate: {len(self.test_dataloader)}")
        
        print("[Server] Starting inference loop...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.backbone_model(images) 

                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                preds = torch.argmax(logits, dim=1)

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

        total_correct_pixels = torch.sum(tp)
        total_valid_pixels = torch.sum(conf_matrix)
        pixel_acc = (total_correct_pixels / (total_valid_pixels + 1e-10)).item()

        print("\n" + "="*40)
        print(f"Evaluate result: \n- mIoU: {miou*100:.2f}%\n- Pixel Accuracy: {pixel_acc*100:.2f}%")
        print("="*40)
            
        return miou, pixel_acc, iou_per_class