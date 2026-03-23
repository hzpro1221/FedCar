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

from algorithms.dataset_pytorch import BDD100KDataset, CityscapesDataset, GTA5Dataset, MapillaryDataset, SynthiaDataset
from .gperxan_client import gPerXAN_Client
from .segformer_b0_gperxan import SegFormerB0_gPerXAN

class gPerXAN_Server:
    def __init__(
        self, 
        num_classes,
        backbone_model, 
        source_domains,
        num_rounds, 
        num_epochs, 
        batch_size,

        num_workers,         
        num_sample,         
        max_steps_per_epch,  

        init_lr,
        min_lr,
        power,
        weight_decay,
        reg_weight
    ):
        """
        Initializes the gPerXAN Server for Personalized Federated Learning.
        This framework manages remote clients that utilize Cross-domain Adaptive Normalization (XON)
        and local-global consistency regularization.

        Args:
            num_classes (int): Number of semantic categories for classification.
            backbone_model (nn.Module): The global backbone model instance.
            source_domains (list): List of names for the source domains.
            num_rounds (int): Total number of communication rounds.
            num_epochs (int): Number of local training epochs per client.
            batch_size (int): Batch size for local training and evaluation.
            num_workers (int): Number of data loading worker processes.
            num_sample (int): Total number of training samples to load per dataset.
            max_steps_per_epch (int): Maximum steps per local epoch to control training duration.
            init_lr (float): Initial learning rate for the optimizer.
            min_lr (float): Minimum learning rate for polynomial decay.
            power (float): Power factor for the learning rate scheduler.
            weight_decay (float): Weight decay coefficient (AdamW).
            reg_weight (float): Scalar weight for the Server-Head regularization loss.
        """
        print("\n" + "="*50)
        print("[Server] Initializing gPerXAN (Personalized FL) Server...")
        self.num_classes = num_classes
        self.backbone_model = backbone_model
        self.source_domains = source_domains
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.num_workers = num_workers
        self.num_sample = num_sample
        self.max_steps_per_epch = max_steps_per_epch

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.reg_weight = reg_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device set to: {self.device}")
        print(f"[Server] Source domains: {self.source_domains}")
        print(f"[Server] Hyperparams: reg_weight={reg_weight} | max_steps={max_steps_per_epch}")

        # Initialize remote clients via Ray
        self.clients = []
        print("[Server] Initializing remote Personalized clients via Ray...")
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                gPerXAN_Client.remote(
                    data=domain,
                    client_id=i,
                    local_model=SegFormerB0_gPerXAN(
                        num_classes=self.num_classes
                    ),
                    num_sample=self.num_sample,
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    max_steps_per_epch=self.max_steps_per_epch,
                    reg_weight=self.reg_weight
                )
            )
        print(f"[Server] Successfully initialized {len(self.clients)} remote clients.")
        print("="*50 + "\n")
    
    def set_seed(self, seed): 
        """Sets the random seed for reproducibility across all libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[Server] Global seed set to {seed}.")
    
    def aggregate(self, local_weights_list, total_samples_list):
        """
        Performs weighted aggregation (FedAvg) on the collected local model weights.
        The formula used is: $$W_{global} = \sum_{k=1}^{K} \frac{n_k}{N} W_k$$
        """
        print("[Server] Starting weighted parameter aggregation...")
        total_samples = sum(total_samples_list)
        
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])

        for i in range(len(local_weights_list)):
            local_w = local_weights_list[i]
            weight_factor = total_samples_list[i] / total_samples

            for key in avg_weights.keys():
                avg_weights[key] += (local_w[key] * weight_factor).to(avg_weights[key].dtype)
        
        print("[Server] Aggregation complete.")
        return avg_weights

    def train(self, checkpoint_path):
        """Main training loop for Federated Learning rounds."""
        print(f"\n[Server] Commencing gPerXAN training for {self.num_rounds} communication rounds.")
        global_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="Round", position=0)

        for round_idx in round_pbar:
            # Broadcast global weights to all available clients
            job_ids = [
                client.train.remote(global_parameters=global_weights) 
                for client in self.clients
            ]

            results = ray.get(job_ids)
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]

            # Aggregate collected weights (Client filters out local Norm parameters automatically)
            global_weights = self.aggregate(
                local_weights_list=local_weights_list, 
                total_samples_list=total_samples_list
            )

            self.backbone_model.load_state_dict(global_weights)
            round_pbar.set_description(f"Finished Round {round_idx + 1}")

        print(f"\n[Server] Training complete. Saving global model to {checkpoint_path}")
        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model
    
    def evaluate(self, target_domain, checkpoint_path):
        """
        Evaluates the current global model on a held-out target domain.
        Synchronizes the sample count to 10% of the training sample size.
        """
        print("\n" + "="*50)
        print(f"[Server] Evaluation started for Target Domain: {target_domain}")

        self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        
        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        print(f"[Server] Loading evaluation dataset with {int(self.num_sample / 10)} samples...")
        dataset = None
        if target_domain == 'cityscape':
            dataset = CityscapesDataset(
                images_dir="dataset/cityscape/leftImg8bit/val",
                labels_dir="dataset/cityscape/gtFine/val",
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "bdd100":
            dataset = BDD100KDataset(
                images_dir="dataset/bdd100/10k/val",
                labels_dir="dataset/bdd100/labels/val",
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "gta5":
            dataset = GTA5Dataset(
                list_of_paths=["dataset/gta5/gta5_part10"],
                num_sample=int(self.num_sample / 10)
            )
        elif target_domain == "mapillary":
            dataset = MapillaryDataset(
                root_dir="dataset/mapillary/validation",
                num_sample=int(self.num_sample / 10)
            ) 
        elif target_domain == "synthia":
            dataset = SynthiaDataset(
                root_dir="dataset/synthia/RAND_CITYSCAPES",
                start_index=6580,
                num_sample=int(self.num_sample / 10)
            )
        
        test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"[Server] Inference loop started on {len(test_dataloader)} batches...")
        with torch.no_grad():
            for images, masks in tqdm(test_dataloader, desc="Inference"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.backbone_model(images) 
                preds = torch.argmax(outputs, dim=1)
                
                # Filter ignore_index (255) pixels for metric calculation
                mask_valid = (masks != 255)
                target = masks[mask_valid]
                predict = preds[mask_valid]
                
                indices = self.num_classes * target + predict
                conf_matrix += torch.bincount(
                    indices, minlength=self.num_classes**2
                ).reshape(self.num_classes, self.num_classes)

        # Metric extraction from confusion matrix
        tp = torch.diag(conf_matrix)
        fp = torch.sum(conf_matrix, dim=0) - tp
        fn = torch.sum(conf_matrix, dim=1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou = torch.mean(iou_per_class).item()
        pixel_acc = (torch.sum(tp) / (torch.sum(conf_matrix) + 1e-10)).item()

        print("\n" + "="*40)
        print(f"Metrics for {target_domain}:")
        print(f"- mIoU: {miou*100:.2f}%")
        print(f"- Pixel Accuracy: {pixel_acc*100:.2f}%")
        print("="*40)
        
        return miou, pixel_acc, iou_per_class