import sys
import os
project_root = "/root/KhaiDD/FedCar"
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

from .fedavg_omg_client import FedAvg_OMG_Client
from .segformer_b0_avg_omg import SegFormerB0_Avg_OMG

class FedAvg_OMG_Server:
    def __init__(
        self, 
        num_classes,
        backbone_model, 
        source_domains,
        num_rounds, 
        num_epochs, 
        batch_size,

        init_lr,
        min_lr,
        power,
        weight_decay,
        
        cagrad_c=0.4, 
        global_lr=1.0
    ):
        """
        1. num_classess: number of class to classify.
        1.1 backbone_model: The instance of backbone model's class. In this work, it's fixed as SegmentFormer-B0.
        2. source_domains: A list of source domains used to train (for simplicity, each client will correspond with a domain).
        3. num_rounds: Number of communication rounds.
        4. num_epochs: Number of epoch (used to train in server side).
        5. batch_size: Number of batch size (also used in server side).

        6. init_lr & min_lr & power: used to schedule learning rate.
        7. weight_decay: Used in AdamW optimizer (in this work, by default the optimizer will be AdamW optimizer)
        """
        print("\n" + "="*50)
        print("[Server] Initializing FedAvg + OMG Server...")
        self.num_classes = num_classes
        self.backbone_model = backbone_model
        self.source_domains = source_domains
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay

        self.cagrad_c = cagrad_c
        self.global_lr = global_lr

        # :vv trivial.. but this is for identifing device (in case you don't know)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Server] Global device set to: {self.device}")
        print(f"[Server] Source domains registered: {self.source_domains}")
        print(f"[Server] Communication Rounds: {self.num_rounds} | Local Epochs: {self.num_epochs}")

        # For each domain -> we init a client, and assign a domain to him
        self.clients = []
        print("[Server] Initializing remote clients via Ray...")
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                FedAvg_OMG_Client.remote(
                    data=domain,
                    client_id=i,
                    local_model=SegFormerB0_Avg_OMG(
                        num_classes=self.num_classes
                    ),

                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,

                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay
                )
            )
        print(f"[Server] Successfully initialized {len(self.clients)} remote clients.")
        print("="*50 + "\n")
    
    def set_seed(self, seed): 
        # for python and numpy
        random.seed(seed)
        np.random.seed(seed)

        # pytorch cpu & gpu (this is only for single GPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[Server] Global seed set to {seed}.")
    
    # =========================================================================
    # CORE LOGIC of FED_OMG (CAGRAD)
    # =========================================================================
    def OMG(self, grad_vec, num_tasks, cagrad_c):
        """
        grad_vec shape: [num_tasks, total_flattened_params]
        """
        grads = grad_vec
        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        # Optimize w with SGD
        w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg + 1e-4).sqrt() * cagrad_c
        w_best = None
        obj_best = np.inf
        
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        # aggregate gradient
        g = ((1 / num_tasks + ww * lmbda).view(-1, 1).to(grads.device) * grads).sum(0) / (1 + cagrad_c ** 2)
        return g

    def aggregate_omg(self, local_weights_list, total_samples_list):
        print("[Server] Starting FedAvg + OMG (CAGrad) aggregation...")
        num_tasks = len(local_weights_list)
        total_samples = sum(total_samples_list)
        
        # Data-size weights
        weights = [n_k / total_samples for n_k in total_samples_list]

        global_state_dict = self.backbone_model.state_dict()
        param_names = [name for name, param in self.backbone_model.named_parameters()]
        buffer_names = [name for name, buf in self.backbone_model.named_buffers()]

        all_domain_grads = []
        for i in range(num_tasks):
            local_w = local_weights_list[i]
            domain_grad_diff = []
            for name in param_names:
                # Delta W = (W_local - W_global) * data_weight
                diff = (local_w[name].to(self.device) - global_state_dict[name].to(self.device)) * weights[i]
                domain_grad_diff.append(diff.view(-1))
            
            domain_grad_vector = torch.cat(domain_grad_diff)
            all_domain_grads.append(domain_grad_vector)
            
        all_domain_grads_tensor = torch.stack(all_domain_grads) # Shape: [num_tasks, total_params]

        omg_grads = self.OMG(all_domain_grads_tensor, num_tasks, self.cagrad_c)
        new_global_weights = copy.deepcopy(global_state_dict)
        
        offset = 0
        for name in param_names:
            param_shape = global_state_dict[name].shape
            param_size = global_state_dict[name].numel()
            
            updated_param_grad = omg_grads[offset : offset + param_size].view(param_shape)
            
            # W_new = W_old + global_lr * omg_grad
            new_global_weights[name] = global_state_dict[name].to(self.device) + (updated_param_grad * self.global_lr)
            offset += param_size

        for name in buffer_names:
            avg_buf = torch.zeros_like(global_state_dict[name], dtype=torch.float32).to(self.device)
            for i in range(num_tasks):
                avg_buf += local_weights_list[i][name].to(self.device).to(torch.float32) * weights[i]
            new_global_weights[name] = avg_buf.to(global_state_dict[name].dtype)

        print("[Server] FedAvg + OMG Aggregation complete.")
        return new_global_weights

    def train(self, checkpoint_path):
        print(f"\n[Server] Commencing Federated Learning process for {self.num_rounds} rounds.")
        global_weights = self.backbone_model.state_dict()
        round_pbar = tqdm(range(self.num_rounds), desc="Round", position=0)

        for round_idx in round_pbar:
            print(f"\n--- [Server] Starting Round {round_idx + 1}/{self.num_rounds} ---")
            job_ids = [
                client.train.remote(global_parameters=global_weights) 
                for client in self.clients
            ]

            results = ray.get(job_ids)
            
            local_weights_list = [r[0] for r in results]
            total_samples_list = [r[1] for r in results]

            global_weights = self.aggregate_omg(
                local_weights_list=local_weights_list, 
                total_samples_list=total_samples_list
            )

            self.backbone_model.load_state_dict(global_weights)
            round_pbar.set_description(f"Num Finished Round {round_idx + 1}")

        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        return self.backbone_model

    def evaluate(self, target_domain, checkpoint_path):
        print("\n" + "="*50)
        print(f"[Server] Starting evaluation on Target Domain: {target_domain}")

        self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        print(f"[Server] Loaded checkpoint from {checkpoint_path}")

        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        print(f"[Server] Loading dataset for {target_domain}...")
        dataset=None
        if target_domain == 'cityscape':
            dataset = CityscapesDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/cityscape/leftImg8bit/val",
                labels_dir="/root/KhaiDD/FedCar/dataset/cityscape/gtFine/val"
            )
        elif target_domain == "bdd100":
            dataset = BDD100KDataset(
                images_dir="/root/KhaiDD/FedCar/dataset/bdd100/10k/val",
                labels_dir="/root/KhaiDD/FedCar/dataset/bdd100/labels/val"
            )
        elif target_domain == "gta5":
            dataset = GTA5Dataset(
                list_of_paths=[
                    # "/root/KhaiDD/FedCar/dataset/gta5/gta5_part8",
                    # "/root/KhaiDD/FedCar/dataset/gta5/gta5_part9",
                    "/root/KhaiDD/FedCar/dataset/gta5/gta5_part10"
                ]
            )
        elif target_domain == "mapillary":
            dataset = MapillaryDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/mapillary/validation"
            ) 
        elif target_domain == "synthia":
            dataset = SynthiaDataset(
                root_dir="/root/KhaiDD/FedCar/dataset/synthia/RAND_CITYSCAPES",
                start_index=6580
            )
        
        self.test_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        print(f"[Server] Dataset loaded. Total batches to evaluate: {len(self.test_dataloader)}")
        
        print("[Server] Starting inference loop...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.backbone_model(images) # Output: [B, 19, 512, 512]

                preds = torch.argmax(outputs, dim=1) # Preds: [B, 512, 512]
                
                # Only caculate on pixel that is not 255 
                mask_valid = (masks != 255)
                
                target = masks[mask_valid]
                predict = preds[mask_valid]
                
                indices = self.num_classes * target + predict
                conf_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

        print("[Server] Calculating metrics from confusion matrix...")
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
        for i, iou in enumerate(iou_per_class):
            print(f"Class {i:2d}: {iou.item()*100:.2f}%")
        return miou, pixel_acc, iou_per_class