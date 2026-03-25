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
import wandb 

from algorithms.dataset_pytorch import (
    BDD100KDataset, CityscapesDataset, GTA5Dataset,
    MapillaryDataset, SynthiaDataset
)
from .our_client import FedCovMatch_Client


class FedCovMatch_Server:
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
        lam_cov=1.0, 
        lam_syn=0.5, 
        lam_cons=0.3,
        ema_beta = 0.6,
        ema_beta_per_round = 0.06
    ):
        print("\n" + "=" * 50)
        print("[Server] Initializing FedCovMatch v2...")

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
        self.lam_cov = lam_cov
        self.lam_syn = lam_syn
        self.lam_cons = lam_cons
        self.ema_beta = ema_beta
        self.ema_beta_per_round = ema_beta_per_round

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = 256
        self.proj_dim = 64

        self.global_stats = {}

        P_init = torch.randn(self.feature_dim, self.proj_dim)
        Q, _   = torch.linalg.qr(P_init)
        self.P_matrix = Q   

        print(f"[Server] Projection matrix: {self.feature_dim} → {self.proj_dim} (orthonormal QR)")
        print(f"[Server] Lambdas: cov={lam_cov}, syn={lam_syn}, cons={lam_cons}")

        self.clients = []
        for i, domain in enumerate(self.source_domains):
            self.clients.append(
                FedCovMatch_Client.remote(
                    data=domain,
                    client_id=i,
                    local_model=copy.deepcopy(self.backbone_model),
                    num_rounds=self.num_rounds,
                    num_epoch=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    max_steps_per_epch=self.max_steps_per_epch,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    power=self.power,
                    weight_decay=self.weight_decay,
                    lam_cov=self.lam_cov,
                    lam_syn=self.lam_syn,
                    lam_cons=self.lam_cons,
                    proj_dim=self.proj_dim,
                    num_classes=self.num_classes
                )
            )
        print(f"[Server] {len(self.clients)} clients initialized.\n" + "=" * 50)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def aggregate(self, local_weights_list, total_samples_list):
        total_samples = sum(total_samples_list)
        avg_weights   = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
        for i, w in enumerate(local_weights_list):
            factor = total_samples_list[i] / total_samples
            for key in avg_weights.keys():
                avg_weights[key] += (w[key] * factor).to(avg_weights[key].dtype)
        return avg_weights

    def update_global_stats(self, local_moments_list, round_idx):
        N = torch.zeros(self.num_classes, device=self.device)
        S = torch.zeros(self.num_classes, self.proj_dim, device=self.device)
        Q = torch.zeros(self.num_classes, self.proj_dim, self.proj_dim, device=self.device)

        for moments in local_moments_list:
            N += moments['n_kc'].to(self.device)
            S += moments['s_kc'].to(self.device)
            Q += moments['Q_kc'].to(self.device)

        momentum = min(self.ema_beta, round_idx * self.ema_beta_per_round)
        n_updated = 0
        
        for c in range(self.num_classes):
            n_c = N[c].item()
            mu_new = S[c] / n_c
            mu_outer = torch.outer(mu_new, mu_new)
            Sig_new = (Q[c] - n_c * mu_outer) / (n_c - 1)
            Sig_new = Sig_new + 1e-3 * torch.eye(self.proj_dim, device=self.device)

            if c not in self.global_stats or momentum == 0.0:
                self.global_stats[c] = {
                    'mu': mu_new.cpu(),
                    'Sigma': Sig_new.cpu()
                }
            else:
                prev_mu = self.global_stats[c]['mu'].to(self.device)
                prev_Sig = self.global_stats[c]['Sigma'].to(self.device)
                self.global_stats[c]['mu'] = (momentum * prev_mu  + (1 - momentum) * mu_new).cpu()
                self.global_stats[c]['Sigma'] = (momentum * prev_Sig + (1 - momentum) * Sig_new).cpu()
            n_updated += 1

        return n_updated

    def train(self, target_domain, checkpoint_path):
        print(f"\n[Server] Starting {self.num_rounds} rounds of federated training.")
        global_weights = self.backbone_model.state_dict()

        pbar = tqdm(range(self.num_rounds), desc="Round")
        for round_idx in pbar:
            job_ids = [
                client.train.remote(
                    global_parameters=global_weights,
                    global_stats=self.global_stats,
                    P_matrix=self.P_matrix
                )
                for client in self.clients
            ]
            results = ray.get(job_ids)

            local_weights_list  = [r[0] for r in results]
            total_samples_list  = [r[1] for r in results]
            local_moments_list  = [r[2] for r in results]

            global_weights = self.aggregate(local_weights_list, total_samples_list)
            self.backbone_model.load_state_dict(global_weights)

            n_classes_updated = self.update_global_stats(local_moments_list, round_idx)

            total_skipped = sum(m.get('steps_skipped', 0) for m in local_moments_list)
            total_steps = sum(m.get('steps_total',   0) for m in local_moments_list)
            skip_pct = 100.0 * total_skipped / max(total_steps, 1)
            current_momentum = min(0.6, round_idx * 0.06)

            pbar.set_description(
                f"Round {round_idx + 1}/{self.num_rounds} | "
                f"stats: {n_classes_updated}/{self.num_classes} classes | "
                f"syn_skip: {total_skipped}/{total_steps} ({skip_pct:.0f}%) | "
                f"momentum: {current_momentum:.2f}"
            )
            
            print(f"\n[Server] Evaluating Round {round_idx + 1}...")
            miou, pixel_acc, _ = self.evaluate(target_domain=target_domain, checkpoint_path=None)
            
            wandb.log({
                "Round": round_idx + 1,
                "Round_Test_mIoU": miou * 100,
                "Round_Test_Pixel_Accuracy": pixel_acc * 100,
                "Algorithm_Stats/Classes_Updated": n_classes_updated,
                "Algorithm_Stats/Syn_Skip_Percentage": skip_pct,
                "Algorithm_Stats/EMA_Momentum": current_momentum
            })

        torch.save(self.backbone_model.state_dict(), checkpoint_path)
        print(f"[Server] Checkpoint saved → {checkpoint_path}")
        return self.backbone_model

    def evaluate(self, target_domain, checkpoint_path=None):
        if checkpoint_path is not None:
            self.backbone_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[Server] Loaded checkpoint from {checkpoint_path}")
            
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

        conf_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

        if not hasattr(self, 'test_dataloader'):
            print(f"[Server] Loading dataset for {target_domain} (First time only)...")
            eval_n = int(self.num_sample / 10) if self.num_sample is not None else None
            dataset = None

            if target_domain == 'cityscape':
                dataset = CityscapesDataset(
                    "dataset/cityscape/leftImg8bit/val",
                    "dataset/cityscape/gtFine/val", num_sample=eval_n
                )
            elif target_domain == "bdd100":
                dataset = BDD100KDataset(
                    "dataset/bdd100/10k/val",
                    "dataset/bdd100/labels/val", num_sample=eval_n
                )
            elif target_domain == "gta5":
                dataset = GTA5Dataset(
                    list_of_paths=["dataset/gta5/gta5_part8",
                                   "dataset/gta5/gta5_part9",
                                   "dataset/gta5/gta5_part10"],
                    num_sample=eval_n
                )
            elif target_domain == "mapillary":
                dataset = MapillaryDataset("dataset/mapillary/validation", num_sample=eval_n)
            elif target_domain == "synthia":
                dataset = SynthiaDataset(
                    "dataset/synthia/RAND_CITYSCAPES",
                    start_index=6580, end_index=None, num_sample=eval_n
                )

            self.test_dataloader = DataLoader(
                dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            
        print(f"[Server] Total batches to evaluate: {len(self.test_dataloader)}")

        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader, desc="Evaluating"):
                images, masks = images.to(self.device), masks.to(self.device)
                preds = torch.argmax(self.backbone_model(images), dim=1)
                valid = (masks != 255)
                idx = self.num_classes * masks[valid] + preds[valid]
                conf_matrix += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
                    self.num_classes, self.num_classes
                )

        tp = torch.diag(conf_matrix)
        fp = conf_matrix.sum(0) - tp
        fn = conf_matrix.sum(1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou = iou_per_class.mean().item()
        pixel_acc = (tp.sum() / (conf_matrix.sum() + 1e-10)).item()
        
        print("\n" + "="*40)
        print(f"Evaluate result: \n- mIoU: {miou*100:.2f}%\n- Pixel Accuracy: {pixel_acc*100:.2f}%")
        print("="*40)
        
        return miou, pixel_acc, iou_per_class