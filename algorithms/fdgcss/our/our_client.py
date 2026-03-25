import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ray
from typing import Tuple, Dict

from algorithms.fdgcss.our.utils.augment_dataloader import get_augmented_dataloader
from algorithms.fdgcss.our.segformer_b0_our import SegFormerB0_CovMatch

@ray.remote(num_gpus=0.2)
class FedCovMatch_Client:
    def __init__(
        self,
        data,
        client_id,
        local_model,
        num_rounds,
        num_epoch,
        batch_size,
        num_workers,
        init_lr,
        min_lr,
        power,
        weight_decay,
        lam_cov,
        lam_syn,
        lam_cons,
        proj_dim,
        max_steps_per_epch,
        num_classes = 19,

        full_cov_pixel_theshold = 640,
        entropy_threshold = 1.8,
        min_pix=16,
        kd_temperature=2.0
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        self.local_model = local_model.to(self.device)

        self.num_rounds = num_rounds
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_steps_per_epch = max_steps_per_epch

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay

        self.lam_cov = lam_cov
        self.lam_syn = lam_syn
        self.lam_cons = lam_cons
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.feature_dim = 256

        self.full_cov_pixel_theshold = full_cov_pixel_theshold
        self.entropy_threshold = entropy_threshold
        self.min_pix = min_pix
        self.kd_temperature = kd_temperature

        ds_map = {
            "cityscape": "Cityscapes",
            "bdd100": "BDD100K",
            "gta5": "GTA5",
            "mapillary": "Mapillary",
            "synthia": "Synthia",
        }
        
        self.syn_dataloader = get_augmented_dataloader(
            root_dir="dataset/augment_data",
            dataset_names=[ds_map[self.data]],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.total_samples = len(self.syn_dataloader.dataset)
        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=255)

        self.optimizer = optim.AdamW(
            self.local_model.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay,
        )
        total_steps = self.num_rounds * self.num_epoch * self.max_steps_per_epch
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=total_steps, power=self.power
        )

    def _downsample_labels(self, y):
        return F.interpolate(y.unsqueeze(1).float(), size=(H_f, W_f), mode="nearest").squeeze(1).long()

    def _quality_gate(self, logits_syn, y, threshold):
        with torch.no_grad():
            valid = y != 255
            if not valid.any():
                return False
            probs = F.softmax(logits_syn.detach(), dim=1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)
            return entropy[valid].mean().item() < threshold

    def _project(self, F_map, P):
        B, D, H, W = F_map.shape
        return F_map.permute(0, 2, 3, 1).reshape(-1, D) @ P

    def _per_class_moments(self, F_proj, y_flat) -> Tuple[Dict, Dict, Dict]:
        mu_dict, Sigma_dict, n_dict = {}, {}, {}
        for c in range(self.num_classes):
            mask = y_flat == c
            n_c = mask.sum().item()
            if n_c < self.min_pix:
                continue
            
            Z = F_proj[mask]
            mu_c = Z.mean(dim=0)
            Zc = Z - mu_c
            Sig_c = Zc.T @ Zc / max(n_c - 1, 1)
            Sig_c = Sig_c + 1e-4 * torch.eye(self.proj_dim, device=self.device)
            
            mu_dict[c] = mu_c
            Sigma_dict[c] = Sig_c
            n_dict[c] = n_c
            
        return mu_dict, Sigma_dict, n_dict

    def _cov_loss(self, mu_dict: Dict, Sigma_dict: Dict, n_dict: Dict, global_stats: Dict) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for c, mu_c in mu_dict.items():
            if c not in global_stats:
                continue

            g_mu = global_stats[c]["mu"].to(self.device)
            g_Sigma = global_stats[c]["Sigma"].to(self.device)

            loss_mu = F.mse_loss(mu_c, g_mu, reduction="mean")

            if n_dict[c] < self.full_cov_pixel_theshold:
                loss_sigma = F.mse_loss(Sigma_dict[c].diagonal(), g_Sigma.diagonal(), reduction="mean")
            else:
                loss_sigma = ((Sigma_dict[c] - g_Sigma) ** 2).mean()

            loss += loss_mu + loss_sigma
            count += 1

        return loss / max(count, 1)

    def train(self, global_parameters: dict, global_stats: dict, P_matrix: torch.Tensor) -> Tuple[dict, int, dict]:
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()

        P = P_matrix.to(self.device)
        gs = {
            c: {"mu": s["mu"].to(self.device), "Sigma": s["Sigma"].to(self.device)}
            for c, s in global_stats.items()
        }

        n_kc = torch.zeros(self.num_classes, device=self.device)
        s_kc = torch.zeros(self.num_classes, self.proj_dim, device=self.device)
        Q_kc = torch.zeros(self.num_classes, self.proj_dim, self.proj_dim, device=self.device)

        steps_skipped = 0
        steps_total = 0

        for epoch in range(self.num_epoch):
            for step, (x_real, x_syn, y) in enumerate(self.syn_dataloader):
                if step >= self.max_steps_per_epch:
                    break

                x_real, x_syn, y = x_real.to(self.device), x_syn.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                logits_real, F_real = self.local_model.forward_features(x_real)
                logits_syn, F_syn = self.local_model.forward_features(x_syn)

                L_real = self.criterion_seg(logits_real, y)

                B, D, H_f, W_f = F_real.shape
                y_flat = self._downsample_labels(y, H_f, W_f).reshape(-1)
                F_proj_real = self._project(F_real, P)

                mu_real, Sigma_real, n_real = self._per_class_moments(F_proj_real, y_flat)

                for c in mu_real:
                    Z_c = F_proj_real[y_flat == c].detach()
                    n_kc[c] += n_real[c]
                    s_kc[c] += Z_c.sum(dim=0)
                    Q_kc[c] += Z_c.T @ Z_c

                L_cov = self._cov_loss(mu_real, Sigma_real, n_real, gs)

                use_syn = self._quality_gate(logits_syn, y, self.entropy_threshold)
                steps_total += 1
                steps_skipped += int(not use_syn)

                L_syn = torch.tensor(0.0, device=self.device)
                L_cons = torch.tensor(0.0, device=self.device)

                if use_syn:
                    valid_mask = y != 255
                    if valid_mask.any():
                        L_syn = self.criterion_seg(logits_syn, y)

                        log_r = logits_real.permute(0, 2, 3, 1)[valid_mask]
                        log_s = logits_syn.permute(0, 2, 3, 1)[valid_mask]
                        
                        prob_real = F.softmax(log_r / self.kd_temperature, dim=-1).detach()
                        log_prob_syn = F.log_softmax(log_s / self.kd_temperature, dim=-1)
                        
                        L_cons = F.kl_div(log_prob_syn, prob_real, reduction="batchmean") * (self.kd_temperature ** 2)

                total_loss = L_real + self.lam_cov * L_cov + self.lam_syn * L_syn + self.lam_cons * L_cons

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

        skip_pct = 100.0 * steps_skipped / max(steps_total, 1)
        print(f"[Client {self.client_id}|{self.data}] syn_skip={steps_skipped}/{steps_total} ({skip_pct:.0f}%)")

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        moments = {
            "n_kc": n_kc.cpu(),
            "s_kc": s_kc.cpu(),
            "Q_kc": Q_kc.cpu(),
            "steps_skipped": steps_skipped,
            "steps_total": steps_total,
        }
        num_samples = min(self.max_steps_per_epch * self.batch_size * self.num_epoch, self.total_samples)
        
        return local_weights, num_samples, moments