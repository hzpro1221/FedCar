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
from algorithms.fdg_css.fedavg.fedavg_client import Base_FedAvg_Client

@ray.remote(num_gpus=0.2)
class FedCovMatch_Client(Base_FedAvg_Client):
    def __init__(
        self,
        lam_cov=1.0,
        lam_syn=0.5,
        lam_cons=0.3,
        proj_dim=64,
        cov_alignment_mode="hybrid",
        entropy_threshold=1.8,
        kd_temperature=2.0,
        **kwargs
    ):
        model_name = kwargs.pop('model_name', '').lower()
        super().__init__(**kwargs)

        self.lam_cov = lam_cov
        self.lam_syn = lam_syn
        self.lam_cons = lam_cons
        self.proj_dim = proj_dim
        
        if "bisenetv2" in model_name:
            self.feature_dim = 128
        elif "topformer" in model_name:
            self.feature_dim = 256 
        else:
            self.feature_dim = 256

        self.entropy_threshold = entropy_threshold
        self.kd_temperature = kd_temperature

        self.cov_alignment_mode = cov_alignment_mode
        assert self.cov_alignment_mode in ["none", "real", "hybrid", "syn"], "Invalid alignment mode."

        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=255)

    def load_augmented_dataset(self, data_domain):
        if self.data_domain != data_domain:
            self.data_domain = data_domain
            
            import os
            print(f"[Worker PID: {os.getpid()}] Loading Augmented Dataset for domain: {data_domain}...")
            
            ds_map = {
                "cityscape": "Cityscapes", "bdd100": "BDD100K",
                "gta5": "GTA5", "mapillary": "Mapillary", "synthia": "Synthia"
            }
            
            self.syn_dataloader = get_augmented_dataloader(
                root_dir="dataset/augment_data",
                dataset_names=[ds_map[data_domain]],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self.total_samples = len(self.syn_dataloader.dataset)

    def _downsample_labels(self, y, H_f, W_f):
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
        num_classes = self.local_model.n_classes if hasattr(self.local_model, 'n_classes') else 19
        
        for c in range(num_classes):
            mask = y_flat == c
            n_c = mask.sum().item()

            if n_c < 2: 
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
        loss = torch.tensor(0.0, device=self.device, requires_grad=True) 
        count = 0

        for c, mu_c in mu_dict.items():
            if c not in global_stats:
                continue

            g_mu = global_stats[c]["mu"].to(self.device)
            g_Sigma = global_stats[c]["Sigma"].to(self.device)

            loss_mu = F.mse_loss(mu_c, g_mu, reduction="mean")
            loss_sigma = F.mse_loss(Sigma_dict[c], g_Sigma, reduction="mean")

            loss = loss + loss_mu + loss_sigma
            count += 1

        if count > 0:
            return loss / count
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train(self, global_parameters: dict, global_stats: dict, P_matrix: torch.Tensor, data_domain: str, client_id: int) -> Tuple[dict, int, dict]:
        self.load_augmented_dataset(data_domain)
        
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()

        self.optimizer = optim.AdamW(
            self.local_model.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay,
        )
        total_steps = self.num_epoch * self.max_steps_per_epch
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=total_steps, power=self.power
        )

        P = P_matrix.to(self.device)
        gs = {
            c: {"mu": s["mu"].to(self.device), "Sigma": s["Sigma"].to(self.device)}
            for c, s in global_stats.items()
        }

        num_classes = self.local_model.n_classes if hasattr(self.local_model, 'n_classes') else 19

        n_kc = torch.zeros(num_classes, device=self.device)
        s_kc = torch.zeros(num_classes, self.proj_dim, device=self.device)
        Q_kc = torch.zeros(num_classes, self.proj_dim, self.proj_dim, device=self.device)

        steps_skipped = 0
        steps_total = 0

        for epoch in range(self.num_epoch):
            for step, (x_real, x_syn, y) in enumerate(self.syn_dataloader):
                if step >= self.max_steps_per_epch:
                    break

                x_real, x_syn, y = x_real.to(self.device), x_syn.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                logits_real, F_real_raw = self.local_model(x_real, return_features=True)
                logits_syn, F_syn_raw = self.local_model(x_syn, return_features=True)

                F_real = F_real_raw[-1] if isinstance(F_real_raw, (list, tuple)) else F_real_raw
                F_syn = F_syn_raw[-1] if isinstance(F_syn_raw, (list, tuple)) else F_syn_raw

                L_real = self.criterion_seg(logits_real, y)

                B, D, H_f, W_f = F_real.shape
                y_flat = self._downsample_labels(y, H_f, W_f).reshape(-1)

                if self.cov_alignment_mode == "none":
                    L_cov = torch.tensor(0.0, device=self.device)
                else:
                    if self.cov_alignment_mode == "real":
                        F_proj = self._project(F_real, P)
                        y_proj = y_flat
                    elif self.cov_alignment_mode == "syn":
                        F_proj = self._project(F_syn, P)
                        y_proj = y_flat
                    elif self.cov_alignment_mode == "hybrid":
                        F_proj_real = self._project(F_real, P)
                        F_proj_syn = self._project(F_syn, P)
                        
                        F_proj = torch.cat([F_proj_real, F_proj_syn], dim=0)
                        y_proj = torch.cat([y_flat, y_flat], dim=0)

                    mu_dict, Sigma_dict, n_dict = self._per_class_moments(F_proj, y_proj)

                    for c in mu_dict:
                        Z_c = F_proj[y_proj == c].detach()
                        n_kc[c] += n_dict[c]
                        s_kc[c] += Z_c.sum(dim=0)
                        Q_kc[c] += Z_c.T @ Z_c

                    L_cov = self._cov_loss(mu_dict, Sigma_dict, n_dict, gs)

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
        print(f"[Client {client_id}|{data_domain}] syn_skip={steps_skipped}/{steps_total} ({skip_pct:.0f}%)")

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