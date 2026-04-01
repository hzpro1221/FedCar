import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import copy
from tqdm import tqdm
import ray
import wandb 

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .our_client import FedCovMatch_Client

class FedCovMatch_Server(FedAvg_Server):
    def __init__(
        self, 
        lam_cov=1.0, 
        lam_syn=0.5, 
        lam_cons=0.3,
        cov_alignment_mode="real",
        entropy_threshold=1.8,
        ema_beta=0.6,
        ema_beta_per_round=0.06,
        feature_dim=256,
        proj_dim=64,
        use_qr=True,
        **kwargs
    ):
        self.lam_cov = lam_cov
        self.lam_syn = lam_syn
        self.lam_cons = lam_cons
        self.ema_beta = ema_beta
        self.ema_beta_per_round = ema_beta_per_round

        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.use_qr = use_qr

        self.cov_alignment_mode = cov_alignment_mode
        self.entropy_threshold = entropy_threshold
        self.global_stats = {}

        super().__init__(**kwargs)
        
        P_init = torch.randn(self.feature_dim, self.proj_dim)
        
        if self.use_qr:
            Q, _   = torch.linalg.qr(P_init)
            self.P_matrix = Q   
            print(f"[Server] Projection matrix: {self.feature_dim} → {self.proj_dim} (Orthonormal QR: ON)")
        else:
            self.P_matrix = P_init / (self.feature_dim ** 0.5)
            print(f"[Server] Projection matrix: {self.feature_dim} → {self.proj_dim} (Orthonormal QR: OFF)") 

        print(f"[Server] Lambdas: cov={lam_cov}, syn={lam_syn}, cons={lam_cons}")

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} FedCovMatch workers via Ray...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedCovMatch_Client.remote(
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
                    lam_cov=self.lam_cov,
                    lam_syn=self.lam_syn,
                    lam_cons=self.lam_cons,
                    cov_alignment_mode=self.cov_alignment_mode,
                    entropy_threshold=self.entropy_threshold,
                    proj_dim=self.proj_dim,
                    model_name=self.model_name, 
                    **kwargs
                )
            )
        return workers

    def update_global_stats(self, local_moments_list, round_idx):
        N = torch.zeros(self.num_classes, device=self.device)
        S = torch.zeros(self.num_classes, self.proj_dim, device=self.device)
        Q = torch.zeros(self.num_classes, self.proj_dim, self.proj_dim, device=self.device)

        for moments in local_moments_list:
            N += moments['n_kc'].to(self.device)
            S += moments['s_kc'].to(self.device)
            Q += moments['Q_kc'].to(self.device)

        momentum = min(self.ema_beta, (round_idx + 1) * self.ema_beta_per_round)
        n_updated = 0
        
        for c in range(self.num_classes):
            n_c = N[c].item()

            if n_c < 1:
                continue

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
            
            tasks = []
            for i, domain in enumerate(self.source_domains):
                tasks.append({
                    "global_parameters": global_weights,
                    "global_stats": self.global_stats,
                    "P_matrix": self.P_matrix,
                    "data_domain": domain,
                    "client_id": i
                })

            results = list(self.actor_pool.map(
                lambda actor, task: actor.train.remote(
                    global_parameters=task["global_parameters"],
                    global_stats=task["global_stats"],
                    P_matrix=task["P_matrix"],
                    data_domain=task["data_domain"],
                    client_id=task["client_id"]
                ),
                tasks
            ))

            local_weights_list  = [r[0] for r in results]
            total_samples_list  = [r[1] for r in results]
            local_moments_list  = [r[2] for r in results]

            global_weights = self.aggregate(local_weights_list, total_samples_list)
            self.update_global_model(global_weights)

            n_classes_updated = self.update_global_stats(local_moments_list, round_idx)

            total_skipped = sum(m.get('steps_skipped', 0) for m in local_moments_list)
            total_steps = sum(m.get('steps_total',   0) for m in local_moments_list)
            skip_pct = 100.0 * total_skipped / max(total_steps, 1)
            current_momentum = min(self.ema_beta, (round_idx + 1) * self.ema_beta_per_round)

            pbar.set_description(
                f"Round {round_idx + 1}/{self.num_rounds} | "
                f"stats: {n_classes_updated}/{self.num_classes} cls | "
                f"syn_skip: {skip_pct:.0f}%"
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