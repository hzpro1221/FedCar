import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import copy
import numpy as np

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .fedomg_client import FedOMG_Client

class FedOMG_Server(FedAvg_Server):
    def __init__(
        self, 
        global_lr=1.0,
        omg_lr=0.1,
        omg_momentum=0.9,
        omg_num_iter=10,
        kappa=0.5,
        **kwargs
    ):
        self.global_lr = global_lr
        self.omg_lr = omg_lr
        self.omg_momentum = omg_momentum
        self.omg_num_iter = omg_num_iter
        self.kappa = kappa
        
        super().__init__(**kwargs)

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} FedOMG workers via Ray...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedOMG_Client.remote(
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

    def OMG(self, grad_vec, weights_tensor):
        grads = grad_vec 
        num_tasks = grads.shape[0]
        
        GG = grads.mm(grads.t()) 
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG_norm = GG / scale.pow(2)
        
        p = weights_tensor.to(grads.device) 
        g_ref = torch.matmul(p.t(), grads).squeeze(0) 
        norm_g_ref = torch.norm(g_ref) + 1e-8
        
        # Tiền tính toán một vài hằng số để tối ưu
        GG_norm_p = torch.matmul(GG_norm, p)
        g_ref_norm_sq = torch.matmul(p.t(), GG_norm_p).squeeze()
        
        w = torch.zeros(num_tasks, 1, requires_grad=True, device=grads.device)
        w_opt = torch.optim.SGD([w], lr=self.omg_lr, momentum=self.omg_momentum)
        
        best_loss = np.inf
        w_best = w.clone().detach() # Khởi tạo an toàn hơn
        
        for i in range(self.omg_num_iter + 1):
            w_opt.zero_grad()

            Gamma = torch.softmax(w, dim=0)

            # Tính các đại lượng cho loss
            gamma_norm_sq = torch.matmul(Gamma.t(), torch.matmul(GG_norm, Gamma)).squeeze()
            dot_product_gamma_ref = torch.matmul(Gamma.t(), GG_norm_p).squeeze()
            
            # Loss chính: ||Gamma*G - p*G||^2  (Dạng mở triển)
            # Thay vì tính loằng ngoằng, ta gom lại:
            distance_sq = gamma_norm_sq - 2 * dot_product_gamma_ref + g_ref_norm_sq
            
            # Đảm bảo loss không bao giờ bị âm do sai số float
            distance_sq = torch.clamp(distance_sq, min=0.0)
            
            loss = distance_sq
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                w_best = w.clone()
                
            if i < self.omg_num_iter:
                loss.backward()
                w_opt.step()

        Gamma_opt = torch.softmax(w_best.detach(), dim=0)
        
        # --- CẬP NHẬT CÁCH TÍNH G_FINAL CHO ĐÚNG BẢN CHẤT OMG ---
        g_Gamma = torch.matmul(Gamma_opt.t(), grads).squeeze(0)
        norm_g_Gamma = torch.norm(g_Gamma) + 1e-8
        
        lambda_coef = self.kappa * (norm_g_ref / norm_g_Gamma)
        g_final = g_ref + lambda_coef * g_Gamma
             
        print(f"[FedOMG] Optimized weights Γ: {Gamma_opt.squeeze().cpu().numpy()}")
        print(f"[FedOMG] Best inner optimization loss: {best_loss:.6f}")
        
        return g_final
    
    def aggregate(self, local_weights_list, total_samples_list):
        print("[Server] Starting FedOMG aggregation...")
        num_tasks = len(local_weights_list)
        total_samples = sum(total_samples_list)
        
        weights = [n_k / total_samples for n_k in total_samples_list]
        print(f"[Server] Data proportion (p): {[f'{w:.4f}' for w in weights]}")

        global_state_dict = self.backbone_model.state_dict()
        param_names = [name for name, param in self.backbone_model.named_parameters()]
        buffer_names = [name for name, buf in self.backbone_model.named_buffers()]

        all_domain_grads = []
        with torch.no_grad():
            for i in range(num_tasks):
                local_w = local_weights_list[i]
                domain_grad_diff = []
                for name in param_names:
                    diff = (global_state_dict[name].to(self.device) - local_w[name].to(self.device))
                    domain_grad_diff.append(diff.view(-1))
                
                domain_grad_vector = torch.cat(domain_grad_diff)
                all_domain_grads.append(domain_grad_vector)
            
        all_domain_grads_tensor = torch.stack(all_domain_grads)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=all_domain_grads_tensor.device).view(-1, 1)
        
        omg_grads = self.OMG(all_domain_grads_tensor, weights_tensor)
        new_global_weights = copy.deepcopy(global_state_dict)
        
        offset = 0
        for name in param_names:
            param_shape = global_state_dict[name].shape
            param_size = global_state_dict[name].numel()
            
            updated_param_delta = omg_grads[offset : offset + param_size].view(param_shape)
            
            new_global_weights[name] = global_state_dict[name].to(self.device) - (updated_param_delta * self.global_lr)
            offset += param_size

        for name in buffer_names:
            avg_buf = torch.zeros_like(global_state_dict[name], dtype=torch.float32).to(self.device)
            for i in range(num_tasks):
                avg_buf += local_weights_list[i][name].to(self.device).to(torch.float32) * weights[i]
            new_global_weights[name] = avg_buf.to(global_state_dict[name].dtype)

        print("[Server] FedOMG Aggregation complete.")
        return new_global_weights