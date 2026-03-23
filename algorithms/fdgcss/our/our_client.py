import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import ray

from algorithms.fdgcss.our.utils.augment_dataloader import get_augmented_dataloader

@ray.remote(num_gpus=0.2)
class FedCovMatch_Client:
    def __init__(
        self,
        data,
        client_id,
        local_model,
        num_epoch,
        batch_size,
        init_lr,
        min_lr,
        power,
        weight_decay,
        lam_cov=1.0,      
        lam_syn=1.0,      
        lam_cons=1.0,     
        proj_dim=64,      
        num_classes=19,
        max_steps_per_epch=10 
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.data = data
        self.local_model = local_model.to(self.device)

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.power = power
        self.weight_decay = weight_decay
        self.max_steps_per_epch = max_steps_per_epch
        
        self.lam_cov = lam_cov
        self.lam_syn = lam_syn
        self.lam_cons = lam_cons
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        ds_map = {"cityscape": "Cityscapes", "bdd100": "BDD100K", "gta5": "GTA5", "mapillary": "Mapillary", "synthia": "Synthia"}
        
        self.syn_dataloader = get_augmented_dataloader(
            root_dir="/root/KhaiDD/FedCar/dataset/augment_data",
            dataset_names=[ds_map[self.data]], 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.num_samples = len(self.syn_dataloader.dataset)

        self.optimizer = optim.AdamW(self.local_model.parameters(), lr=init_lr, weight_decay=weight_decay)
        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion_cons = nn.MSELoss() 
        self.scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=num_epoch, power=self.power)

    def train(self, global_parameters, global_stats, P_matrix):
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()
        P_matrix = P_matrix.to(self.device)
        
        for c in range(self.num_classes):
            if c in global_stats:
                global_stats[c]['Sigma'] = global_stats[c]['Sigma'].to(self.device)

        n_kc = torch.zeros(self.num_classes, device=self.device)
        s_kc = torch.zeros(self.num_classes, self.proj_dim, device=self.device)
        Q_kc = torch.zeros(self.num_classes, self.proj_dim, self.proj_dim, device=self.device)
        
        for epoch in range(self.num_epoch):
            for step, (x_real, x_syn, y) in enumerate(self.syn_dataloader):
                if step > self.max_steps_per_epch: 
                    break
                
                x_real, x_syn, y = x_real.to(self.device), x_syn.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                logits_real, F_real = self.local_model.forward_features(x_real)
                logits_syn, F_syn = self.local_model.forward_features(x_syn)
                
                L_real = self.criterion_seg(logits_real, y)
                L_syn = self.criterion_seg(logits_syn, y) 
                L_cons = self.criterion_cons(logits_real, logits_syn)
                
                L_cov = torch.tensor(0.0, device=self.device)
                dist_syn_total = torch.tensor(0.0, device=self.device)
                valid_syn_classes = 0
                
                B, D, H_f, W_f = F_real.shape
                
                # downsample labels
                y_float = y.unsqueeze(1).float() 
                y_downsampled = F.interpolate(y_float, size=(H_f, W_f), mode='nearest').squeeze(1).long() 
                y_flat = y_downsampled.reshape(-1) 
                
                # flatten Features
                F_flat_real = F_real.permute(0, 2, 3, 1).reshape(-1, D) 
                F_flat_syn = F_syn.permute(0, 2, 3, 1).reshape(-1, D)

                for c in range(self.num_classes):
                    if c == 255: 
                        continue
                    
                    mask_c = (y_flat == c) 
                    
                    Z_kc_real = F_flat_real[mask_c] 
                    
                    if Z_kc_real.size(0) >= 2: 
                        Z_tilde_real = torch.matmul(Z_kc_real, P_matrix) 
                        mu_batch_real = Z_tilde_real.mean(dim=0)
                        Z_centered_real = Z_tilde_real - mu_batch_real
                        Sigma_batch_real = torch.matmul(Z_centered_real.T, Z_centered_real) / (Z_tilde_real.size(0) - 1)
                        
                        if c in global_stats:
                            L_cov += torch.norm(Sigma_batch_real - global_stats[c]['Sigma'], p='fro')**2
                            
                        n_kc[c] += Z_tilde_real.size(0)
                        s_kc[c] += Z_tilde_real.sum(dim=0)
                        Q_kc[c] += torch.matmul(Z_tilde_real.T, Z_tilde_real)

                    if c in global_stats:
                        Z_kc_syn = F_flat_syn[mask_c]
                        
                        if Z_kc_syn.size(0) >= 2:
                            Z_tilde_syn = torch.matmul(Z_kc_syn, P_matrix)
                            mu_batch_syn = Z_tilde_syn.mean(dim=0)
                            Z_centered_syn = Z_tilde_syn - mu_batch_syn
                            Sigma_batch_syn = torch.matmul(Z_centered_syn.T, Z_centered_syn) / (Z_tilde_syn.size(0) - 1)
                            
                            dist_syn_total += torch.norm(Sigma_batch_syn - global_stats[c]['Sigma'], p='fro')**2
                            valid_syn_classes += 1

                if valid_syn_classes > 0 and global_stats:
                    mean_dist_syn = dist_syn_total / valid_syn_classes
                    dynamic_syn_weight = torch.exp(-mean_dist_syn) 
                else:
                    dynamic_syn_weight = torch.tensor(1.0, device=self.device)

                total_Loss = L_real + self.lam_cov * L_cov \
                           + dynamic_syn_weight * (self.lam_syn * L_syn + self.lam_cons * L_cons)
                
                total_Loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        moments = {
            'n_kc': n_kc.cpu(),
            's_kc': s_kc.cpu(),
            'Q_kc': Q_kc.cpu()
        }
        return local_weights, self.num_samples, moments