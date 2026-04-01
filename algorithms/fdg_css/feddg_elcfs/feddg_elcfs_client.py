import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torch.func import functional_call 
import ray

from algorithms.fdg_css.fedavg.fedavg_client import Base_FedAvg_Client

@ray.remote(num_gpus=0.2)
class FedDG_ELCFS_Client(Base_FedAvg_Client):
    def __init__(
        self, 
        meta_step_size=1e-3, 
        clip_value=100.0,
        cont_weight=0.1,
        hook_layer_name='bga', 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.meta_step_size = meta_step_size
        self.clip_value = clip_value
        self.cont_weight = cont_weight
        self.hook_layer_name = hook_layer_name
        
    def extract_contour_embedding(self, masks_y_k, Z_k):
        if Z_k.shape[2:] != masks_y_k.shape[2:]:
            Z_k = F.interpolate(Z_k, size=masks_y_k.shape[2:], mode='bilinear', align_corners=False)
            
        B, C, H, W = Z_k.shape
        _, num_classes, _, _ = masks_y_k.shape
        
        h_features = [] 
        
        for c in range(num_classes):
            y_k = masks_y_k[:, c:c+1, :, :] 
            numerator = torch.sum(Z_k * y_k, dim=(2, 3)) 
            denominator = torch.sum(y_k, dim=(2, 3)) + 1e-8
            
            h_k = numerator / denominator
            h_features.append(h_k)
            
        return h_features

    def generate_frequency_augmented_batch(self, images_x, external_amp_bank=None, L=0.1):
        fft_src = torch.fft.fft2(images_x, dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_src_shifted = torch.fft.fftshift(amp_src, dim=(-2, -1))
        
        if external_amp_bank is not None and len(external_amp_bank) > 0:
            idx = torch.randint(0, len(external_amp_bank), (1,)).item()
            amp_trg = external_amp_bank[idx].to(images_x.device)
            if amp_trg.dim() == 3:
                amp_trg = amp_trg.unsqueeze(0).expand_as(amp_src)
        else:
            amp_trg = torch.roll(amp_src, shifts=1, dims=0)
            
        amp_trg_shifted = torch.fft.fftshift(amp_trg, dim=(-2, -1))
        
        _, _, h, w = amp_src.shape
        b_h, b_w = int(h * L), int(w * L)
        c_h, c_w = h // 2, w // 2
        
        amp_src_shifted[:, :, c_h-b_h:c_h+b_h+1, c_w-b_w:c_w+b_w+1] = \
            amp_trg_shifted[:, :, c_h-b_h:c_h+b_h+1, c_w-b_w:c_w+b_w+1]
            
        amp_src_new = torch.fft.ifftshift(amp_src_shifted, dim=(-2, -1))
        fft_src_new = amp_src_new * torch.exp(1j * pha_src)
        
        images_fda = torch.fft.ifft2(fft_src_new, dim=(-2, -1)).real
        return torch.clamp(images_fda, 0, 1)

    def get_boundary_masks(self, masks, num_classes):
        valid_masks = torch.where(masks == 255, torch.zeros_like(masks), masks)
        one_hot = F.one_hot(valid_masks, num_classes).permute(0, 3, 1, 2).float() 
        
        dilation = F.max_pool2d(one_hot, kernel_size=5, stride=1, padding=2)
        erosion = -F.max_pool2d(-one_hot, kernel_size=3, stride=1, padding=1)
        
        contour = one_hot - erosion
        background = dilation - one_hot
        return contour, background

    def train(self, global_parameters, data_domain, client_id, freq_bank=None):
        self.load_dataset(data_domain)
        
        self.local_model.load_state_dict(global_parameters)
        self.local_model.to(self.device)
        self.local_model.train()

        self.optimizer = optim.AdamW(self.local_model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, 
            total_iters=self.num_epoch * self.max_steps_per_epch, 
            power=self.power
        )
        
        self.current_feat_head = None 
        def hook_fn(module, input, output):
            self.current_feat_head = output[-1] if isinstance(output, (list, tuple)) else output

        target_layer = dict(self.local_model.named_modules()).get(self.hook_layer_name, None)
        if target_layer is None:
            raise ValueError(f"Not found layer '{self.hook_layer_name}'")
        hook_handle = target_layer.register_forward_hook(hook_fn)

        for epoch in range(self.num_epoch):
            for step, (images, masks) in enumerate(self.train_dataloader):
                if step >= self.max_steps_per_epch: break
                
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                
                images_fda = self.generate_frequency_augmented_batch(images, external_amp_bank=freq_bank, L=0.1)
                num_classes = self.local_model.n_classes if hasattr(self.local_model, 'n_classes') else 19
                masks_y_bd, masks_y_bg = self.get_boundary_masks(masks, num_classes)
                
                outputs_inner = self.local_model(images)
                logits_inner = outputs_inner[0] if isinstance(outputs_inner, (tuple, list)) else outputs_inner
                
                loss_inner = self.criterion(logits_inner, masks)
                feat_inner = self.current_feat_head 

                grads = torch.autograd.grad(loss_inner, self.local_model.parameters(), retain_graph=True, allow_unused=True)
                
                fast_weights = OrderedDict()
                for (name, param), grad in zip(self.local_model.named_parameters(), grads):
                    if grad is not None:
                        fast_weights[name] = param - self.meta_step_size * torch.clamp(grad, -self.clip_value, self.clip_value)
                    else:
                        fast_weights[name] = param 

                outputs_outer = functional_call(self.local_model, fast_weights, (images_fda,))
                logits_outer = outputs_outer[0] if isinstance(outputs_outer, (tuple, list)) else outputs_outer
                
                loss_outer_ce = self.criterion(logits_outer, masks)
                feat_outer = self.current_feat_head 

                inner_contour_embs = self.extract_contour_embedding(masks_y_bd, feat_inner)
                inner_bg_embs = self.extract_contour_embedding(masks_y_bg, feat_inner)
                
                outer_contour_embs = self.extract_contour_embedding(masks_y_bd, feat_outer)
                outer_bg_embs = self.extract_contour_embedding(masks_y_bg, feat_outer)

                cont_loss = 0.0
                
                if hasattr(self, 'cont_loss_func'):
                    for c in range(1, num_classes): 
                        ct_em = torch.cat((inner_contour_embs[c], outer_contour_embs[c]), 0)
                        bg_em = torch.cat((inner_bg_embs[c], outer_bg_embs[c]), 0)
                        combined_em = torch.cat((ct_em, bg_em), 0)
                        
                        labels = torch.cat([torch.ones(ct_em.shape[0]), torch.zeros(bg_em.shape[0])]).to(self.device)
                        cont_loss += self.cont_loss_func(combined_em, labels)
                    
                    cont_loss = cont_loss / (num_classes - 1)

                total_loss = loss_inner + loss_outer_ce + self.cont_weight * cont_loss
                
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step() 

        hook_handle.remove()
        local_weights = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        num_samples_trained = min(self.max_steps_per_epch * self.batch_size, self.total_samples)
        
        return local_weights, num_samples_trained, client_id