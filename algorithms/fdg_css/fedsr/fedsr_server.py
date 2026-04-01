import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import copy
from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .fedsr_client import FedSR_Client

class FedSR_Server(FedAvg_Server):
    def __init__(self, z_dim=128, alpha=0.01, beta=0.001, hook_layer_name=None, feat_dim=None, **kwargs):
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta
        
        model_name = kwargs.get('model_name', '').lower()

        if hook_layer_name is None or feat_dim is None:
            if "bisenetv2" in model_name:
                self.hook_layer_name = 'bga'
                self.feat_dim = 128
                print(f"[Server] Auto-configured for BiSeNetV2 (hook: '{self.hook_layer_name}', feat_dim: {self.feat_dim})")
            elif "topformer" in model_name:
                self.hook_layer_name = 'decode_head.fusion_conv'
                self.feat_dim = 256
                print(f"[Server] Auto-configured for Topformer (hook: '{self.hook_layer_name}', feat_dim: {self.feat_dim})")
            else:
                self.hook_layer_name = 'bga'
                self.feat_dim = 128
                print(f"[Server] Warning: Unknown model '{model_name}'. Using default hook: 'bga', feat_dim: 128")
        else:
            self.hook_layer_name = hook_layer_name
            self.feat_dim = feat_dim

        super().__init__(**kwargs)

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} FedSR workers via Ray ActorPool...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedSR_Client.remote(
                    num_classes=self.num_classes,
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

                    z_dim=self.z_dim,
                    alpha=self.alpha,
                    beta=self.beta,
                    hook_layer_name=self.hook_layer_name, 
                    feat_dim=self.feat_dim,
                    **kwargs
                )
            )
        return workers