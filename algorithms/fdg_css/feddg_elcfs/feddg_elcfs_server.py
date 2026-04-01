import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import copy

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from .feddg_elcfs_client import FedDG_ELCFS_Client

class FedDG_ELCFS_Server(FedAvg_Server):
    def __init__(
        self, 
        meta_step_size=1e-3, 
        clip_value=100.0,
        cont_weight=0.1,
        hook_layer_name=None, 
        **kwargs
    ):
        self.meta_step_size = meta_step_size
        self.clip_value = clip_value
        self.cont_weight = cont_weight
        
        model_name = kwargs.get('model_name', '').lower()

        if hook_layer_name is None:
            if "bisenetv2" in model_name:
                self.hook_layer_name = 'bga'
            elif "topformer" in model_name:
                self.hook_layer_name = 'decode_head.fusion_conv'
            else:
                self.hook_layer_name = 'bga'
            print(f"[Server] Auto-configured FedDG hook: '{self.hook_layer_name}'")
        else:
            self.hook_layer_name = hook_layer_name

        super().__init__(**kwargs)

    def _init_worker_pool(self, **kwargs):
        print(f"[Server] Initializing {self.max_concurrent_clients} FedDG (ELCFS) workers via Ray ActorPool...")
        workers = []
        for _ in range(self.max_concurrent_clients):
            workers.append(
                FedDG_ELCFS_Client.remote(
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
                    meta_step_size=self.meta_step_size,
                    clip_value=self.clip_value,
                    cont_weight=self.cont_weight,
                    hook_layer_name=self.hook_layer_name,
                    **kwargs
                )
            )
        return workers