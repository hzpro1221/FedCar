import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import ray
from algorithms.fdg_css.fedavg.fedavg_client import Base_FedAvg_Client

@ray.remote(num_gpus=0.2)
class FedOMG_Client(Base_FedAvg_Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)