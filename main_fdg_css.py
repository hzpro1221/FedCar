import sys
import os
import json 
import numpy as np 
import time
import gc

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import ray
import torch
import wandb

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from algorithms.fdg_css.fedsr.fedsr_server import FedSR_Server
from algorithms.fdg_css.feddg_ga.feddg_ga_server import FedDG_GA_Server
from algorithms.fdg_css.fedomg.fedomg_server import FedOMG_Server
from algorithms.fdg_css.feddg_elcfs.feddg_elcfs_server import FedDG_ELCFS_Server
from algorithms.fdg_css.gperxan.gperxan_server import gPerXAN_Server

from algorithms.fdgcss.sinobn_lab.silobn_lab_server import SiloBN_LAB_Server
from algorithms.fdgcss.fedema.fedema_server import FedEMA_Server
from algorithms.fdgcss.our.our_server import FedCovMatch_Server

wandb.login(key="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")

# "feddg_elcfs" -> meta-learning, run later

ALGORITHMS =  ["our"]  # ["fedavg", "fedsr", "feddg_ga", "fedomg", "gperxan", "sinobn_lab", "fedema", "our"]
MODEL_NAME = "topformer"  # "bisenetv2" or "topformer"

# Leave-One-Domain-Out Setup
ALL_DOMAINS = ["cityscape", "gta5", "mapillary", "synthia", "bdd100"] 

NUM_ROUNDS = 120 
NUM_EPOCHS = 5
BATCH_SIZE = 16

NUM_WORKERS = 4
MAX_CONCURRENT_CLIENTS = 1

NUM_SAMPLE = 2000
MAX_STEP_PER_EPCH = 100

INIT_LR = 1e-3 
MIN_LR = 2e-4
POWER = 0.9
WEIGHT_DECAY = 0.01
SEEDS = [2026]  

NUM_CLASSES = 19

CHECKPOINT_DIR = "checkpoints_fdg_css"
RESULTS_DIR = "results_fdg_css"
WANDB_PROJECT = "FDGCSS"

def main():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiment_results = {}

    for algo in ALGORITHMS:
        print(f"\n{'='*60}")
        print(f"[Experiment] Starting runs for algorithm: {algo.upper()} | Model: {MODEL_NAME.upper()}")
        print(f"{'='*60}")
        
        experiment_results[algo] = {}

        for target_domain in ALL_DOMAINS:
            source_domains = [d for d in ALL_DOMAINS if d != target_domain]
            
            print(f"\n{'>'*50}")
            print(f"[LODO] Target Domain: {target_domain.upper()}")
            print(f"[LODO] Source Domains: {source_domains}")
            print(f"{'>'*50}")

            experiment_results[algo][target_domain] = {
                "miou_list": [],
                "pixel_acc_list": []
            }

            for seed in SEEDS:
                print(f"\n--- Running {algo.upper()} | Model: {MODEL_NAME} | Target: {target_domain} | Seed: {seed} ---")
                
                checkpoint_filename = f"{algo}_{MODEL_NAME}_target_{target_domain}_seed_{seed}.pth"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
                
                run_name = f"{algo.upper()}_{MODEL_NAME.upper()}_{target_domain}_s{seed}"
                wandb.init(
                    project=WANDB_PROJECT,
                    name=run_name,
                    group=algo.upper(),          
                    tags=["LODO", target_domain, MODEL_NAME], 
                    reinit=True,                
                    config={
                        "algorithm": algo,
                        "model_name": MODEL_NAME,
                        "target_domain": target_domain,
                        "source_domains": source_domains,
                        "seed": seed,
                        "num_rounds": NUM_ROUNDS,
                        "num_epochs": NUM_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "init_lr": INIT_LR,
                        "max_steps_per_epch": MAX_STEP_PER_EPCH,
                        "max_concurrent_clients": MAX_CONCURRENT_CLIENTS
                    }
                )
                
                base_server_kwargs = dict(
                    num_classes=NUM_CLASSES,
                    model_name=MODEL_NAME, 
                    source_domains=source_domains,
                    num_rounds=NUM_ROUNDS,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    max_concurrent_clients=MAX_CONCURRENT_CLIENTS,
                    num_sample=NUM_SAMPLE,
                    max_steps_per_epch=MAX_STEP_PER_EPCH,
                    init_lr=INIT_LR,
                    min_lr=MIN_LR,
                    power=POWER,
                    weight_decay=WEIGHT_DECAY
                )

                if algo == "fedavg":
                    server = FedAvg_Server(**base_server_kwargs)
                elif algo == "fedsr":
                    server = FedSR_Server(
                        **base_server_kwargs,
                        z_dim=256,       
                        alpha=0.01,    
                        beta=0.001     
                    )
                elif algo == "feddg_ga":
                    server = FedDG_GA_Server(
                        **base_server_kwargs,
                        ga_step_size=0.05
                    )
                elif algo == "fedomg":
                    server = FedOMG_Server(
                        **base_server_kwargs,
                        global_lr=1.0,               
                        omg_lr=0.1,                 
                        omg_momentum=0.9,           
                        omg_num_iter=5,
                        kappa=0.1
                    )   
                elif algo == "feddg_elcfs":
                    server = FedDG_ELCFS_Server(
                        **base_server_kwargs,
                        meta_step_size=0.001,  
                        clip_value=100.0,       
                        cont_weight=0.1        
                    )
                elif algo == "gperxan":
                    server = gPerXAN_Server(
                        **base_server_kwargs,
                        reg_weight=0.01
                    )     
                elif algo == "sinobn_lab":
                    server = SiloBN_LAB_Server(
                        **base_server_kwargs,
                        hnm_perc=0.25
                    )
                elif algo == "fedema":
                    server = FedEMA_Server(
                        **base_server_kwargs,
                        beta=0.9,
                        lambda_ent=0.1
                    )
                elif algo == "our":
                    server = FedCovMatch_Server(
                        **base_server_kwargs,
                        lam_cov=1.0,
                        lam_syn=0.5,
                        lam_cons=0.3,
                        feature_dim=128 if MODEL_NAME == 'bisenetv2' else 256, 
                        proj_dim=128
                    )
                else:
                    raise NotImplementedError(f"Algorithm '{algo}' is not implemented yet.")

                server.set_seed(seed)
                
                server.train(target_domain=target_domain, checkpoint_path=checkpoint_path)

                miou, pixel_acc, iou_per_class = server.evaluate(
                    target_domain=target_domain, 
                    checkpoint_path=checkpoint_path
                )

                wandb.log({
                    "Final_Test_mIoU": miou * 100,
                    "Final_Test_Pixel_Accuracy": pixel_acc * 100
                })

                experiment_results[algo][target_domain]["miou_list"].append(miou)
                experiment_results[algo][target_domain]["pixel_acc_list"].append(pixel_acc)

                print(f"[Main] Cleaning up {algo.upper()} server from VRAM...")
                
                del server 
                
                gc.collect()
                torch.cuda.empty_cache()
                
                wandb.finish()
                print("[Main] Done. Ready for the next run.")

            miou_mean = np.mean(experiment_results[algo][target_domain]["miou_list"])
            miou_std = np.std(experiment_results[algo][target_domain]["miou_list"])
            
            acc_mean = np.mean(experiment_results[algo][target_domain]["pixel_acc_list"])
            acc_std = np.std(experiment_results[algo][target_domain]["pixel_acc_list"])

            experiment_results[algo][target_domain]["aggregated"] = {
                "miou_mean": miou_mean,
                "miou_std": miou_std,
                "pixel_acc_mean": acc_mean,
                "pixel_acc_std": acc_std
            }

            print(f"\n{'='*40}")
            print(f"Final Aggregated Results for Target: {target_domain.upper()}")
            print(f"- mIoU: {miou_mean*100:.2f}% ± {miou_std*100:.2f}%")
            print(f"- Pixel Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
            print(f"{'='*40}")

    algorithm_str = "_".join(ALGORITHMS)
    results_file_path = os.path.join(RESULTS_DIR, f"result_{MODEL_NAME}_{algorithm_str}.json")
    
    with open(results_file_path, "w") as f:
        json.dump(experiment_results, f, indent=4)
    print(f"\n[Main] All LODO experiments complete. Results saved to {results_file_path}")

    ray.shutdown()

if __name__ == "__main__":
    main()