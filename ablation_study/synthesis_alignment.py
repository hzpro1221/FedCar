import os
import sys
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

from algorithms.fdgcss.our.our_server import FedCovMatch_Server

wandb.login(key="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")

MODEL_NAME = "topformer"  # "bisenetv2" or "topformer"

COV_MODES = ["none", "hybrid", "real", "syn"] 
PROJ_DIM = 128

ALL_DOMAINS = ["cityscape", "gta5", "mapillary"] 

NUM_ROUNDS = 100
NUM_EPOCHS = 3
BATCH_SIZE = 8

NUM_WORKERS = 4
MAX_CONCURRENT_CLIENTS = 1

NUM_SAMPLE = 2000
MAX_STEP_PER_EPCH = 20

INIT_LR = 1e-3 
MIN_LR = 2e-4
POWER = 0.9
WEIGHT_DECAY = 0.01
SEEDS = [2026]  

NUM_CLASSES = 19

CHECKPOINT_DIR = "checkpoints_ablation_cov"
RESULTS_DIR = "results_ablation_cov"
WANDB_PROJECT = "FDGCSS_Ablation_Cov_Alignment"

def main():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiment_results = {}
    
    feature_dim = 128 if MODEL_NAME == 'bisenetv2' else 256

    print(f"\n{'='*60}")
    print(f"[Experiment] Starting Ablation Study: Covariance Alignment Source | Model: {MODEL_NAME.upper()}")
    print(f"Modes being tested: {COV_MODES}")
    print(f"{'='*60}")

    for target_domain in [d for d in ALL_DOMAINS if d == "gta5"]:
        source_domains = [d for d in ALL_DOMAINS if d != target_domain]
        
        print(f"\n{'>'*50}")
        print(f"[LODO] Target Domain: {target_domain.upper()}")
        print(f"[LODO] Source Domains: {source_domains}")
        print(f"{'>'*50}")

        experiment_results[target_domain] = {}

        for mode in COV_MODES:
            mode_label = f"Cov_{mode.upper()}"
            
            experiment_results[target_domain][mode_label] = {
                "miou_list": [],
                "pixel_acc_list": []
            }

            for seed in SEEDS:
                print(f"\n--- Target: {target_domain} | Mode: {mode_label} | Seed: {seed} ---")
                
                checkpoint_filename = f"our_{MODEL_NAME}_target_{target_domain}_{mode_label}_seed_{seed}.pth"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
                
                run_name = f"{mode_label}_{MODEL_NAME.upper()}_{target_domain}_s{seed}"
                
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

                wandb.init(
                    project=WANDB_PROJECT,
                    name=run_name,
                    group=f"Ablation_{mode_label}",          
                    tags=["Ablation", "Cov_Alignment", target_domain, mode, MODEL_NAME], 
                    reinit=True,                
                    config={
                        "algorithm": "our",
                        "model_name": MODEL_NAME,
                        "target_domain": target_domain,
                        "source_domains": source_domains,
                        "seed": seed,
                        "feature_dim": feature_dim,
                        "proj_dim": PROJ_DIM,
                        "cov_alignment_mode": mode,
                        **base_server_kwargs
                    }
                )

                server = FedCovMatch_Server(
                    **base_server_kwargs,
                    lam_cov=1.0,
                    lam_syn=0.5,
                    lam_cons=0.3,
                    feature_dim=feature_dim,
                    proj_dim=PROJ_DIM,
                    cov_alignment_mode=mode, 
                    use_qr=True 
                )

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

                experiment_results[target_domain][mode_label]["miou_list"].append(miou)
                experiment_results[target_domain][mode_label]["pixel_acc_list"].append(pixel_acc)

                print(f"[Main] Cleaning up Server from VRAM...")
                
                del server

                gc.collect()
                torch.cuda.empty_cache()
                
                wandb.finish()
                print("[Main] VRAM cleared successfully. Ready for the next run.")

            miou_mean = np.mean(experiment_results[target_domain][mode_label]["miou_list"])
            miou_std = np.std(experiment_results[target_domain][mode_label]["miou_list"])
            
            acc_mean = np.mean(experiment_results[target_domain][mode_label]["pixel_acc_list"])
            acc_std = np.std(experiment_results[target_domain][mode_label]["pixel_acc_list"])

            experiment_results[target_domain][mode_label]["aggregated"] = {
                "miou_mean": miou_mean,
                "miou_std": miou_std,
                "pixel_acc_mean": acc_mean,
                "pixel_acc_std": acc_std
            }

            print(f"\n{'='*40}")
            print(f"Results for Target: {target_domain.upper()} | Mode: {mode_label}")
            print(f"- mIoU: {miou_mean*100:.2f}% ± {miou_std*100:.2f}%")
            print(f"- Pixel Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
            print(f"{'='*40}")

    results_file_path = os.path.join(RESULTS_DIR, f"ablation_cov_alignment_results_{MODEL_NAME}.json")
    
    with open(results_file_path, "w") as f:
        json.dump(experiment_results, f, indent=4)
    print(f"\n[Server] Ablation Study complete. Results saved to {results_file_path}")

    ray.shutdown()

if __name__ == "__main__":
    main()