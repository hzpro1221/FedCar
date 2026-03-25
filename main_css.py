import sys
import os
import json 
import numpy as np 
import time
import gc

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import wandb

from algorithms.css.centralized.centralized import Centralized
from algorithms.css.spc_net.spc_net import SPC_Net 
from algorithms.css.sens_aug.sens_aug import SensAug

ALGORITHMS = ["centralized", "spc_net", "sens_aug"] 
# Leave-One-Domain-Out Setup
ALL_DOMAINS = ["cityscape", "gta5", "mapillary", "synthia", "bdd100"] 

TOTAL_EPOCHS = 500
BATCH_SIZE = 16

NUM_WORKERS = 4
NUM_SAMPLE = 2000
MAX_STEP_PER_EPCH = 100

INIT_LR = 1e-3 
MIN_LR = 2e-4
POWER = 0.9
WEIGHT_DECAY = 0.01
SEEDS = [2026]  

NUM_CLASSES = 19

CHECKPOINT_DIR = "checkpoints_css"
RESULTS_DIR = "results_css"
WANDB_PROJECT = "CSS" 

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiment_results = {}

    for algo in ALGORITHMS:
        print(f"\n{'='*60}")
        print(f"[Experiment] Starting runs for algorithm: {algo.upper()}")
        print(f"{'='*60}")
        
        experiment_results[algo] = {}

        for target_domain in ALL_DOMAINS:
            source_domains = [d for d in ALL_DOMAINS if d != target_domain]
            
            print(f"\n{'>'*50}")
            print(f"[LODO] Target Domain: {target_domain.upper()}")
            print(f"[LODO] Source Domains: {source_domains} (COMBINED)")
            print(f"{'>'*50}")

            experiment_results[algo][target_domain] = {
                "miou_list": [],
                "pixel_acc_list": []
            }
            
            for seed in SEEDS:
                print(f"\n--- Running {algo.upper()} | Target: {target_domain} | Seed: {seed} ---")
                
                checkpoint_filename = f"{algo}_target_{target_domain}_seed_{seed}.pth"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)

                run_name = f"{algo.upper()}_{target_domain}_s{seed}"
                wandb.init(
                    project=WANDB_PROJECT,
                    name=run_name,
                    group=algo.upper(),          
                    tags=["LODO", target_domain, "Centralized"], 
                    reinit=True,                
                    config={
                        "algorithm": algo,
                        "target_domain": target_domain,
                        "source_domains": source_domains,
                        "seed": seed,
                        "num_epochs": TOTAL_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "init_lr": INIT_LR,
                        "num_workers": NUM_WORKERS,             
                        "num_sample": NUM_SAMPLE,                
                        "max_steps_per_epch": MAX_STEP_PER_EPCH 
                    }
                )

                if algo == "centralized":
                    trainer = Centralized(
                        num_classes=NUM_CLASSES,
                        source_domains=source_domains,
                        num_epochs=TOTAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,              
                        num_sample=NUM_SAMPLE,                
                        max_steps_per_epch=MAX_STEP_PER_EPCH,  
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )
                elif algo == "spc_net":
                    trainer = SPC_Net(
                        num_classes=NUM_CLASSES,
                        source_domains=source_domains,
                        num_epochs=TOTAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,              
                        num_sample=NUM_SAMPLE,                
                        max_steps_per_epch=MAX_STEP_PER_EPCH, 
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY                        
                    )
                elif algo == "sens_aug":
                    trainer = SensAug(
                        num_classes=NUM_CLASSES,
                        source_domains=source_domains,
                        num_epochs=TOTAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,              
                        num_sample=NUM_SAMPLE,                 
                        max_steps_per_epch=MAX_STEP_PER_EPCH,  
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY                        
                    )
                else:
                    raise NotImplementedError(f"Algorithm '{algo}' is not implemented yet.")

                trainer.set_seed(seed)

                trainer.train(target_domain=target_domain, checkpoint_path=checkpoint_path)

                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)
                
                miou, pixel_acc, iou_per_class = trainer.evaluate(
                    target_domain=target_domain, 
                    checkpoint_path=checkpoint_path
                )

                wandb.log({
                    "Final_Test_mIoU": miou * 100,
                    "Final_Test_Pixel_Accuracy": pixel_acc * 100
                })

                experiment_results[algo][target_domain]["miou_list"].append(miou)
                experiment_results[algo][target_domain]["pixel_acc_list"].append(pixel_acc)

                print(f"[Main] Cleaning up {algo.upper()} trainer from VRAM...")

                del trainer
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
    results_file_path = os.path.join(RESULTS_DIR, f"result_algo_{algorithm_str}.json")
    
    with open(results_file_path, "w") as f:
        json.dump(experiment_results, f, indent=4)
    print(f"\n[Main] All LODO Centralized experiments complete. Results saved to {results_file_path}")

if __name__ == "__main__":
    main()