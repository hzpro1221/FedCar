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

from algorithms.fdg_css.fedavg.fedavg_server import FedAvg_Server
from algorithms.fdg_css.fedavg.segformer_b0_avg import SegFormerB0_Avg

from algorithms.fdg_css.fedsr.fedsr_server import FedSR_Server
from algorithms.fdg_css.fedsr.segformer_b0_sr import SegFormerB0_SR

from algorithms.fdg_css.fedavg_ga.fedavg_ga_server import FedAvg_GA_Server
from algorithms.fdg_css.fedavg_ga.segformer_b0_avg_ga import SegFormerB0_Avg_GA

from algorithms.fdg_css.fedavg_omg.fedavg_omg_server import FedAvg_OMG_Server
from algorithms.fdg_css.fedavg_omg.segformer_b0_avg_omg import SegFormerB0_Avg_OMG

from algorithms.fdg_css.feddg.feddg_server import FedDG_Server
from algorithms.fdg_css.feddg.segformer_b0_dg import SegFormerB0_DG

from algorithms.fdg_css.gperxan.gperxan_server import gPerXAN_Server
from algorithms.fdg_css.gperxan.segformer_b0_gperxan import SegFormerB0_gPerXAN

# ==========================================
# EXPERIMENT CONFIGURATIONS
# ==========================================
ALGORITHMS = ["fedavg"] # ["fedavg", "fedsr", "fedavg+ga", "fedavg+omg", "feddg", "gperxan"] 

# Leave-One-Domain-Out Setup
ALL_DOMAINS = ["cityscape", "gta5", "mapillary", "synthia", "bdd100"] # ["cityscape", "gta5", "mapillary", "synthia", "bdd100"]

NUM_ROUNDS = 40 
NUM_EPOCHS = 5
BATCH_SIZE = 4

NUM_WORKERS = 4
NUM_SAMPLE = 2000
MAX_STEP_PER_EPCH = 200

INIT_LR = 1e-3 
MIN_LR = 2e-4
POWER = 0.9
WEIGHT_DECAY = 0.01
SEEDS = [2026]  

# Hard fix
NUM_CLASSES = 19

CHECKPOINT_DIR = "checkpoints_fdg_css"
RESULTS_DIR = "results_fdg_css"
# ==========================================

def main():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiment_results = {}

    for algo in ALGORITHMS:
        print(f"\n{'='*60}")
        print(f"[Experiment] Starting runs for algorithm: {algo.upper()}")
        print(f"{'='*60}")
        
        experiment_results[algo] = {}

        # --- LEAVE-ONE-DOMAIN-OUT LOOP ---
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
                print(f"\n--- Running {algo.upper()} | Target: {target_domain} | Seed: {seed} ---")
                
                checkpoint_filename = f"{algo}_target_{target_domain}_seed_{seed}.pth"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)

                if algo == "fedavg":
                    global_backbone = SegFormerB0_Avg(num_classes=NUM_CLASSES)
                    
                    server = FedAvg_Server(
                        num_classes=NUM_CLASSES,
                        backbone_model=global_backbone,
                        source_domains=source_domains,
                        
                        num_rounds=NUM_ROUNDS,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        
                        num_workers=NUM_WORKERS,
                        num_sample=NUM_SAMPLE,
                        max_steps_per_epch=MAX_STEP_PER_EPCH,
                        
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )
                elif algo == "fedsr":
                    global_backbone = SegFormerB0_SR(num_classes=NUM_CLASSES)    

                    server = FedSR_Server(
                        num_classes=NUM_CLASSES,
                        backbone_model=global_backbone,
                        source_domains=source_domains,
                        num_rounds=NUM_ROUNDS,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )
                elif algo == "fedavg+ga":
                    global_backbone = SegFormerB0_Avg_GA(num_classes=NUM_CLASSES)    

                    server = FedAvg_GA_Server(
                        num_classes=NUM_CLASSES,
                        backbone_model=global_backbone,
                        source_domains=source_domains,
                        num_rounds=NUM_ROUNDS,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )
                elif algo == "fedavg+omg":
                    global_backbone = SegFormerB0_Avg_OMG(num_classes=NUM_CLASSES)    

                    server = FedAvg_OMG_Server(
                        num_classes=NUM_CLASSES,
                        backbone_model=global_backbone,
                        source_domains=source_domains,
                        num_rounds=NUM_ROUNDS,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )   
                elif algo == "feddg":
                    global_backbone = SegFormerB0_DG(num_classes=NUM_CLASSES)    

                    server = FedDG_Server(
                        num_classes=NUM_CLASSES,
                        backbone_model=global_backbone,
                        source_domains=source_domains,
                        num_rounds=NUM_ROUNDS,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )   
                elif algo == "gperxan":
                    global_backbone = SegFormerB0_gPerXAN(num_classes=NUM_CLASSES)    

                    server = gPerXAN_Server(
                        num_classes=NUM_CLASSES,
                        backbone_model=global_backbone,
                        source_domains=source_domains,
                        num_rounds=NUM_ROUNDS,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        init_lr=INIT_LR,
                        min_lr=MIN_LR,
                        power=POWER,
                        weight_decay=WEIGHT_DECAY
                    )                                                           
                else:
                    raise NotImplementedError(f"Algorithm '{algo}' is not implemented yet.")

                server.set_seed(seed)

                server.train(checkpoint_path=checkpoint_path)

                print("\n[Main] Terminating Ray clients to free VRAM for evaluation...")
                if hasattr(server, 'clients'):
                    for client in server.clients:
                        ray.kill(client) 
                    server.clients.clear() 
                
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)
                print("[Main] VRAM cleared successfully.")

                miou, pixel_acc, iou_per_class = server.evaluate(
                    target_domain=target_domain, 
                    checkpoint_path=checkpoint_path
                )

                experiment_results[algo][target_domain]["miou_list"].append(miou)
                experiment_results[algo][target_domain]["pixel_acc_list"].append(pixel_acc)

                print(f"[Main] Cleaning up {algo.upper()} server and backbone from VRAM...")
                
                del server
                del global_backbone

                gc.collect()
                torch.cuda.empty_cache()
                
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
    print(f"\n[Server] All LODO experiments complete. Results saved to {results_file_path}")

    ray.shutdown()

if __name__ == "__main__":
    main()