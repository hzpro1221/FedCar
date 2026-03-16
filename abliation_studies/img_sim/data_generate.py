import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
import os
import math
import glob
import re
import numpy as np
import json
import itertools
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import (
    process_images_edges,
    get_pytorch_edges
)

def get_rgb_files(folder_path, max_samples):
    if not os.path.exists(folder_path):
        print(f"Warning: Directory not found: {folder_path}")
        return []
    search_pattern = os.path.join(folder_path, "*_leftImg8bit.png")
    return sorted(glob.glob(search_pattern))[:max_samples]

def get_mean_image_embedding(files, batch_size, model, processor, device="cuda"):
    print(f"\n[CLIP] Calculating mean image embedding from {len(files)} selected samples...")
    
    if not files:
        raise ValueError("No images provided for CLIP embedding calculation.")

    all_embeds = []
    for i in tqdm(range(0, len(files), batch_size), desc="Processing Image Batches"):
        batch_files = files[i:i+batch_size]
        images = [Image.open(f).convert("RGB") for f in batch_files]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeds = model.get_image_features(**inputs)
            if not isinstance(embeds, torch.Tensor):
                if hasattr(embeds, "image_embeds"):
                    embeds = embeds.image_embeds
                elif hasattr(embeds, "pooler_output"):
                    embeds = embeds.pooler_output
                else:
                    embeds = embeds[0] 
                    
            all_embeds.append(embeds)
            
    all_embeds = torch.cat(all_embeds, dim=0)
    mean_embed = all_embeds.mean(dim=0, keepdim=True)
    return F.normalize(mean_embed, p=2, dim=-1), F.normalize(all_embeds, p=2, dim=-1)

def find_domain_prompts(mean_img_embed, keywords, batch_size, num_domain, angle_deg, max_combinations, model, processor, device="cuda"):
    print(f"\n[CLIP] Searching for {num_domain} prompts with angle {angle_deg} degrees...")
    target_cos = np.cos(np.radians(angle_deg))
    
    template = "a photo of a {} city"
    candidates = []
    
    total_combinations = sum(math.comb(len(keywords), r) for r in range(1, max_combinations + 1))
    with tqdm(total=total_combinations, desc="Generating Candidate Prompts") as pbar:
        for r in range(1, max_combinations + 1):
            for combo in itertools.combinations(keywords, r):
                candidates.append(template.format(", ".join(combo)))
                pbar.update(1)

    print(f"Generated {len(candidates)} candidate prompts from keywords (combinations 1 to {max_combinations}).")
    
    all_text_embeds = []
    for i in tqdm(range(0, len(candidates), batch_size), desc="Encoding Candidates"):
        batch = candidates[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
            if not isinstance(embeds, torch.Tensor):
                if hasattr(embeds, "text_embeds"):
                    embeds = embeds.text_embeds
                elif hasattr(embeds, "pooler_output"):
                    embeds = embeds.pooler_output
                else:
                    embeds = embeds[0]

            all_text_embeds.append(embeds)

    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_text_embeds = F.normalize(all_text_embeds, p=2, dim=-1)

    selected_prompts = []
    selected_embeds = []
    mean_img_embed = mean_img_embed.to(device)
    available_indices = list(range(len(candidates)))

    for i in tqdm(range(num_domain), desc="Finding Domains"):
        cand_embeds = all_text_embeds[available_indices]
        
        cos_img = torch.matmul(cand_embeds, mean_img_embed.T).squeeze(-1)
        errors = torch.abs(cos_img - target_cos)
        
        for prev_embed in selected_embeds:
            cos_prev = torch.matmul(cand_embeds, prev_embed.T).squeeze(-1)
            errors += torch.pow(F.relu(cos_prev - 0.8), 2) * 5
        
        best_local_idx = torch.argmin(errors).item()
        best_global_idx = available_indices[best_local_idx]
        
        selected_prompts.append(candidates[best_global_idx])
        selected_embeds.append(all_text_embeds[best_global_idx:best_global_idx+1])
        
        available_indices.pop(best_local_idx)
        tqdm.write(f" -> Found Domain {i+1}: '{selected_prompts[-1]}' (Total Error: {errors[best_local_idx]:.4f})")
        
    return selected_prompts

def get_conversion_matrix(
    keywords, 
    max_combinations,
    clip_model,
    clip_processor,
    sd_tokenizer,
    sd_text_encoder    
):
    
    print(f"\n[Alignment] Generating {num_anchors} random anchor prompts...")
    template = "a photo of a {} city"
    anchors = set()

if __name__ == "__main__":
    # ================= SETTINGS =================
    WIDTH, HEIGHT = 512, 512
    NUMBER_INFERENCE_STEP = 20
    NUMBER_INFERENCE_BASE_STEP = 20
    
    CONTROLNET_CONDITIONING_SCALE = 2.0
    STRENGTH = 0.8
    INITIAL_EDGE_THRESHOLD = 0.1
    
    # number of data used to augmenting
    NUM_SAMPLES = 1
    NUM_GEN_PER_PROMPT = 4
        
    # path to image & label
    IMAGE_FOLDER = "/root/KhaiDD/FedCar/dataset/cityscape/leftImg8bit/train/aachen"
    LABEL_FOLDER = "/root/KhaiDD/FedCar/dataset/cityscape/gtFine/train/aachen" 

    CLIP_BATCH_SIZE = 266
    
    # the maximum different between prompt and mean representation of original image   
    IMAGE_EMBED_SIMILARITY = 0.9

    # used to create conversion matrix
    MAX_COMBINATIONS = 5
    
    BASE_OUTPUT_DIR = f"/root/KhaiDD/FedCar/abliation_studies/img_sim/sim_{IMAGE_EMBED_SIMILARITY}/" 
    
    keywords = [
    # --- Original & Atmospheric ---
    "sunny", "snowy", "stormy", "arid", "golden hour", "neon-drenched", 
    "midnight", "backlit", "pristine", "ruined", "overgrown", "muddy", 
    "cyberpunk", "vintage", "vibrant", "monochrome", "industrial", 
    "tropical", "volcanic", "underground", "foggy", "hazy", "sandstorm", 
    "overcast", "twilight", "thunderstorm",

    # --- Artistic & Aesthetic ---
    "steampunk", "synthwave", "minimalist", "brutalist", "ghibli-style", 
    "utopian", "dystopian", "apocalyptic", "sketch", "oil painting",
    "watercolor", "pixel art", "pop art", "surrealist", "gothic", 
    "futuristic", "vaporwave", "graffiti", "noir", "baroque",

    # --- Photography & Cinematic ---
    "cinematic", "motion blur", "bokeh", "fisheye", "infrared", 
    "long exposure", "aerial view", "low-poly", "photorealistic",
    "blue hour", "high noon", "silhouette", "harsh shadows", "soft lighting", 
    "lens flare", "double exposure", "tilt-shift", "hyper-realistic",

    # --- Materials & Textures ---
    "rain-slicked", "rusty", "matte", "glossy", "metallic", "carbon-fiber", 
    "pixelated", "cracked", "dusty", "polished", "frozen", "fluorescent",

    # --- Technical & Sensor Styles ---
    "thermal", "x-ray", "wireframe", "glitch", "satellite", "dashcam", 
    "night vision", "lidar-style", "point cloud", "surveillance", "cctv", "blueprint",

    # --- Environments & Mood ---
    "suburban", "rural", "canyon", "tundra", "rainforest", "arctic", 
    "cityscape", "pastel", "sepia", "high-contrast", "moody", "ethereal"
    ]

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # ================= Step 1 : get conversion matrix =================
    rgb_files = get_rgb_files(IMAGE_FOLDER, NUM_SAMPLES)
    print(f"\n-> Found {len(rgb_files)} total samples to process.")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # find conversion matrix using Orthogonal Procrustes
     

    # get mean  
    mean_img_embed, original_embeds = get_mean_image_embedding(
        rgb_files, CLIP_BATCH_SIZE, clip_model, clip_processor
    )
    
    discovered_prompts = find_domain_prompts(
        mean_img_embed, keywords, CLIP_BATCH_SIZE, NUM_DOMAIN, DOMAIN_ANGLE, MAX_COMBINATIONS, clip_model, clip_processor
    )
    
    prompts = [f"{p} environment" for p in discovered_prompts]

    del clip_model
    del clip_processor
    torch.cuda.empty_cache()

    # ================= step 4: generate image with Controlnet =================
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    CENTER_PROMPT = f"{discovered_prompts[0]} environment"
    
    optimized_embeds = optimize_domain_embeddings(
        pipe=pipe, 
        center_prompt=CENTER_PROMPT, 
        domain_prompts=prompts, 
        target_sim_center=0.8,       
        target_dist_center=15.0,     
        target_dist_between=25.0,    
        steps=100
    )

    def encode_text_sd(prompt_text):
        inputs = pipe.tokenizer(
            prompt_text, padding="max_length", max_length=pipe.tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        ).to(pipe.device)
        with torch.no_grad():
            return pipe.text_encoder(inputs.input_ids)[0]

    print(f"\n================ STARTING DATA PROCESSING ({len(rgb_files)} base samples) ================")
    
    for idx, rgb_path in enumerate(rgb_files):
        basename = os.path.basename(rgb_path)
        prefix = basename.replace('_leftImg8bit.png', '')
        print(f"\n---> [Base Sample {idx + 1}/{len(rgb_files)}] Processing: {basename}")
        
        label_dir = LABEL_FOLDER 
        segment_path = os.path.join(label_dir, f"{prefix}_gtFine_color.png")
        
        edge_path = os.path.join(label_dir, f"{prefix}_gtFine_labelIds.png")
        json_path = os.path.join(label_dir, f"{prefix}_gtFine_polygons.json") 
        
        if not (os.path.exists(segment_path) and os.path.exists(edge_path) and os.path.exists(json_path)):
            print(f"  [!] Missing label files for {basename}, skipping...")
            continue
        
        img_rgb = Image.open(rgb_path).convert("RGB").resize((WIDTH, HEIGHT))
        img_seg = Image.open(segment_path).convert("RGB").resize((WIDTH, HEIGHT), Image.NEAREST)
        label_img_pil = Image.open(edge_path).convert("L").resize((WIDTH, HEIGHT), Image.NEAREST)

        edge_data = process_images_edges([("dummy", edge_path)], h=HEIGHT, w=WIDTH)
        img_edge = edge_data[0]['edge_pil']

        with open(json_path, 'r') as f: 
            poly_data = json.load(f)
        
        orig_w, orig_h = poly_data['imgWidth'], poly_data['imgHeight']
        label_data_dict = {}
        for obj in poly_data['objects']:
            label_name = obj['label']
            mask_pil = Image.new('L', (orig_w, orig_h), 0)
            ImageDraw.Draw(mask_pil).polygon([tuple(p) for p in obj['polygon']], outline=255, fill=255)
            mask_tensor = to_tensor(mask_pil.resize((WIDTH, HEIGHT), Image.NEAREST)).to("cuda")
            
            if label_name in label_data_dict:
                label_data_dict[label_name]['mask'] = torch.clamp(label_data_dict[label_name]['mask'] + mask_tensor, 0, 1)
            else:
                label_data_dict[label_name] = {'mask': mask_tensor}

        for p_idx, base_prompt in enumerate(prompts):
            safe_prompt_name = re.sub(r'[^a-zA-Z0-9]+', '_', discovered_prompts[p_idx]).strip('_').lower()
            
            prompt_out_dir = os.path.join(BASE_OUTPUT_DIR, safe_prompt_name)
            orig_dir = os.path.join(prompt_out_dir, "original")
            gen_dir = os.path.join(prompt_out_dir, "generated")
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(gen_dir, exist_ok=True)
            
            img_rgb.save(os.path.join(orig_dir, f"sample_{idx}_img.png"))
            img_seg.save(os.path.join(orig_dir, f"sample_{idx}_segment.png"))
            label_img_pil.save(os.path.join(orig_dir, f"sample_{idx}_label.png"))
            img_edge.save(os.path.join(orig_dir, f"sample_{idx}_edge.png"))

            print(f"  -> [Domain {p_idx + 1}/{len(prompts)}] Generating {NUM_GEN_PER_PROMPT} images for: '{base_prompt}'")

            current_base_embeds = optimized_embeds[p_idx]
            original_base_embeds = encode_text_sd(base_prompt)
            
            shift_vector = current_base_embeds - original_base_embeds

            for gen_idx in range(NUM_GEN_PER_PROMPT):
                prefix_gen = f"sample_{idx}_gen_{gen_idx}"
                print(f"     ... Generating iteration {gen_idx + 1}/{NUM_GEN_PER_PROMPT}")
                
                generator = torch.manual_seed(idx * 100 + gen_idx) 

                with torch.no_grad():
                    base_output = pipe(
                        prompt_embeds=current_base_embeds,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=img_rgb,
                        control_image=img_seg,
                        strength=STRENGTH,
                        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                        num_inference_steps=NUMBER_INFERENCE_BASE_STEP,
                        height=HEIGHT,
                        width=WIDTH,
                        output_type="pt",
                        generator=generator
                    )
                    final_stitched_image = torch.clamp(base_output.images.to(torch.float32), 0.0, 1.0)

                for label_name, data in label_data_dict.items():
                    mask_pt = data['mask']
                    seg_prompt = f"{base_prompt}, photorealistic {label_name}"
                    
                    orig_seg_embeds = encode_text_sd(seg_prompt)
                    shifted_seg_embeds = orig_seg_embeds + shift_vector
                    
                    with torch.no_grad():
                        seg_out = pipe(
                            prompt_embeds=shifted_seg_embeds, 
                            negative_prompt=NEGATIVE_PROMPT,
                            image=img_rgb,
                            control_image=img_seg,
                            strength=STRENGTH,
                            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                            num_inference_steps=NUMBER_INFERENCE_STEP,
                            height=HEIGHT,
                            width=WIDTH,
                            output_type="pt",
                            generator=generator
                        )
                        gen_seg = torch.clamp(seg_out.images.to(torch.float32), 0.0, 1.0)
                        final_stitched_image = final_stitched_image * (1.0 - mask_pt) + gen_seg * mask_pt

                final_stitched_image = torch.nan_to_num(final_stitched_image, nan=0.0)
                final_pil = to_pil_image(final_stitched_image[0].cpu())
                
                final_pil.save(os.path.join(gen_dir, f"{prefix_gen}_img.png"))
                label_img_pil.save(os.path.join(gen_dir, f"{prefix_gen}_label.png"))
                img_seg.save(os.path.join(gen_dir, f"{prefix_gen}_segment.png"))
                
                final_edges = get_pytorch_edges(final_stitched_image, threshold=INITIAL_EDGE_THRESHOLD)
                to_pil_image(final_edges.squeeze(0).cpu()).save(os.path.join(gen_dir, f"{prefix_gen}_edge.png"))

    # ================= Step 5: T-SNE VISUALIZATION =================
    print("\n================ STARTING T-SNE 3D VISUALIZATION ================")
    
    del pipe
    del controlnet
    torch.cuda.empty_cache()

    print("[t-SNE] Reloading CLIP model to calculate embeddings for generated images...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    all_domain_embeds = {"Original": original_embeds.cpu().numpy()}

    for p_idx, base_prompt in enumerate(prompts):
        safe_prompt_name = re.sub(r'[^a-zA-Z0-9]+', '_', discovered_prompts[p_idx]).strip('_').lower()
        gen_dir = os.path.join(BASE_OUTPUT_DIR, safe_prompt_name, "generated")
        
        gen_files = sorted(glob.glob(os.path.join(gen_dir, "*_img.png")))
        if not gen_files:
            continue
            
        domain_embeds = []
        for img_path in tqdm(gen_files, desc=f"Encoding Generated Domain {p_idx+1}"):
            img = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=[img], return_tensors="pt", padding=True).to("cuda")
            
            with torch.no_grad():
                embeds = clip_model.get_image_features(**inputs)
                if not isinstance(embeds, torch.Tensor):
                    embeds = embeds.image_embeds if hasattr(embeds, "image_embeds") else (embeds.pooler_output if hasattr(embeds, "pooler_output") else embeds[0])
                domain_embeds.append(embeds)
                
        domain_embeds = torch.cat(domain_embeds, dim=0)
        domain_embeds = F.normalize(domain_embeds, p=2, dim=-1)
        
        domain_label = f"Domain {p_idx+1} ({discovered_prompts[p_idx]})"
        all_domain_embeds[domain_label] = domain_embeds.cpu().numpy()

    print("[t-SNE] Running TSNE algorithm (3D)...")
    X_list = []
    labels = []
    for label, emb in all_domain_embeds.items():
        X_list.append(emb)
        labels.extend([label] * emb.shape[0])
        
    X = np.concatenate(X_list, axis=0)
    
    n_samples = X.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = list(all_domain_embeds.keys())
    
    cmap = plt.get_cmap('tab10')
    
    for idx, label in enumerate(unique_labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        
        marker = 'X' if label == "Original" else 'o'
        size = 120 if label == "Original" else 60
        alpha = 1.0 if label == "Original" else 0.7
        
        ax.scatter(
            X_tsne[indices, 0], X_tsne[indices, 1], X_tsne[indices, 2], 
            label=label, color=cmap(idx), marker=marker, 
            s=size, alpha=alpha, edgecolors='k' if label == "Original" else 'none'
        )
        
    ax.set_title(f"t-SNE 3D Distribution of Target vs Generated Domains (Perplexity={perplexity})", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3") 
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    tsne_save_path = os.path.join(BASE_OUTPUT_DIR, "tsne_3d_distribution.png")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    
    ax.view_init(elev=20., azim=45) 
    tsne_save_path_alt = os.path.join(BASE_OUTPUT_DIR, "tsne_3d_distribution_alt_angle.png")
    plt.savefig(tsne_save_path_alt, dpi=300, bbox_inches='tight')

    plt.close(fig)
    
    print(f"\n[COMPLETE] t-SNE 3D Plots saved successfully to: {BASE_OUTPUT_DIR}")
    print("\n[ALL TASKS FINISHED SUCCESSFULLY!]")