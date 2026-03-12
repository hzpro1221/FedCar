import os
import requests
import zipfile
import argparse
from tqdm import tqdm

def download_file(url, dest_path):
    print(f"\n[DOWNLOADER] Starting download to '{dest_path}'...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)
    print("Download complete!")

def extract_zip(zip_path, extract_to):
    print(f"[EXTRACTOR] Unzipping '{zip_path}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted into: {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is corrupted. The download link may have expired.")

def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    
    images_zip = os.path.join(args.dest_dir, "bdd100k_images_10k.zip")
    labels_zip = os.path.join(args.dest_dir, "bdd100k_sem_seg_labels.zip")
    
    if not os.path.exists(images_zip):
        download_file(args.images_url, images_zip)
    else:
        print(f"Images zip already exists at {images_zip}. Skipping download.")
        
    if not os.path.exists(labels_zip):
        download_file(args.labels_url, labels_zip)
    else:
        print(f"Labels zip already exists at {labels_zip}. Skipping download.")
        
    extract_zip(images_zip, args.dest_dir)
    extract_zip(labels_zip, args.dest_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dest_dir', 
        type=str,
        required=True 
    )
    
    parser.add_argument(
        '--images_url', 
        type=str, 
        required=True
    )
    
    parser.add_argument(
        '--labels_url', 
        type=str, 
        required=True
    )

    args = parser.parse_args()
    main(args)