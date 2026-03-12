import os
import requests
import zipfile
import argparse
from tqdm import tqdm

def download_file(url, dest_path):
    # if file already exist -> skip
    if os.path.exists(dest_path):
        print(f"[SKIP] {os.path.basename(dest_path)} already exists.")
        return True
    
    print(f"\n[DOWNLOADER] Starting download: {os.path.basename(dest_path)}")
    try:
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
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    print(f"[EXTRACTOR] Unzipping '{zip_path}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted: {os.path.basename(zip_path)}")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is corrupted.")

def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    
    # remeber, each url is separated by a ','
    all_urls = [u.strip() for u in args.urls.split(',')]
    
    # The url list shoule be Part1_Img, Part1_Lbl, Part2_Img, Part2_Lbl...
    if len(all_urls) % 2 != 0:
        print("Warning: You provided an odd number of URLs. Ensure every Image link has a Label link.")

    for i in range(0, len(all_urls), 2):
        part_num = (i // 2) + 1
        img_url = all_urls[i]
        
        img_name = f"gta5_part{part_num}_images.zip"
        img_path = os.path.join(args.dest_dir, img_name)
        if download_file(img_url, img_path):
            extract_zip(img_path, args.dest_dir)

        if i + 1 < len(all_urls):
            lbl_url = all_urls[i+1]
            lbl_name = f"gta5_part{part_num}_labels.zip"
            lbl_path = os.path.join(args.dest_dir, lbl_name)
            if download_file(lbl_url, lbl_path):
                extract_zip(lbl_path, args.dest_dir)
        
        print(f"--- Finished Processing Part {part_num} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dest_dir', 
        type=str, 
        required=True
    )

    parser.add_argument(
        '--urls', 
        type=str, 
        required=True
    )
    args = parser.parse_args()
    main(args)