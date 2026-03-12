import os
import requests
import zipfile
from tqdm import tqdm
import argparse

def download_file(url, dest_path):
    print("\n[DOWNLOADER] Currently downloading mapilary dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)
    print('Download finish!')

def extract_zip(zip_path, save_folder):
    print(f"\n[EXTRACTOR] start extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_folder)
        print(f"Finish extract, saved to {save_folder}")
    except zipfile.BadZipFile:
        print(f"Error, zip file is broken!")

def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)

    data_zip = os.path.join(args.dest_dir, args.zip_name)

    # if download file is already exist -> skip
    if not os.path.exists(data_zip):
        download_file(
            url=args.download_url,
            dest_path=data_zip
        )
    else:
        print("File data is already exist")
    
    # extract
    extract_zip(
        zip_path=data_zip,
        save_folder=args.dest_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dest_dir', 
        type=str,
        required=True 
    )
    
    parser.add_argument(
        '--zip_name', 
        type=str, 
        required=True
    )
    
    parser.add_argument(
        '--download_url', 
        type=str, 
        required=True
    )

    args = parser.parse_args()
    main(args)