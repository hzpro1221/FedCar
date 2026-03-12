import appdirs
import getpass
import hashlib
import json
import os
import requests
import zipfile
import glob
import stat
import argparse  # Added for command-line parsing
from tqdm import tqdm
from builtins import input

def _login():
    appname = 'cityscapes_downloader'
    appauthor = 'cityscapes'
    data_dir = appdirs.user_data_dir(appname, appauthor)
    credentials_file = os.path.join(data_dir, 'credentials.json')

    if os.path.isfile(credentials_file):
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
    else:
        username = input("Cityscapes username or email address: ")
        password = getpass.getpass("Cityscapes password: ")

        credentials = {
            'username': username,
            'password': password
        }

        store_question = f"Store credentials unencrypted in '{credentials_file}' [y/N]: "
        store = input(store_question).strip().lower()
        if store in ['y', 'yes']:
            os.makedirs(data_dir, exist_ok=True)
            with open(credentials_file, 'w') as f:
                json.dump(credentials, f)
            os.chmod(credentials_file, stat.S_IREAD | stat.S_IWRITE)

    session = requests.Session()
    r = session.get("https://www.cityscapes-dataset.com/login", allow_redirects=False)
    r.raise_for_status()
    credentials['submit'] = 'Login'
    r = session.post("https://www.cityscapes-dataset.com/login", data=credentials, allow_redirects=False)
    r.raise_for_status()

    if r.status_code != 302:
        if os.path.isfile(credentials_file):
            os.remove(credentials_file)
        raise Exception("Bad credentials. Please check your username and password.")

    return session

def _parse_size_to_bytes(size_str):
    size_str = size_str.upper()
    if size_str.endswith("KB"): return float(size_str[:-2]) * 1024
    elif size_str.endswith("MB"): return float(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"): return float(size_str[:-2]) * 1024 * 1024 * 1024
    else: raise ValueError("Invalid size format.")

def downloader(package_name, destination_path, resume=True):
    os.makedirs(destination_path, exist_ok=True)
    session = _login()

    r = session.get("https://www.cityscapes-dataset.com/downloads/?list", allow_redirects=False)
    r.raise_for_status()
    packages = r.json()
    
    name_to_id = {p['name']: p['packageID'] for p in packages}
    name_to_bytes = {p['name']: _parse_size_to_bytes(p['size']) for p in packages}
    
    if package_name not in name_to_id:
        raise Exception(f"Package '{package_name}' does not exist or you have not accepted the EULA on the website.")

    local_filename = os.path.join(destination_path, package_name)
    package_id = name_to_id[package_name]

    print(f"\n[DOWNLOADER] Processing package '{package_name}'...")

    url_md5 = f"https://www.cityscapes-dataset.com/md5-sum/?packageID={package_id}"
    r_md5 = session.get(url_md5, allow_redirects=False)
    r_md5.raise_for_status()
    md5sum = r_md5.text.split()[0]

    file_mode = 'ab' if resume and os.path.exists(local_filename) else 'wb'
    initial_size = os.path.getsize(local_filename) if file_mode == 'ab' else 0

    if initial_size < name_to_bytes[package_name]:
        print(f"Downloading data to '{local_filename}'...")
        url_file = f"https://www.cityscapes-dataset.com/file-handling/?packageID={package_id}"
        resume_header = {'Range': f'bytes={initial_size}-'} if initial_size > 0 else {}
        
        with open(local_filename, file_mode) as f:
            with session.get(url_file, allow_redirects=False, stream=True, headers=resume_header) as r_file:
                r_file.raise_for_status()
                with tqdm(total=name_to_bytes[package_name], initial=initial_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for chunk in r_file.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        print("File is already fully downloaded.")

    print("Verifying MD5 Checksum...")
    hash_md5 = hashlib.md5()
    with open(local_filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    if md5sum != hash_md5.hexdigest():
        raise Exception("MD5 sum mismatch. The file might be corrupted, please delete and re-download.")
    print("MD5 checksum valid!")

    if local_filename.endswith('.zip'):
        print(f"Extracting '{local_filename}'...")
        try:
            with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                zip_ref.extractall(destination_path)
            print(f"Successfully extracted to directory: {destination_path}")
        except zipfile.BadZipFile:
            print("Error: Corrupted zip file, cannot extract.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Cityscapes dataset packages.")
    
    parser.add_argument(
        '--dest_dir', 
        type=str, 
        default='/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data',
        help='Destination directory to save and extract the data.'
    )
    
    parser.add_argument(
        '--package', 
        type=str, 
        default='gtFine_trainvaltest.zip',
        help='Name of the Cityscapes package to download.'
    )

    args = parser.parse_args()
    downloader(package_name=args.package, destination_path=args.dest_dir, resume=True)