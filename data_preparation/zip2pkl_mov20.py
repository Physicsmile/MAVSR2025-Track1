import os
import zipfile

import pickle
from pathlib import Path
from tqdm import tqdm
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def encode_images_from_zip(zip_path, source_dir, target_dir, path):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    pkl_filename = Path(target_dir) / Path(path + '.pkl')
    if pkl_filename.exists():
        print(f"{pkl_filename} already exists, skipping...")
        return

    with zipfile.ZipFile(zip_path, 'r') as z:
        img_files = [f for f in z.namelist()]
        img_files.sort(key=lambda x: int(x[:5])) 
        img_data_list = []
        for filename in img_files:
            with z.open(filename) as img_file:
                encoded_img = img_file.read()  
                img_data_list.append(encoded_img)

    Path(pkl_filename).parent.mkdir(parents=True, exist_ok=True)
    print("Saving:", pkl_filename)
    with open(pkl_filename, 'wb') as f:
        pickle.dump(img_data_list, f)

def process_zip_file(uttid, folder_path, target_dir, meta_dir):
    zip_file = Path(folder_path) / Path(uttid + '.zip')
    print("Processing:", zip_file)
    try:
        encode_images_from_zip(zip_file, folder_path, target_dir, uttid)
    except Exception as e:
        with open('mov20_lip_error.log', 'a') as f:
            f.write(f"Error processing {zip_file}: {e}\n")
        print(f"Error processing {zip_file}: {e}")

def process_all_zips(folder_path, target_dir, meta_dir):
    data = load_json(meta_dir)
    for uttid in tqdm(data.items(), total=len(data)):
        process_zip_file(uttid, folder_path, target_dir, meta_dir)

source_dir = "../data/MOV20_zip/lip_imgs_96"
target_dir = "../data/MOV20/lip_imgs_96"
meta_dir = "../data/MOV20/mov20_all_data.csv"

process_all_zips(source_dir, target_dir, meta_dir)