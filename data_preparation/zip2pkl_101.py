import os
import zipfile
import simplejpeg
import pickle
from pathlib import Path
from tqdm import tqdm

def encode_images_from_zip(zip_path, source_dir, target_dir, path ,start_frame, end_frame):

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    pkl_filename = Path(target_dir) / Path(path+'.pkl')
    if pkl_filename.exists():
        return

    with zipfile.ZipFile(zip_path, 'r') as z:
        frame_pattern = f'%05d'  
        img_files = [f for f in z.namelist() if int(f[:5]) >= start_frame and int(f[:5]) <= end_frame]

        img_files.sort(key=lambda x: int(x[:5]))

        img_data_list = []
        for filename in img_files:
            with z.open(filename) as img_file:
                encoded_img = img_file.read() 
                img_data_list.append(encoded_img)
    Path(pkl_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_filename, 'wb') as f:
        pickle.dump(img_data_list, f)


def process_all_zips(folder_path, target_dir,meta_dir):
    with open(meta_dir, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines,total=len(lines)):
            path,start_time,end_time,durtion,frames,start_frame,end_frame,text = line.strip().split('\t')
            zip_file = Path(folder_path) / Path(path[:-4]+'.zip')
            try:
                encode_images_from_zip(zip_file, folder_path,target_dir,path,int(start_frame),int(end_frame))
            except Exception as e:
                with open('101face_error.log', 'a') as f:
                    f.write(f"Error processing {zip_file}: {e}\n")
                print(f"Error processing {zip_file}: {e}")


source_dir = '../data/CAS-VSR-S101_zip/lip_imgs_112'
target_dir = '../data/CAS-VSR-S101/lip_imgs_112'
meta_dir = "../data/CAS-VSR-S101/train/longform_transcripts.csv"
process_all_zips(source_dir, target_dir,meta_dir)
